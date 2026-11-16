"""
Statistical analysis for the Korean LLM FinTech-advice audit.

What this version does that the prior one did not
-------------------------------------------------
1. Adds a deterministic *response-feature* layer (scripts/03b_response_features.py)
   so the analysis is not dependent on a single LLM judge that ceiling-effects.
   Programmatic outcomes (e.g. mentioned official channel, mentioned KFSA/FSS,
   warned about leverage/principal loss) are tested alongside judge scores.

2. Skips zero-variance outcomes gracefully (no more R² = -inf).

3. Three statistical layers per outcome:
     (a) cluster-robust OLS (linear / linear-probability) with persona clusters,
     (b) mixed-effects model with random intercept by persona_id (statsmodels MixedLM)
         when the outcome is non-binary,
     (c) for *binary* programmatic outcomes, scenario-stratified logistic regression.

4. Effect sizes:
     - within-scenario age-group differences in mean (Cohen's d-like).
     - omega^2 / partial eta^2 for scenario, model, demographics.

5. Multiple-comparison correction:
     - Benjamini–Hochberg FDR across outcome × test family.

6. Persona-aggregated dataset:
     - one row per persona = mean across the 6 scenarios. Solves the
       repeated-measures inflation issue and lets us cleanly attribute
       persona-level differences.

7. Predicted-margins plot for age × scenario.

Outputs
-------
results/analysis_dataset.csv         - merged judge + features + persona table
results/analysis_persona_level.csv   - one row per persona aggregate
results/regression_index.csv         - one row per outcome with R² and notes
results/regression_<outcome>.txt     - cluster-robust OLS / LPM
results/mixedlm_<outcome>.txt        - random-intercept fits (when applicable)
results/logit_<outcome>_S<id>.txt    - scenario-stratified logits
results/effect_sizes.csv             - omega^2 / partial eta^2 / d-like
results/fdr_significant.csv          - all p-values + BH-adjusted q
results/summary_*.csv                - by model/scenario/age/region
results/summary_by_model_overall.csv - marginal mean/std/n per advice model
results/model_leaderboard_long.csv   - per-outcome ranks (best/worst models)
results/model_pairwise_tests.csv    - paired Wilcoxon A vs B (same persona×scenario), Holm
results/model_ols_coefficients.csv  - C(model) coefs vs ref, + C(scenario), cluster by persona
results/model_friedman_by_scenario.csv - nonparametric model difference tests within scenario
results/plot_*.png                   - bar / heatmap / interaction plots
results/heatmap_*_model_scenario.png - model × scenario means for key outcomes

Usage
-----
    .venv/bin/python scripts/04_analyze.py --outdir results
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Outcome registry
# ---------------------------------------------------------------------------

JUDGE_CONTINUOUS = [
    "riskiness_score",
    "suitability_score",
    "disclosure_completeness",
    "korean_context_accuracy",
    "actionability_score",
    "respectful_tone",
]

JUDGE_BINARY = [
    "consumer_protection_warning",
    "hallucinated_institution_or_law",
]

# Deterministic outcomes derived from response_features.csv
PROG_BINARY = [
    "scam_no_click_present",
    "scam_official_channel_present",
    "scam_report_action_present",
    "inst_fss_present",
    "inst_kisa_present",
    "inst_police_present",
    "inst_consumer_agency_present",
    "risk_principal_loss_present",
    "risk_leverage_present",
    "risk_credit_impact_present",
    "risk_fees_present",
    "priv_data_minimization_present",
    "priv_password_present",
]

PROG_CONTINUOUS = [
    "resp_n_chars",
    "resp_n_sentences",
    "resp_n_list_items",
    "resp_n_percent_mentions",
    "scam_total_count",
    "risk_total_count",
    "inst_total_count",
    "priv_total_count",
    "hedge_total_count",
]

ALL_OUTCOMES = JUDGE_CONTINUOUS + JUDGE_BINARY + PROG_BINARY + PROG_CONTINUOUS

# For leaderboard ranking: lower mean is better (safer / fewer errors).
LOWER_IS_BETTER_OUTCOMES = frozenset({"riskiness_score", "hallucinated_institution_or_law"})


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------


def model_display_name(m: str) -> str:
    if not isinstance(m, str):
        return str(m)
    return m.replace("openrouter/", "", 1) if m.startswith("openrouter/") else m

def bucket_rare(s: pd.Series, max_levels: int, other_label: str = "기타") -> pd.Series:
    s = s.fillna(other_label).astype(str).str.strip().replace("", other_label)
    vc = s.value_counts()
    if len(vc) <= max_levels:
        return s
    keep = set(vc.head(max_levels - 1).index)
    return s.where(s.isin(keep), other_label)


def build_analysis_dataset(personas: pd.DataFrame,
                           responses: pd.DataFrame,
                           scores: pd.DataFrame,
                           features: pd.DataFrame) -> pd.DataFrame:
    keys = ["persona_id", "scenario_id", "model_name"]
    df = scores.merge(
        responses[keys + ["scenario_title", "response_text"]],
        on=keys, how="left",
    )
    df = df.merge(features, on=keys, how="left")
    df = df.merge(personas, on="persona_id", how="left")

    for c in ALL_OUTCOMES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    occ = "std_occupation" if "std_occupation" in df.columns else (
        "occupation" if "occupation" in df.columns else None
    )
    if occ is not None:
        df["occupation_bucket"] = bucket_rare(df[occ], max_levels=20)

    if "std_region" in df.columns and df["std_region"].nunique(dropna=True) > 18:
        df["region_bucket"] = bucket_rare(df["std_region"], max_levels=18)

    # Vulnerability proxy: low education OR age>=60 OR single-parent / elderly-alone household.
    edu = df.get("std_education", pd.Series([""] * len(df))).astype(str)
    age_n = pd.to_numeric(df.get("std_age_numeric", df.get("std_age")), errors="coerce")
    hh = df.get("std_household", pd.Series([""] * len(df))).astype(str)
    df["vulnerable_proxy"] = (
        edu.str.contains("초등|중학|무학", na=False)
        | (age_n >= 60)
        | hh.str.contains("혼자|한부모|손자녀", na=False)
    ).astype(int)

    return df


def aggregate_to_persona(df: pd.DataFrame) -> pd.DataFrame:
    """One row per persona = mean over scenarios. Useful to control repeated-measures."""
    keep_persona_cols = [c for c in df.columns
                         if c.startswith("std_") or c in
                         ("persona_id", "age_group", "occupation_bucket", "region_bucket",
                          "marital_status", "vulnerable_proxy")]
    keep_persona_cols = list(dict.fromkeys(keep_persona_cols))
    persona_meta = df[keep_persona_cols].drop_duplicates("persona_id")

    means = (
        df[["persona_id"] + [c for c in ALL_OUTCOMES if c in df.columns]]
        .groupby("persona_id", as_index=False)
        .mean(numeric_only=True)
    )
    return persona_meta.merge(means, on="persona_id", how="inner")


# ---------------------------------------------------------------------------
# Variance & helpers
# ---------------------------------------------------------------------------

def usable_outcomes(df: pd.DataFrame, candidates: list[str], min_std: float = 1e-6) -> list[str]:
    out = []
    for c in candidates:
        if c not in df.columns:
            continue
        v = pd.to_numeric(df[c], errors="coerce").dropna()
        if v.std() <= min_std:
            continue
        out.append(c)
    return out


def fdr_bh(pvals: list[float]) -> list[float]:
    """Benjamini–Hochberg adjusted q-values."""
    pvals = np.array(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return []
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out.tolist()


def candidate_demographics(df: pd.DataFrame) -> list[str]:
    parts = []
    region_col = "region_bucket" if "region_bucket" in df.columns else "std_region"
    for col in ["age_group", "std_sex", region_col, "std_education",
                "std_household", "occupation_bucket", "vulnerable_proxy"]:
        if col not in df.columns or df[col].notna().sum() == 0:
            continue
        nunique = df[col].nunique(dropna=True)
        if 2 <= nunique <= 25:
            parts.append(f"C({col})")
    return parts


# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------

def fit_cluster_ols(df: pd.DataFrame, outcome: str, covars: list[str]):
    rhs = " + ".join(covars + ["C(scenario_id)"])
    sub = df.dropna(subset=[outcome, "persona_id"]).copy()
    formula = f"{outcome} ~ {rhs}"
    fitted = smf.ols(formula, data=sub).fit(
        cov_type="cluster",
        cov_kwds={"groups": sub["persona_id"].astype(str)},
    )
    return fitted, formula, sub


def fit_mixed(df: pd.DataFrame, outcome: str, covars: list[str]):
    sub = df.dropna(subset=[outcome, "persona_id"]).copy()
    rhs = " + ".join(covars + ["C(scenario_id)"])
    formula = f"{outcome} ~ {rhs}"
    md = smf.mixedlm(formula, data=sub, groups=sub["persona_id"].astype(str))
    fitted = md.fit(method="lbfgs", reml=False, disp=False)
    icc = float(fitted.cov_re.iloc[0, 0]) / (
        float(fitted.cov_re.iloc[0, 0]) + float(fitted.scale)
    )
    return fitted, formula, sub, icc


def scenario_stratified_logits(df: pd.DataFrame, outcome: str, covars: list[str], outdir: Path):
    """For binary outcomes: one logit per scenario so age effects can move sign by scenario."""
    rows = []
    for sid, sub in df.groupby("scenario_id"):
        sub = sub.dropna(subset=[outcome, "persona_id"]).copy()
        y = sub[outcome]
        if y.nunique() < 2 or len(sub) < 30:
            continue
        rhs_terms = []
        for term in covars:
            inner = term[2:-1] if term.startswith("C(") else term
            if inner in sub.columns and sub[inner].nunique(dropna=True) >= 2:
                rhs_terms.append(term)
        if not rhs_terms:
            continue
        formula = f"{outcome} ~ " + " + ".join(rhs_terms)
        try:
            res = smf.logit(formula, data=sub).fit(disp=False, maxiter=200)
            with open(outdir / f"logit_{outcome}_{sid}.txt", "w", encoding="utf-8") as f:
                f.write(res.summary().as_text())
            rows.append({
                "outcome": outcome,
                "scenario_id": sid,
                "n": int(res.nobs),
                "pseudo_r2": float(res.prsquared),
                "llr_p": float(res.llr_pvalue),
            })
        except Exception as e:
            rows.append({
                "outcome": outcome,
                "scenario_id": sid,
                "n": len(sub),
                "pseudo_r2": np.nan,
                "llr_p": np.nan,
                "error": repr(e)[:160],
            })
    return rows


def effect_size_table(df: pd.DataFrame, outcomes: list[str], outdir: Path):
    """Partial eta^2 for scenario, model_name, age_group, vulnerable_proxy on each outcome."""
    rows = []
    factors = [c for c in ["scenario_id", "model_name", "age_group", "vulnerable_proxy"]
               if c in df.columns and df[c].nunique(dropna=True) >= 2]
    for outcome in outcomes:
        sub = df.dropna(subset=[outcome])
        if sub[outcome].std() < 1e-6 or len(sub) < 20:
            continue
        for f_ in factors:
            try:
                m = smf.ols(f"{outcome} ~ C({f_})", data=sub).fit()
                ss_between = float(np.sum((m.fittedvalues - sub[outcome].mean()) ** 2))
                ss_total = float(np.sum((sub[outcome] - sub[outcome].mean()) ** 2))
                if ss_total <= 0:
                    continue
                eta2 = ss_between / ss_total
                rows.append({
                    "outcome": outcome,
                    "factor": f_,
                    "n": int(m.nobs),
                    "eta2": eta2,
                    "f_stat": float(m.fvalue),
                    "p_value": float(m.f_pvalue),
                })
            except Exception:
                continue
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_bh"] = fdr_bh(out["p_value"].fillna(1.0).tolist())
        out = out.sort_values(["factor", "q_bh"])
    out.to_csv(outdir / "effect_sizes.csv", index=False, encoding="utf-8-sig")
    return out


# ---------------------------------------------------------------------------
# Cross-model comparison (paired designs: same persona × scenario across models)
# ---------------------------------------------------------------------------

def write_model_overall_summary(df: pd.DataFrame, outdir: Path, outcomes: list[str]) -> pd.DataFrame:
    cols = [c for c in outcomes if c in df.columns]
    if not cols or "model_name" not in df.columns:
        return pd.DataFrame()
    rows: list[dict] = []
    for m, sub in df.groupby("model_name"):
        row: dict = {
            "model_name": m,
            "model_display": model_display_name(str(m)),
            "n_cells": int(len(sub)),
        }
        for c in cols:
            v = pd.to_numeric(sub[c], errors="coerce").dropna()
            row[f"{c}_mean"] = float(v.mean()) if len(v) else np.nan
            row[f"{c}_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(outdir / "summary_by_model_overall.csv", index=False, encoding="utf-8-sig")
    return out


def write_model_leaderboard(df: pd.DataFrame, outdir: Path, outcomes: list[str]) -> pd.DataFrame:
    rows = []
    for oc in outcomes:
        if oc not in df.columns or "model_name" not in df.columns:
            continue
        sub = df[["model_name", oc]].dropna()
        if sub[oc].std() < 1e-9:
            continue
        lower_better = oc in LOWER_IS_BETTER_OUTCOMES
        stats_m = (
            sub.groupby("model_name")[oc]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"mean": "mean_outcome", "std": "std_outcome", "count": "n"})
        )
        stats_m = stats_m.sort_values("mean_outcome", ascending=lower_better)
        stats_m["rank"] = range(1, len(stats_m) + 1)
        stats_m["objective"] = "minimize_mean" if lower_better else "maximize_mean"
        stats_m["outcome"] = oc
        stats_m["model_display"] = stats_m["model_name"].map(model_display_name)
        rows.append(stats_m)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out = out[
        ["outcome", "rank", "model_name", "model_display", "n", "mean_outcome", "std_outcome", "objective"]
    ]
    out.to_csv(outdir / "model_leaderboard_long.csv", index=False, encoding="utf-8-sig")
    return out


def model_pairwise_wilcoxon(df: pd.DataFrame, outcomes: list[str]) -> pd.DataFrame:
    if "persona_id" not in df.columns or "scenario_id" not in df.columns or "model_name" not in df.columns:
        return pd.DataFrame()
    keys = ["persona_id", "scenario_id"]
    rows: list[dict] = []
    for outcome in outcomes:
        if outcome not in df.columns:
            continue
        sub = df[keys + ["model_name", outcome]].dropna(subset=[outcome])
        if sub[outcome].nunique() < 2:
            continue
        try:
            wide = sub.pivot_table(index=keys, columns="model_name", values=outcome, aggfunc="first")
        except Exception:
            continue
        models = sorted(wide.columns.astype(str).tolist())
        if len(models) < 2:
            continue
        pvals: list[float] = []
        meta: list[tuple] = []
        for i, m1 in enumerate(models):
            for m2 in models[i + 1 :]:
                pair = wide[[m1, m2]].dropna()
                if len(pair) < 20:
                    continue
                d = pair[m1].astype(float) - pair[m2].astype(float)
                if np.allclose(d.values, 0):
                    continue
                try:
                    stat, p = stats.wilcoxon(d.values, zero_method="wilcox", alternative="two-sided")
                except Exception:
                    continue
                pvals.append(float(p))
                meta.append((outcome, m1, m2, int(len(pair)), float(stat), float(np.median(d))))
        if not pvals:
            continue
        _, p_holm, _, _ = multipletests(pvals, method="holm")
        for j, p_h in enumerate(p_holm):
            o, m1, m2, n_pair, stat, med = meta[j]
            rows.append({
                "outcome": o,
                "model_a": m1,
                "model_b": m2,
                "model_a_display": model_display_name(m1),
                "model_b_display": model_display_name(m2),
                "n_pairs": n_pair,
                "median_diff_a_minus_b": med,
                "wilcoxon_statistic": stat,
                "p_value": pvals[j],
                "p_holm": float(p_h),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["outcome", "p_holm"])
    return out


def write_model_ols_coefficients(df: pd.DataFrame, outdir: Path, outcomes: list[str]) -> pd.DataFrame:
    rows = []
    needs = {"persona_id", "model_name", "scenario_id"}
    if not needs.issubset(df.columns):
        return pd.DataFrame()
    for outcome in outcomes:
        if outcome not in df.columns:
            continue
        sub = df.dropna(subset=[outcome, "persona_id", "model_name", "scenario_id"]).copy()
        if len(sub) < 50:
            continue
        formula = f"{outcome} ~ C(model_name) + C(scenario_id)"
        try:
            fitted = smf.ols(formula, data=sub).fit(
                cov_type="cluster",
                cov_kwds={"groups": sub["persona_id"].astype(str)},
            )
        except Exception:
            continue
        for term in fitted.params.index:
            if "model_name" not in term:
                continue
            try:
                ci_df = fitted.conf_int()
                clo = float(ci_df.loc[term, 0])
                chi = float(ci_df.loc[term, 1])
            except Exception:
                clo, chi = np.nan, np.nan
            rows.append({
                "outcome": outcome,
                "term": term,
                "coef": float(fitted.params[term]),
                "std_err": float(fitted.bse[term]) if term in fitted.bse.index else np.nan,
                "z_pvalue": float(fitted.pvalues[term]) if term in fitted.pvalues.index else np.nan,
                "ci_low": clo,
                "ci_high": chi,
                "n": int(fitted.nobs),
                "r2": float(fitted.rsquared),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(outdir / "model_ols_coefficients.csv", index=False, encoding="utf-8-sig")
    return out


def model_friedman_by_scenario(df: pd.DataFrame, outcomes: list[str]) -> pd.DataFrame:
    if not {"persona_id", "scenario_id", "model_name"}.issubset(df.columns):
        return pd.DataFrame()
    rows = []
    for outcome in outcomes:
        if outcome not in df.columns:
            continue
        for sid, chunk in df.groupby("scenario_id"):
            sub = chunk[["persona_id", "model_name", outcome]].dropna(subset=[outcome])
            if len(sub) < 30:
                continue
            try:
                wide = sub.pivot_table(index="persona_id", columns="model_name", values=outcome, aggfunc="first")
            except Exception:
                continue
            wide = wide.dropna(how="any")
            if wide.shape[0] < 20 or wide.shape[1] < 2:
                continue
            arrs = [wide[c].values.astype(float) for c in wide.columns]
            try:
                stat, p = stats.friedmanchisquare(*arrs)
            except Exception:
                continue
            rows.append({
                "outcome": outcome,
                "scenario_id": sid,
                "n_personas_complete": int(wide.shape[0]),
                "n_models": int(wide.shape[1]),
                "friedman_chi2": float(stat),
                "p_value": float(p),
            })
    out = pd.DataFrame(rows)
    if not out.empty and len(out) > 1:
        out["q_bh"] = fdr_bh(out["p_value"].fillna(1.0).tolist())
    elif not out.empty:
        out["q_bh"] = out["p_value"]
    if not out.empty:
        out = out.sort_values(["outcome", "p_value"])
    return out


def plot_model_scenario_heatmaps(df: pd.DataFrame, outdir: Path, outcomes: list[str]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib unavailable for model heatmaps: {e}")
        return
    configure_korean_font()
    for outcome in outcomes:
        if outcome not in df.columns or "model_name" not in df.columns:
            continue
        if df[outcome].std() < 1e-9:
            continue
        piv = df.pivot_table(index="model_name", columns="scenario_id", values=outcome, aggfunc="mean")
        if piv.empty or piv.shape[0] < 2:
            continue
        idx = [model_display_name(m) for m in piv.index]
        plt.figure(figsize=(9, max(4, 0.35 * len(idx))))
        im = plt.imshow(piv.values, cmap="RdYlGn" if outcome not in LOWER_IS_BETTER_OUTCOMES else "RdYlGn_r",
                        aspect="auto", vmin=float(np.nanmin(piv.values)), vmax=float(np.nanmax(piv.values)))
        plt.colorbar(im, label=outcome)
        plt.xticks(range(len(piv.columns)), [str(c) for c in piv.columns])
        plt.yticks(range(len(idx)), idx)
        plt.title(f"{outcome}: mean by model × scenario")
        plt.xlabel("scenario_id")
        plt.ylabel("advice model")
        plt.tight_layout()
        plt.savefig(outdir / f"heatmap_{outcome}_model_scenario.png", dpi=180)
        plt.close()


def write_model_comparison_bundle(df: pd.DataFrame, outdir: Path, outcomes: list[str]) -> None:
    """Marginal summaries, paired tests, cluster OLS model terms, and Friedman by scenario."""
    write_model_overall_summary(df, outdir, outcomes)
    write_model_leaderboard(df, outdir, outcomes)
    pair = model_pairwise_wilcoxon(df, outcomes)
    if not pair.empty:
        pair.to_csv(outdir / "model_pairwise_tests.csv", index=False, encoding="utf-8-sig")
    write_model_ols_coefficients(df, outdir, outcomes)
    fried_f = [
        o for o in outcomes
        if o in JUDGE_CONTINUOUS or o in JUDGE_BINARY
        or o in ("scam_no_click_present", "inst_fss_present")
    ]
    fried = model_friedman_by_scenario(df, list(dict.fromkeys(fried_f)))
    if not fried.empty:
        fried.to_csv(outdir / "model_friedman_by_scenario.csv", index=False, encoding="utf-8-sig")

    judge_heat = [
        c for c in [
            "suitability_score",
            "riskiness_score",
            "korean_context_accuracy",
            "hallucinated_institution_or_law",
            "scam_no_click_present",
        ]
        if c in outcomes
    ]
    plot_model_scenario_heatmaps(df, outdir, judge_heat)


# ---------------------------------------------------------------------------
# Summaries & plots
# ---------------------------------------------------------------------------

def write_summaries(df: pd.DataFrame, outdir: Path, outcomes: list[str]) -> None:
    df.groupby(["model_name", "scenario_id"])[outcomes].mean(numeric_only=True) \
        .reset_index().to_csv(outdir / "summary_by_model_scenario.csv", index=False, encoding="utf-8-sig")
    if "age_group" in df.columns:
        df.groupby(["age_group", "scenario_id"])[outcomes].mean(numeric_only=True) \
            .reset_index().to_csv(outdir / "summary_by_age_scenario.csv", index=False, encoding="utf-8-sig")
    if "std_region" in df.columns:
        df.groupby(["std_region"])[outcomes].mean(numeric_only=True) \
            .reset_index().to_csv(outdir / "summary_by_region.csv", index=False, encoding="utf-8-sig")
    if "vulnerable_proxy" in df.columns:
        df.groupby(["vulnerable_proxy", "scenario_id"])[outcomes].mean(numeric_only=True) \
            .reset_index().to_csv(outdir / "summary_by_vulnerable_scenario.csv", index=False, encoding="utf-8-sig")


def configure_korean_font() -> None:
    import platform
    import matplotlib.pyplot as plt
    if platform.system() == "Darwin":
        plt.rcParams["font.sans-serif"] = ["AppleGothic", "Apple SD Gothic Neo", "DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = ["Noto Sans CJK KR", "NanumGothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def make_plots(df: pd.DataFrame, outdir: Path, outcomes: list[str]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib unavailable: {e}")
        return
    configure_korean_font()
    plot_outcomes = [c for c in outcomes if c in df.columns and df[c].std() > 1e-6]

    for outcome in plot_outcomes[:14]:
        means = df.groupby("scenario_id")[outcome].mean(numeric_only=True).sort_index()
        if means.empty:
            continue
        plt.figure(figsize=(7, 4))
        means.plot(kind="bar", color="#4878CF")
        plt.ylabel(outcome)
        plt.xlabel("Scenario")
        plt.title(f"{outcome} by scenario (mean)")
        plt.tight_layout()
        plt.savefig(outdir / f"plot_{outcome}_by_scenario.png", dpi=180)
        plt.close()

    if "age_group" in df.columns:
        for outcome in ["scam_no_click_present", "inst_police_present", "inst_fss_present",
                        "risk_credit_impact_present", "riskiness_score"]:
            if outcome not in df.columns or df[outcome].std() <= 1e-6:
                continue
            piv = df.pivot_table(index="age_group", columns="scenario_id", values=outcome, aggfunc="mean")
            if piv.empty:
                continue
            plt.figure(figsize=(8, 4.5))
            im = plt.imshow(piv.values, cmap="magma", aspect="auto", vmin=0, vmax=max(piv.values.max(), 1e-3))
            plt.colorbar(im, label=outcome)
            plt.xticks(range(len(piv.columns)), piv.columns)
            plt.yticks(range(len(piv.index)), piv.index)
            plt.title(f"{outcome}: age_group × scenario")
            plt.tight_layout()
            plt.savefig(outdir / f"heatmap_{outcome}_age_scenario.png", dpi=180)
            plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_official_models(path: Path) -> set[str] | None:
    """Read the configured advice-model slate (one model per line, '#' comments allowed).

    Used to drop legacy/incomplete model runs (e.g. a model swapped out of the
    slate mid-study) from the analysis so per-model comparisons stay balanced.
    Returns None if the file is missing/empty, meaning "don't filter."
    """
    if not path.exists():
        return None
    models = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            models.add(line)
    return models or None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--personas", default="data/personas_sample.csv")
    parser.add_argument("--responses", default="data/model_outputs.csv")
    parser.add_argument("--scores", default="data/evaluation_scores.csv")
    parser.add_argument("--features", default="data/response_features.csv")
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--advice-models-file", default="config/advice_models.txt",
                        help="Restrict analysis to models listed here (drops legacy/incomplete "
                             "model runs not in the current slate). Pass an empty string to disable.")
    args = parser.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    personas = pd.read_csv(args.personas)
    responses = pd.read_csv(args.responses)
    scores = pd.read_csv(args.scores)
    if not Path(args.features).exists():
        raise SystemExit(f"Missing {args.features}. Run scripts/03b_response_features.py first.")
    features = pd.read_csv(args.features)

    df = build_analysis_dataset(personas, responses, scores, features)

    official_models = (
        load_official_models(Path(args.advice_models_file)) if args.advice_models_file else None
    )
    if official_models is not None:
        present = set(df["model_name"].dropna().unique())
        dropped = sorted(present - official_models)
        if dropped:
            before_rows, before_n = len(df), len(present)
            df = df[df["model_name"].isin(official_models)].copy()
            print(f"Excluded {len(dropped)} model(s) not in {args.advice_models_file}: {dropped} "
                  f"-> removed {before_rows - len(df)} rows "
                  f"({before_n} -> {df['model_name'].nunique()} models kept).")

    df.to_csv(outdir / "analysis_dataset.csv", index=False, encoding="utf-8-sig")

    persona_df = aggregate_to_persona(df)
    persona_df.to_csv(outdir / "analysis_persona_level.csv", index=False, encoding="utf-8-sig")

    judge_outcomes = usable_outcomes(df, JUDGE_CONTINUOUS + JUDGE_BINARY)
    prog_outcomes = usable_outcomes(df, PROG_BINARY + PROG_CONTINUOUS)
    all_used = judge_outcomes + prog_outcomes

    write_summaries(df, outdir, all_used)
    write_model_comparison_bundle(df, outdir, all_used)
    eff = effect_size_table(df, all_used, outdir)
    print(f"Effect-size table: {len(eff)} rows; "
          f"top factors per outcome -> {outdir/'effect_sizes.csv'}")

    covars = candidate_demographics(df)
    fixed_index_rows = []
    mixed_rows = []
    fdr_pvals = []

    for outcome in all_used:
        is_binary = outcome in JUDGE_BINARY or outcome in PROG_BINARY
        try:
            fitted, formula, _ = fit_cluster_ols(df, outcome, covars)
            with open(outdir / f"regression_{outcome}.txt", "w", encoding="utf-8") as f:
                kind = "Linear-probability (binary outcome)" if is_binary else "Cluster-robust OLS"
                f.write(f"# {kind}\n# {formula}\n\n")
                f.write(fitted.summary().as_text())
            fixed_index_rows.append({
                "outcome": outcome,
                "kind": "lpm" if is_binary else "ols",
                "n_rows": int(fitted.nobs),
                "n_personas": int(df.dropna(subset=[outcome])["persona_id"].nunique()),
                "r2": float(fitted.rsquared),
                "r2_adj": float(fitted.rsquared_adj),
                "f_pvalue": float(fitted.f_pvalue) if fitted.f_pvalue is not None else np.nan,
                "formula": formula,
            })
            if fitted.f_pvalue is not None and not np.isnan(fitted.f_pvalue):
                fdr_pvals.append((outcome, "ols_global_F", float(fitted.f_pvalue)))
        except Exception as e:
            fixed_index_rows.append({"outcome": outcome, "kind": "ols", "error": repr(e)[:200]})

        if not is_binary:
            try:
                mfit, mformula, _, icc = fit_mixed(df, outcome, covars)
                with open(outdir / f"mixedlm_{outcome}.txt", "w", encoding="utf-8") as f:
                    f.write(f"# MixedLM (random intercept by persona_id)\n# {mformula}\n# ICC(persona) = {icc:.3f}\n\n")
                    f.write(mfit.summary().as_text())
                mixed_rows.append({
                    "outcome": outcome,
                    "icc_persona": icc,
                    "n_rows": int(mfit.nobs),
                    "log_likelihood": float(mfit.llf),
                })
            except Exception as e:
                mixed_rows.append({"outcome": outcome, "error": repr(e)[:200]})

        if is_binary:
            scenario_stratified_logits(df, outcome, covars, outdir)

    pd.DataFrame(fixed_index_rows).to_csv(
        outdir / "regression_index.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(mixed_rows).to_csv(
        outdir / "mixedlm_index.csv", index=False, encoding="utf-8-sig")

    if fdr_pvals:
        df_p = pd.DataFrame(fdr_pvals, columns=["outcome", "test", "p_value"])
        df_p["q_bh"] = fdr_bh(df_p["p_value"].tolist())
        df_p["significant_q05"] = df_p["q_bh"] < 0.05
        df_p.sort_values("p_value").to_csv(
            outdir / "fdr_significant.csv", index=False, encoding="utf-8-sig")

    make_plots(df, outdir, all_used)
    print(f"Wrote analysis to {outdir}")


if __name__ == "__main__":
    main()
