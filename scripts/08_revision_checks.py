"""
Revision checks requested in conference feedback (2026-11).

1. RQ1 regression: outcome ~ C(model_name) + C(scenario_id) + demographics, persona-clustered
   SEs; report the global Wald test and a *demographics-only* joint Wald test separately,
   together with the rank of the cluster covariance and the test degrees of freedom.
2. Effect sizes: one-way eta^2 (what effect_sizes.csv reports) next to partial eta^2 from
   an additive model (model + scenario + age_group, Type II sums of squares).
3. Model contrasts with 95% CIs: OLS on C(model_name) + C(scenario_id), persona-clustered.
4. Sample bookkeeping: missing generations / judge scores; second-judge missing values.
5. Ceiling shares (proportion of responses at the top/bottom of each scale, by model).
6. GroupKFold random-forest baseline using only model_name + scenario_id.

Usage
-----
    .venv/bin/python scripts/08_revision_checks.py --outdir results/revision
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")

RUBRIC = [
    "riskiness_score",
    "suitability_score",
    "disclosure_completeness",
    "korean_context_accuracy",
    "actionability_score",
    "hallucinated_institution_or_law",
]
DEMOGRAPHICS = [
    "age_group", "std_sex", "std_region", "std_education",
    "std_household", "occupation_bucket", "vulnerable_proxy",
]
REF_MODEL = "openrouter/anthropic/claude-opus-4.6"


def short(m: str) -> str:
    return m.replace("openrouter/", "")


def demo_terms(df: pd.DataFrame) -> list[str]:
    return [f"C({c})" for c in DEMOGRAPHICS if c in df.columns and 2 <= df[c].nunique() <= 25]


def joint_wald(res, prefixes: tuple[str, ...]):
    names = res.params.index
    idx = [i for i, n in enumerate(names) if n.startswith(prefixes)]
    R = np.zeros((len(idx), len(names)))
    for r, i in enumerate(idx):
        R[r, i] = 1.0
    w = res.wald_test(R, scalar=True, use_f=True)
    V = res.cov_params().values[np.ix_(idx, idx)]
    rank = int(np.linalg.matrix_rank(V))
    return float(w.statistic), int(w.df_num), float(w.df_denom), float(w.pvalue), len(idx), rank


def rq1_regressions(df: pd.DataFrame) -> pd.DataFrame:
    demo = demo_terms(df)
    rows = []
    for y in RUBRIC + ["scam_no_click_present"]:
        sub = df.dropna(subset=[y]).copy()
        groups = sub["persona_id"].astype(str)
        for spec, rhs in [
            ("original (demo + scenario)", demo + ["C(scenario_id)"]),
            ("revised (model + scenario + demo)", ["C(model_name)", "C(scenario_id)"] + demo),
        ]:
            res = smf.ols(f"{y} ~ {' + '.join(rhs)}", data=sub).fit(
                cov_type="cluster", cov_kwds={"groups": groups}
            )
            k = len(res.params) - 1
            V = res.cov_params().values[1:, 1:]
            row = {
                "outcome": y, "spec": spec, "n": int(res.nobs),
                "n_clusters": groups.nunique(), "n_params_ex_const": k,
                "rank_cluster_cov": int(np.linalg.matrix_rank(V)),
                "r2": res.rsquared, "r2_adj": res.rsquared_adj,
                "global_F_p_statsmodels": float(res.f_pvalue),
            }
            demo_prefix = tuple(t[:-1] for t in demo)  # "C(age_group" etc.
            F, d1, d2, p, nr, rk = joint_wald(res, demo_prefix)
            row.update({"demo_F": F, "demo_df_num": d1, "demo_df_denom": d2, "demo_p": p,
                        "demo_restrictions": nr, "demo_cov_rank": rk})
            if "model" in spec:
                F, d1, d2, p, nr, rk = joint_wald(res, ("C(model_name)",))
                row.update({"model_F": F, "model_df_num": d1, "model_p": p})
                # R2 gain from demographics over model + scenario
                base = smf.ols(f"{y} ~ C(model_name) + C(scenario_id)", data=sub).fit()
                row["r2_model_scenario_only"] = base.rsquared
                row["delta_r2_demographics"] = res.rsquared - base.rsquared
            rows.append(row)
    out = pd.DataFrame(rows)
    # BH-FDR across the demographic joint tests of the revised specification
    rev = out["spec"].str.startswith("revised")
    from statsmodels.stats.multitest import multipletests
    out.loc[rev, "demo_q_bh"] = multipletests(out.loc[rev, "demo_p"], method="fdr_bh")[1]
    return out


PARSIMONIOUS = ["age_group", "std_sex", "std_education", "vulnerable_proxy"]


def persona_level_demographic_tests(df: pd.DataFrame, n_perm: int = 2000, seed: int = 11) -> pd.DataFrame:
    """Demographics vary only between personas, so test them on 100 persona-level means of the
    residuals from y ~ C(model_name) + C(scenario_id). The persona-clustered Wald test with ~63
    persona-level dummies and 100 clusters is near-singular; here we report a classical F test
    and a permutation p-value (demographic rows shuffled across personas)."""
    rng = np.random.default_rng(seed)
    persona = df.drop_duplicates("persona_id").set_index("persona_id")
    rows = []
    for y in RUBRIC + ["scam_no_click_present"]:
        sub = df.dropna(subset=[y]).copy()
        base = smf.ols(f"{y} ~ C(model_name) + C(scenario_id)", data=sub).fit()
        sub["resid"] = base.resid
        pm = sub.groupby("persona_id")["resid"].mean().rename("resid").to_frame()
        for label, cols in [("all 7 demographics", DEMOGRAPHICS), ("age, sex, education, vulnerable", PARSIMONIOUS)]:
            cols = [c for c in cols if c in persona.columns]
            d = pm.join(persona[cols])
            rhs = " + ".join(f"C({c})" for c in cols)
            fit = smf.ols(f"resid ~ {rhs}", data=d).fit()
            r2 = fit.rsquared
            perm = []
            for _ in range(n_perm):
                dp = d.copy()
                dp[cols] = d[cols].sample(frac=1.0, random_state=int(rng.integers(1e9))).to_numpy()
                perm.append(smf.ols(f"resid ~ {rhs}", data=dp).fit().rsquared)
            rows.append({
                "outcome": y, "demographic_set": label, "n_personas": int(fit.nobs),
                "df_model": int(fit.df_model), "df_resid": int(fit.df_resid),
                "r2_persona_level": r2, "F": float(fit.fvalue), "F_p": float(fit.f_pvalue),
                "perm_mean_r2_null": float(np.mean(perm)),
                "perm_p": (1 + np.sum(np.array(perm) >= r2)) / (n_perm + 1),
                "sd_persona_resid": float(pm["resid"].std()),
            })
    out = pd.DataFrame(rows)
    from statsmodels.stats.multitest import multipletests
    for label in out["demographic_set"].unique():
        m = out["demographic_set"] == label
        out.loc[m, "perm_q_bh"] = multipletests(out.loc[m, "perm_p"], method="fdr_bh")[1]
    return out


def eta_squared(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y in RUBRIC:
        sub = df.dropna(subset=[y])
        sst = float(((sub[y] - sub[y].mean()) ** 2).sum())
        full = smf.ols(f"{y} ~ C(model_name) + C(scenario_id) + C(age_group)", data=sub).fit()
        aov = sm.stats.anova_lm(full, typ=2)
        for f in ["model_name", "scenario_id", "age_group"]:
            one = smf.ols(f"{y} ~ C({f})", data=sub).fit()
            ss_eff = float(aov.loc[f"C({f})", "sum_sq"])
            rows.append({
                "outcome": y, "factor": f,
                "eta2_oneway": float(((one.fittedvalues - sub[y].mean()) ** 2).sum()) / sst,
                "partial_eta2_additive": ss_eff / (ss_eff + float(aov.loc["Residual", "sum_sq"])),
            })
    return pd.DataFrame(rows)


def model_contrasts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y in RUBRIC:
        sub = df.dropna(subset=[y]).copy()
        res = smf.ols(
            f"{y} ~ C(model_name, Treatment(reference='{REF_MODEL}')) + C(scenario_id)", data=sub
        ).fit(cov_type="cluster", cov_kwds={"groups": sub["persona_id"].astype(str)})
        ci = res.conf_int()
        for term in res.params.index:
            if "model_name" not in term:
                continue
            m = term.split("[T.")[1].rstrip("]")
            rows.append({
                "outcome": y, "model": short(m), "reference": short(REF_MODEL),
                "diff_vs_ref": res.params[term], "ci_low": ci.loc[term, 0],
                "ci_high": ci.loc[term, 1], "p": res.pvalues[term],
            })
    return pd.DataFrame(rows)


def hallucination_ratio_ci(df: pd.DataFrame, n_boot: int = 2000, seed: int = 7) -> dict:
    """Persona cluster bootstrap for the DeepSeek / Opus hallucination-rate ratio."""
    sub = df.dropna(subset=["hallucinated_institution_or_law"])
    a = "openrouter/deepseek/deepseek-chat"
    b = REF_MODEL
    per = sub.groupby(["persona_id", "model_name"])["hallucinated_institution_or_law"].agg(["sum", "count"])
    per = per.unstack("model_name")
    rng = np.random.default_rng(seed)
    ids = per.index.to_numpy()
    ratios, diffs = [], []
    for _ in range(n_boot):
        s = per.loc[rng.choice(ids, size=len(ids), replace=True)]
        ra = s[("sum", a)].sum() / s[("count", a)].sum()
        rb = s[("sum", b)].sum() / s[("count", b)].sum()
        diffs.append(ra - rb)
        ratios.append(ra / rb if rb > 0 else np.nan)
    ra = per[("sum", a)].sum() / per[("count", a)].sum()
    rb = per[("sum", b)].sum() / per[("count", b)].sum()
    return {
        "rate_deepseek": ra, "rate_opus": rb, "ratio": ra / rb,
        "ratio_ci_low": np.nanpercentile(ratios, 2.5), "ratio_ci_high": np.nanpercentile(ratios, 97.5),
        "diff": ra - rb, "diff_ci_low": np.percentile(diffs, 2.5), "diff_ci_high": np.percentile(diffs, 97.5),
        "events_deepseek": int(per[("sum", a)].sum()), "events_opus": int(per[("sum", b)].sum()),
    }


def ceiling_shares(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m, sub in df.groupby("model_name"):
        row = {"model": short(m), "n": len(sub)}
        for y in ["disclosure_completeness", "korean_context_accuracy", "actionability_score", "suitability_score"]:
            row[f"{y}_share5"] = (sub[y] == 5).mean()
        row["riskiness_score_share1"] = (sub["riskiness_score"] == 1).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def bookkeeping(root: Path, df: pd.DataFrame) -> dict:
    out = pd.read_csv(root / "data/model_outputs.csv")
    out = out[~out["model_name"].str.contains("sonnet-4.5")]
    err = out[out["status"] != "ok"]
    j_err = df[df["korean_context_accuracy"].isna()]
    info = {
        "generated_ok": int((out["status"] == "ok").sum()),
        "generation_errors": err[["persona_id", "scenario_id", "model_name", "error"]].to_dict("records"),
        "judge_errors": j_err[["persona_id", "scenario_id", "model_name", "judge_error"]].to_dict("records"),
    }
    j2 = pd.read_csv(root / "data/validation_judge2.csv")
    miss = j2[j2["hallucinated_institution_or_law"].isna()]
    info["judge2_rows"] = len(j2)
    info["judge2_hallucination_missing"] = len(miss)
    info["judge2_hallucination_missing_by_model"] = miss["model_name"].map(short).value_counts().to_dict()
    info["judge2_hallucination_missing_by_scenario"] = miss["scenario_id"].value_counts().to_dict()
    info["judge2_missing_raw_values"] = miss["judge_status"].value_counts().to_dict()
    return info


def rf_baseline(df: pd.DataFrame) -> pd.DataFrame:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GroupKFold, cross_val_score

    rows = []
    X = pd.get_dummies(df[["model_name", "scenario_id"]].astype(str))
    for y in RUBRIC[:5]:
        mask = df[y].notna()
        est = RandomForestRegressor(n_estimators=500, min_samples_leaf=4, random_state=42, n_jobs=-1)
        sc = cross_val_score(est, X[mask], df.loc[mask, y], groups=df.loc[mask, "persona_id"],
                             cv=GroupKFold(n_splits=5), scoring="r2")
        rows.append({"outcome": y, "features": "model_name + scenario_id only",
                     "cv_r2_mean": sc.mean(), "cv_r2_sd": sc.std()})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/revision")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(root / "results/analysis_dataset.csv")

    rq1 = rq1_regressions(df)
    rq1.to_csv(outdir / "rq1_regression_tests.csv", index=False)
    plt_ = persona_level_demographic_tests(df)
    plt_.to_csv(outdir / "rq1_persona_level_demographic_tests.csv", index=False)
    eta = eta_squared(df)
    eta.to_csv(outdir / "eta_squared_oneway_vs_partial.csv", index=False)
    con = model_contrasts(df)
    con.to_csv(outdir / "model_contrasts_ci.csv", index=False)
    hr = hallucination_ratio_ci(df)
    pd.DataFrame([hr]).to_csv(outdir / "hallucination_ratio_ci.csv", index=False)
    ceil = ceiling_shares(df)
    ceil.to_csv(outdir / "ceiling_shares.csv", index=False)
    rf = rf_baseline(df)
    rf.to_csv(outdir / "rf_baseline_model_scenario_only.csv", index=False)
    book = bookkeeping(root, df)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    print(plt_.round(4).to_string())
    print(book)


if __name__ == "__main__":
    main()
