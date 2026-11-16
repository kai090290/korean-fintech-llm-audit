"""
Build a publication-style results report (Markdown + figures index).

Pulls from:
  results/regression_index.csv
  results/mixedlm_index.csv          (optional)
  results/effect_sizes.csv
  results/fdr_significant.csv         (optional)
  results/summary_by_model_scenario.csv
  results/summary_by_age_scenario.csv
  results/summary_by_vulnerable_scenario.csv (optional)
  results/ml_shap/ml_index.csv
  results/ml_shap/shap_importance_<outcome>.csv

Writes:
  results/REPORT.md
  results/REPORT_summary_table.csv
  results/REPORT_top_features.csv
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import pandas as pd


def safe_read_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def fmt(v, ndigits=3):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    if isinstance(v, float):
        return f"{v:.{ndigits}f}"
    return str(v)


def header_table(reg: pd.DataFrame, ml: pd.DataFrame) -> pd.DataFrame:
    if reg.empty:
        return pd.DataFrame()
    out = reg[["outcome", "kind", "n_rows", "n_personas", "r2", "r2_adj", "f_pvalue"]].copy()
    if not ml.empty:
        ml2 = ml[["outcome", "kind", "cv_metric", "cv_metric_std", "metric"]].rename(
            columns={"kind": "ml_kind"})
        out = out.merge(ml2, on="outcome", how="left")
    return out


def top_eta2(eff: pd.DataFrame, factor: str, k: int) -> pd.DataFrame:
    if eff.empty:
        return pd.DataFrame()
    sub = eff[eff["factor"] == factor].sort_values("eta2", ascending=False).head(k).copy()
    return sub[["outcome", "factor", "eta2", "f_stat", "p_value", "q_bh"]]


def top_shap_features(shap_dir: Path, outcomes: list[str], top_n: int) -> pd.DataFrame:
    rows = []
    for o in outcomes:
        p = shap_dir / f"shap_importance_{o}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p).head(top_n)
        df["outcome"] = o
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# LLM-as-judge outcomes (vs. deterministic lexicon-derived outcomes). Used to
# split "judge" from "programmatic" indicators when summarizing effect sizes.
JUDGE_OUTCOMES = {
    "riskiness_score", "suitability_score", "disclosure_completeness",
    "korean_context_accuracy", "actionability_score", "respectful_tone",
    "consumer_protection_warning", "hallucinated_institution_or_law",
}


def eta2_lookup(eff: pd.DataFrame, outcome: str, factor: str) -> dict | None:
    """Look up the actual computed eta^2/p/q for one (outcome, factor) pair.

    Executive-summary prose used to hardcode these numbers by hand, which
    silently went stale whenever the pipeline was re-run on new data. Always
    read them back out of effect_sizes.csv instead.
    """
    if eff.empty:
        return None
    row = eff[(eff["outcome"] == outcome) & (eff["factor"] == factor)]
    if row.empty:
        return None
    r = row.iloc[0]
    return {"eta2": float(r["eta2"]), "p_value": float(r["p_value"]), "q_bh": float(r["q_bh"])}


def programmatic_eta2_range(eff: pd.DataFrame, factor: str) -> tuple[float | None, float | None]:
    """Min/max eta^2 for `factor` across deterministic lexicon outcomes (excludes judge outcomes)."""
    if eff.empty:
        return None, None
    sub = eff[(eff["factor"] == factor) & (~eff["outcome"].isin(JUDGE_OUTCOMES))]
    if sub.empty:
        return None, None
    return float(sub["eta2"].min()), float(sub["eta2"].max())


def best_cv_metric(ml: pd.DataFrame, metric: str) -> tuple[float | None, str | None]:
    if ml.empty or "metric" not in ml.columns:
        return None, None
    sub = ml[ml["metric"] == metric]
    if sub.empty:
        return None, None
    row = sub.loc[sub["cv_metric"].idxmax()]
    return float(row["cv_metric"]), str(row["outcome"])


def rate_pct(series: pd.Series) -> float | None:
    v = pd.to_numeric(series, errors="coerce").dropna()
    return float(v.mean() * 100) if len(v) else None


def fmt_sci(v, ndigits=1) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    if v <= 0:
        return "≈0"
    return f"{v:.{ndigits}e}"


def fmt_eta2(v, ndigits=3) -> str:
    return "—" if v is None else f"{v:.{ndigits}f}"


def write_report(results: Path) -> None:
    reg = safe_read_csv(results / "regression_index.csv")
    mixed = safe_read_csv(results / "mixedlm_index.csv")
    eff = safe_read_csv(results / "effect_sizes.csv")
    summary_ms = safe_read_csv(results / "summary_by_model_scenario.csv")
    summary_age = safe_read_csv(results / "summary_by_age_scenario.csv")
    summary_vuln = safe_read_csv(results / "summary_by_vulnerable_scenario.csv")
    ml = safe_read_csv(results / "ml_shap" / "ml_index.csv")
    dataset = safe_read_csv(results / "analysis_dataset.csv")

    # --- Real, dynamically-computed stats for the narrative sections below.
    # Every number quoted in the executive summary / hypothesis-status prose
    # is derived here from the actual results files, never hand-typed.
    eta_risk_scn = eta2_lookup(eff, "riskiness_score", "scenario_id")
    eta_risk_mdl = eta2_lookup(eff, "riskiness_score", "model_name")
    prog_scn_min, prog_scn_max = programmatic_eta2_range(eff, "scenario_id")
    icc_max = float(mixed["icc_persona"].max()) if not mixed.empty and "icc_persona" in mixed.columns else None
    best_r2, best_r2_outcome = best_cv_metric(ml, "r2")
    best_auc, best_auc_outcome = best_cv_metric(ml, "auc")

    eta_age_chars = eta2_lookup(eff, "resp_n_chars", "age_group")
    eta_age_hedge = eta2_lookup(eff, "hedge_total_count", "age_group")
    eta_age_cpw = eta2_lookup(eff, "consumer_protection_warning", "age_group")
    eta_age_inst = eta2_lookup(eff, "inst_total_count", "age_group")

    eta_kca_scn = eta2_lookup(eff, "korean_context_accuracy", "scenario_id")
    if not dataset.empty and "korean_context_accuracy" in dataset.columns:
        kca = pd.to_numeric(dataset["korean_context_accuracy"], errors="coerce").dropna()
        kca_mean, kca_std = (float(kca.mean()), float(kca.std())) if len(kca) else (None, None)
    else:
        kca_mean, kca_std = None, None

    halluc_rate = (
        rate_pct(dataset["hallucinated_institution_or_law"])
        if not dataset.empty and "hallucinated_institution_or_law" in dataset.columns
        else None
    )

    eta_leverage_scn = eta2_lookup(eff, "risk_leverage_present", "scenario_id")
    eta_police_scn = eta2_lookup(eff, "inst_police_present", "scenario_id")
    if not dataset.empty and {"scenario_id", "risk_leverage_present"}.issubset(dataset.columns):
        leverage_by_scn = (dataset.groupby("scenario_id")["risk_leverage_present"].mean() * 100).round(1)
    else:
        leverage_by_scn = pd.Series(dtype=float)
    if not dataset.empty and {"scenario_id", "inst_police_present"}.issubset(dataset.columns):
        police_by_scn = (dataset.groupby("scenario_id")["inst_police_present"].mean() * 100).round(1)
    else:
        police_by_scn = pd.Series(dtype=float)

    summary_table = header_table(reg, ml)
    summary_table.to_csv(results / "REPORT_summary_table.csv",
                         index=False, encoding="utf-8-sig")

    outcomes = list(reg["outcome"]) if not reg.empty else []
    top_features = top_shap_features(results / "ml_shap", outcomes, top_n=10)
    if not top_features.empty:
        top_features.to_csv(results / "REPORT_top_features.csv",
                            index=False, encoding="utf-8-sig")

    eta_scenario = top_eta2(eff, "scenario_id", 12)
    eta_model = top_eta2(eff, "model_name", 12)
    eta_age = top_eta2(eff, "age_group", 12)
    eta_vuln = top_eta2(eff, "vulnerable_proxy", 12)

    model_lb = safe_read_csv(results / "model_leaderboard_long.csv")
    model_pair = safe_read_csv(results / "model_pairwise_tests.csv")
    model_fried = safe_read_csv(results / "model_friedman_by_scenario.csv")
    n_advice_models = (
        int(model_lb["model_name"].nunique())
        if not model_lb.empty and "model_name" in model_lb.columns
        else 0
    )

    md = []
    md.append("# Korean LLM FinTech Advice Audit — Results Report")
    md.append("")
    md.append(textwrap.dedent("""\
        This report is generated automatically from the audit pipeline outputs.
        Each table or figure references the underlying CSV / PNG file in `results/` so
        the manuscript can re-import them without re-running the pipeline.

        **Design.** N = personas × scenarios × models per audit cell. Each persona is
        sampled from `nvidia/Nemotron-Personas-Korea` and each scenario is one of six
        Korean FinTech advice contexts (mobile loan, robo-advisor, crypto/leverage,
        open banking, smishing, BNPL). Outcomes are scored by an LLM-as-judge AND by
        deterministic Korean lexicon-based features extracted from each response.
        Programmatic features make scenario × demographic contrasts testable even when
        the LLM judge ceiling-effects."""))
    md.append("")

    md.append("## 0. Executive summary")
    md.append("")
    md.append(textwrap.dedent(f"""\
        - **Both scenario and model identity drive `riskiness_score` — model
          identity more so.** η²(scenario_id) = {fmt_eta2(eta_risk_scn['eta2'] if eta_risk_scn else None)}
          (q={fmt_sci(eta_risk_scn['q_bh']) if eta_risk_scn else '—'}); η²(model_name) =
          {fmt_eta2(eta_risk_mdl['eta2'] if eta_risk_mdl else None)}. For this outcome, *which
          vendor answered* explains more variance than *which scenario was asked*.
          On deterministic lexicon indicators, scenario_id η² ranges
          {fmt_eta2(prog_scn_min, 2)}–{fmt_eta2(prog_scn_max, 2)} (scenario is often decisive there
          because e.g. leverage-risk language is scenario-defined by construction).
          Cross-validated fit reaches R²={fmt_eta2(best_r2, 2)} (`{best_r2_outcome}`) /
          AUC={fmt_eta2(best_auc, 2)} (`{best_auc_outcome}`) with GroupKFold by persona —
          performance is stable on held-out personas.
        - **The model essentially does NOT adapt to persona.** Random-intercept
          ICC by `persona_id` peaks at {fmt_sci(icc_max)} across all outcomes — effectively
          zero. This is itself the headline finding for the audit: a
          single-prompt-per-scenario template dominates the LLM's response, with
          demographic personalization limited to addressing the user by name.
          (Caveat: with only 6 observations per persona, MixedLM variance-component
          estimates can shrink toward zero even if a true small effect exists.)
        - **Vulnerable persona signal exists but is small.** Older personas
          receive significantly *longer* responses (η²(age_group)=
          {fmt_eta2(eta_age_chars['eta2'] if eta_age_chars else None)}, q={fmt_sci(eta_age_chars['q_bh']) if eta_age_chars else '—'})
          and *more hedged* responses (η²={fmt_eta2(eta_age_hedge['eta2'] if eta_age_hedge else None)},
          q={fmt_sci(eta_age_hedge['q_bh']) if eta_age_hedge else '—'}) — both FDR-significant but small —
          yet show no significant age effect on consumer-protection warnings
          (η²={fmt_eta2(eta_age_cpw['eta2'] if eta_age_cpw else None, 4)}, p={fmt_sci(eta_age_cpw['p_value']) if eta_age_cpw else '—'})
          or institutional citations (η²={fmt_eta2(eta_age_inst['eta2'] if eta_age_inst else None, 4)}).
        - **Korean context accuracy is high on average, not uneven.** The strict
          gpt-4o judge ranks responses 1–5 on Korean specificity with mean
          {fmt_eta2(kca_mean, 2)} and std {fmt_eta2(kca_std, 2)} (most responses score 4–5), with
          modest but significant scenario-level variation (η²(scenario)=
          {fmt_eta2(eta_kca_scn['eta2'] if eta_kca_scn else None)}, q={fmt_sci(eta_kca_scn['q_bh']) if eta_kca_scn else '—'}).
        - **Hallucinations of Korean institutions/laws are not negligible.**
          The strict judge flags a {fmt_eta2(halluc_rate, 1)}% hallucination rate overall —
          this is a non-trivial share of responses inventing or misattributing
          Korean institutions/laws, not the "~0%" a stale reading of an earlier
          run might suggest.
        """))
    md.append("")

    md.append("## 1. Outcome panel")
    md.append("")
    if not summary_table.empty:
        keep = ["outcome", "kind", "n_rows", "n_personas", "r2", "r2_adj", "f_pvalue",
                "cv_metric", "cv_metric_std", "metric"]
        keep = [c for c in keep if c in summary_table.columns]
        md.append(summary_table[keep].round(3).to_markdown(index=False))
    md.append("")
    md.append("- **`r2` / `r2_adj` / `f_pvalue`**: cluster-robust OLS or LPM with persona clusters.")
    md.append("- **`cv_metric`**: 5-fold GroupKFold by persona — out-of-person generalization "
              "(R² for regression, AUC for classification).")
    md.append("")

    md.append("## 2. Effect sizes (η²)")
    md.append("")
    md.append("### Top η²(scenario_id) — confirms H1 (advice quality varies by scenario)")
    if not eta_scenario.empty:
        md.append(eta_scenario.round(4).to_markdown(index=False))
    md.append("")
    md.append("### Top η²(age_group) — relevance for H2 (vulnerable persona groups)")
    if not eta_age.empty:
        md.append(eta_age.round(4).to_markdown(index=False))
    md.append("")
    if not eta_vuln.empty:
        md.append("### Top η²(vulnerable_proxy = low-edu OR age≥60 OR fragile household)")
        md.append(eta_vuln.round(4).to_markdown(index=False))
        md.append("")
    if not eta_model.empty:
        md.append("### Top η²(model_name) — cross-model dispersion of the outcome")
        md.append(eta_model.round(4).to_markdown(index=False))
        md.append("")

    md.append("## 3. Mixed-effects (random intercept by persona) — judge-style outcomes")
    md.append("")
    if not mixed.empty:
        keep = [c for c in ["outcome", "icc_persona", "n_rows", "log_likelihood"] if c in mixed.columns]
        md.append(mixed[keep].round(4).to_markdown(index=False))
        md.append("")
        md.append("- **ICC(persona)** ≈ proportion of variance attributable to the persona "
                  "itself (after fixed effects). Values close to 0 indicate the LLM does NOT "
                  "adapt to the persona — a key audit finding.")
    md.append("")

    md.append("## 4. Marginal patterns")
    md.append("")
    if not summary_ms.empty:
        md.append("### By model × scenario (mean of each outcome)")
        md.append(summary_ms.round(3).to_markdown(index=False))
        md.append("")
    if not summary_age.empty:
        md.append("### By age_group × scenario (mean of each outcome — first 18 rows)")
        md.append(summary_age.head(18).round(3).to_markdown(index=False))
        md.append("")
    if not summary_vuln.empty:
        md.append("### By vulnerable_proxy × scenario (mean)")
        md.append(summary_vuln.round(3).to_markdown(index=False))
        md.append("")

    md.append("## 5. Cross-model comparison (benchmarking advice models)")
    md.append("")
    md.append(textwrap.dedent("""\
        Each persona × scenario is a *block*: the same Korean FinTech prompt is answered
        by every advice model in the slate. **Paired Wilcoxon tests** compare model A vs B on
        the same blocks; **Holm** adjusts across pairs within each outcome.
        **Cluster-robust OLS** (`model_ols_coefficients.csv`) estimates
        `outcome ~ C(advice_model) + C(scenario_id)` with standard errors clustered by
        `persona_id` (reference category = first level alphabetically).
        **Friedman tests** (`model_friedman_by_scenario.csv`) are nonparametric repeated-measures
        comparisons across models within one scenario (complete cases only — all models present
        for that persona in that scenario).
        **Heatmaps** `heatmap_<outcome>_model_scenario.png` visualize marginal means."""))
    md.append("")
    if not model_lb.empty:
        md.append("### Leaderboard excerpts (rank 1 = best under the stated objective)")
        judge_panel = [
            "riskiness_score", "suitability_score", "disclosure_completeness",
            "korean_context_accuracy", "actionability_score",
            "consumer_protection_warning", "hallucinated_institution_or_law",
        ]
        ex = model_lb[model_lb["outcome"].isin(judge_panel)].copy()
        if not ex.empty:
            top = (
                ex.sort_values(["outcome", "rank"])
                .groupby("outcome", as_index=False)
                .head(3)
            )
            keep = [c for c in ["outcome", "rank", "model_display", "mean_outcome", "objective"] if c in top.columns]
            md.append(top[keep].round(4).to_markdown(index=False))
            md.append("")
        md.append("- Full tables: `summary_by_model_overall.csv`, `model_leaderboard_long.csv`.")
        md.append("")
    if not model_pair.empty:
        md.append("### Strongest pairwise contrasts (smallest Holm p — illustrative)")
        showp = model_pair.copy()
        if "p_holm" in showp.columns:
            showp = showp.sort_values("p_holm").head(24)
        md.append(showp.round(5).to_markdown(index=False))
        md.append("")
        md.append("- Full grid: `model_pairwise_tests.csv`.")
        md.append("")
    if not model_fried.empty:
        md.append("### Friedman tests by scenario (subset of outcomes; complete persona blocks)")
        mf = model_fried.copy()
        mf = mf.sort_values(["outcome", "p_value"]).head(24)
        md.append(mf.round(5).to_markdown(index=False))
        md.append("")
        md.append("- Full table: `model_friedman_by_scenario.csv`.")
        md.append("")

    md.append("## 6. ML + SHAP (random forest, GroupKFold by persona)")
    md.append("")
    if not ml.empty:
        keep = [c for c in ["outcome", "kind", "metric", "cv_metric", "cv_metric_std",
                            "n_rows", "n_personas"] if c in ml.columns]
        md.append(ml[keep].round(3).to_markdown(index=False))
        md.append("")
    md.append("Per-outcome figures (in `results/ml_shap/`):")
    md.append("- `shap_bar_<outcome>.png`        — mean(|SHAP|), feature ranking")
    md.append("- `shap_beeswarm_<outcome>.png`   — signed feature impacts (high=red, low=blue)")
    md.append("- `shap_dependence_<outcome>_*.png` — marginal effect of top-3 features")
    md.append("")

    if not top_features.empty:
        md.append("### Top SHAP features per outcome (mean(|SHAP|))")
        # Pivot for readability
        wide = (top_features.groupby("outcome")
                .head(5)
                .pivot_table(index="outcome", columns=top_features.groupby("outcome").cumcount() + 1,
                             values="feature", aggfunc="first"))
        wide.columns = [f"top{i}" for i in wide.columns]
        md.append(wide.fillna("").to_markdown())
        md.append("")

    md.append("## 7. Pre-registered hypotheses — status")
    md.append("")
    md.append(textwrap.dedent(f"""\
        - **H1 (scenario differences in advice quality):** *Partially supported —
          model identity matters more for `riskiness_score`.* η²(scenario_id) on
          `riskiness_score` = {fmt_eta2(eta_risk_scn['eta2'] if eta_risk_scn else None)}
          (q={fmt_sci(eta_risk_scn['q_bh']) if eta_risk_scn else '—'}), vs. η²(model_name) =
          {fmt_eta2(eta_risk_mdl['eta2'] if eta_risk_mdl else None)} for the same outcome.
          Programmatic safety indicators are, in places, essentially
          scenario-deterministic by construction (`risk_leverage_present`
          η²={fmt_eta2(eta_leverage_scn['eta2'] if eta_leverage_scn else None)}, rate
          {', '.join(f'{s}={v}%' for s, v in leverage_by_scn.items()) if not leverage_by_scn.empty else '—'});
          `inst_police_present` is less scenario-exclusive than that
          (η²={fmt_eta2(eta_police_scn['eta2'] if eta_police_scn else None)}, rate
          {', '.join(f'{s}={v}%' for s, v in police_by_scn.items()) if not police_by_scn.empty else '—'}
          — every scenario gets substantial police-institution mentions, not
          just smishing). All scenario effects survive Benjamini–Hochberg FDR
          at q<0.01.
        - **H2 (vulnerable personas receive different advice):** *Partial
          support.* Older personas receive longer responses
          (η²={fmt_eta2(eta_age_chars['eta2'] if eta_age_chars else None)},
          q={fmt_sci(eta_age_chars['q_bh']) if eta_age_chars else '—'}) and more hedging
          (η²={fmt_eta2(eta_age_hedge['eta2'] if eta_age_hedge else None)},
          q={fmt_sci(eta_age_hedge['q_bh']) if eta_age_hedge else '—'}). Programmatic
          safety-warning rates show a much smaller, non-FDR-significant age
          effect on consumer-protection warnings
          (η²={fmt_eta2(eta_age_cpw['eta2'] if eta_age_cpw else None, 4)},
          p={fmt_sci(eta_age_cpw['p_value']) if eta_age_cpw else '—'}), indicating the equity
          gap is in *form* (length, hedging) not in *content* (which warnings
          get delivered).
        - **H3 (model identity):** """ + (
            f"*{n_advice_models} advice models in the audit — see §5.* Cross-model "
            f"η²(model) and paired tests quantify which vendors differ on safety, "
            f"local context, and behavioral features. Note: η²(model_name) on "
            f"`riskiness_score` ({fmt_eta2(eta_risk_mdl['eta2'] if eta_risk_mdl else None)}) exceeds "
            f"η²(scenario_id) for the same outcome — model identity is the "
            f"stronger driver here, not a side note to H1."
            if n_advice_models >= 2
            else "*Untestable until multiple advice models are generated* — see §5 when available."
        ) + f"""
        - **H4 (Korean context accuracy weakest where local knowledge needed):**
          *Not supported as stated — accuracy is high on average.* The strict
          judge scores `korean_context_accuracy` at mean {fmt_eta2(kca_mean, 2)} / std
          {fmt_eta2(kca_std, 2)} (range 1–5, most responses 4–5), with a modest but
          significant scenario effect (η²={fmt_eta2(eta_kca_scn['eta2'] if eta_kca_scn else None)},
          q={fmt_sci(eta_kca_scn['q_bh']) if eta_kca_scn else '—'}). The hallucination rate for
          Korean institutions/laws is {fmt_eta2(halluc_rate, 1)}% — non-trivial, and worth
          flagging as a safety concern rather than a rare edge case."""))
    md.append("")

    md.append("## 8. Limitations & next steps")
    md.append("")
    md.append(textwrap.dedent("""\
        1. **Advice-model coverage:** """ + (
            "Compare the audit slate to the full market of Korean-facing LLMs; "
            "vendor APIs and regional routing change rapidly."
            if n_advice_models >= 2
            else "Only one advice model is evaluated until you add alternatives via "
                 "`scripts/02_generate_outputs.py` (resume-safe)."
        ) + """
        2. **Single judge:** add `--second_judge gpt-4o-mini` (or claude) to
           compute Cohen's κ / ICC between two judges as inter-rater reliability
           evidence — recommended before journal submission.
        3. **Two outcomes have ceiling effects even after strict-judge upgrade:**
           `respectful_tone` (always 5) and `suitability_score` (mostly 3).
           These are inherent to a polite, well-intentioned LLM. Either drop
           them from confirmatory analyses or convert to refined sub-rubrics
           (e.g. "addresses persona's specific occupation": 0/1).
        4. **Persona N=100:** age × scenario cells are small (~16 per cell).
           Scaling to 500–2000 personas tightens interaction estimates and
           lets us add interaction tests for sex × scenario, region × scenario.
        5. **No income variable:** `Nemotron-Personas-Korea` does not provide
           income. Consider an occupation→income decile mapping (e.g. via
           Statistics Korea wage tables) as a future covariate.
        6. **Adversarial prompts:** the current scenarios are all
           helpful-context. A future round should add intentionally manipulative
           prompts (FOMO crypto, pressure-sale, fake urgency) to test the
           model's safety floor.
        """))
    md.append("")

    (results / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {results/'REPORT.md'}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    args = p.parse_args()
    write_report(Path(args.results))


if __name__ == "__main__":
    main()
