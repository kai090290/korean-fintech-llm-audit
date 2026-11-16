"""
Measurement-validity and RQ2 checks requested in the mock review (2026-11).

1. Consistency between the primary judge's hallucination flag and its Korean-context score.
2. Within-model association of the Korean-context score with response length and the number
   of institution mentions (separates between-model from within-model association).
3. Credit and installment scenarios (S1, S6): content of low- vs high-scoring responses.
4. RQ2: lexical indicators vs judge scores on the same responses (cross-tabulations).

Outputs: results/revision2/*.csv

Usage:
    python scripts/10_measurement_rq2_checks.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision2"
SHORT = {
    "openrouter/anthropic/claude-opus-4.6": "Opus 4.6",
    "openrouter/deepseek/deepseek-chat": "DeepSeek",
    "openrouter/google/gemini-3-flash-preview": "Gemini 3 Flash",
    "openrouter/moonshotai/kimi-k2.5": "Kimi K2.5",
    "openrouter/openai/gpt-4o": "GPT-4o",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "results/analysis_dataset.csv")
    df = df.dropna(subset=["korean_context_accuracy"]).copy()
    df["model"] = df["model_name"].map(SHORT)
    df["kca_high"] = df["korean_context_accuracy"] >= 4

    # 1. hallucination flag vs Korean-context score
    h = df[df["hallucinated_institution_or_law"] == 1]
    t1 = (h.groupby("model")
            .agg(flagged=("kca_high", "size"), kca_high=("kca_high", "sum"),
                 kca_mean=("korean_context_accuracy", "mean")))
    t1.loc["All"] = [len(h), int(h["kca_high"].sum()), h["korean_context_accuracy"].mean()]
    t1["share_kca_high"] = t1["kca_high"] / t1["flagged"]
    t1.to_csv(OUT / "hallucination_flag_vs_kca.csv")

    # 2. within-model Spearman correlations
    rows = []
    for m, g in df.groupby("model"):
        rows.append({"model": m, "n": len(g), "sd_kca": g["korean_context_accuracy"].std(),
                     "rho_length": g["resp_n_chars"].corr(g["korean_context_accuracy"], method="spearman"),
                     "rho_inst": g["inst_total_count"].corr(g["korean_context_accuracy"], method="spearman"),
                     "mean_chars": g["resp_n_chars"].mean(), "mean_inst": g["inst_total_count"].mean()})
    rows.append({"model": "All (pooled)", "n": len(df), "sd_kca": df["korean_context_accuracy"].std(),
                 "rho_length": df["resp_n_chars"].corr(df["korean_context_accuracy"], method="spearman"),
                 "rho_inst": df["inst_total_count"].corr(df["korean_context_accuracy"], method="spearman"),
                 "mean_chars": df["resp_n_chars"].mean(), "mean_inst": df["inst_total_count"].mean()})
    pd.DataFrame(rows).to_csv(OUT / "kca_length_institution_within_model.csv", index=False)

    # 3. S1/S6 content of low vs high responses
    cols = ["risk_fees_present", "risk_credit_impact_present", "risk_debt_burden_present",
            "inst_fss_present", "inst_total_count", "resp_n_chars",
            "disclosure_completeness", "actionability_score"]
    parts = []
    for sid in ["S1", "S6"]:
        s = df[df["scenario_id"] == sid]
        g = s.groupby(np.where(s["kca_high"], "KCA>=4", "KCA<=3"))[cols].mean()
        g["n"] = s.groupby(np.where(s["kca_high"], "KCA>=4", "KCA<=3")).size()
        g.insert(0, "scenario_id", sid)
        parts.append(g)
    pd.concat(parts).to_csv(OUT / "credit_installment_low_vs_high.csv")
    low = df[df["scenario_id"].isin(["S1", "S6"]) & ~df["kca_high"]]
    low["model"].value_counts().to_csv(OUT / "credit_installment_low_by_model.csv")

    # 4. RQ2 cross-tabulations
    inst = pd.crosstab(df["inst_total_count"] > 0, df["kca_high"], margins=True)
    inst.index = ["no institution mentioned", "institution mentioned", "All"][: len(inst)]
    inst.columns = ["KCA<=3", "KCA>=4", "All"]
    inst.to_csv(OUT / "rq2_institution_mention_vs_kca.csv")
    s5 = df[df["scenario_id"] == "S5"]
    s5.groupby("model")[["scam_no_click_present", "scam_official_channel_present",
                         "scam_report_action_present", "korean_context_accuracy"]].mean() \
      .to_csv(OUT / "rq2_s5_lexicon_vs_kca_by_model.csv")
    nc = pd.crosstab(s5["scam_no_click_present"], s5["kca_high"], margins=True)
    nc.to_csv(OUT / "rq2_s5_noclick_vs_kca.csv")

    pd.set_option("display.width", 200)
    for f in sorted(OUT.glob("*.csv")):
        print("==", f.name)
        print(pd.read_csv(f).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
