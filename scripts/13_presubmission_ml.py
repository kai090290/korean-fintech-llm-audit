"""Pre-submission ML recheck (2026-11-23): re-estimate table 10 and figure 7 with
institution mentions counted without double counting (see 12_presubmission_checks.py).

Steps
1. results/presubmission/analysis_dataset_inst_dedup.csv: copy of results/analysis_dataset.csv
   where inst_total_count and inst_fsc_count use the de-duplicated counts from
   results/presubmission/inst_counts_by_response.csv (run 12_presubmission_checks.py first).
2. Table 10: persona-grouped 5-fold CV R² of the random forest for the five 1–5 rubric scores,
   on the original and the corrected data (same functions as 05_ml_shap_analysis.py).
3. Figure 7: mean |SHAP| for Korean-context accuracy on the corrected data, drawn with the
   manuscript figure code (07_manuscript_figures.py, Korean labels).

Run: python scripts/13_presubmission_ml.py      (about 3 minutes)
"""

from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
ml = importlib.import_module("05_ml_shap_analysis")
figs = importlib.import_module("07_manuscript_figures")

OUT = ROOT / "results" / "presubmission"
RUBRIC = ["riskiness_score", "suitability_score", "disclosure_completeness", "korean_context_accuracy", "actionability_score"]
KEY = ["persona_id", "scenario_id", "model_name"]


def build_dedup_dataset() -> Path:
    df = pd.read_csv(ROOT / "results" / "analysis_dataset.csv", low_memory=False)
    c = pd.read_csv(OUT / "inst_counts_by_response.csv")
    m = df.merge(c, on=KEY, how="left")
    has = m["inst_raw"].notna()
    # "금융위원회"/"금융위" is the only overlapping pair, so the removed double counts all belong to the FSC label
    m.loc[has, "inst_fsc_count"] = m.loc[has, "inst_fsc_count"] - (m.loc[has, "inst_raw"] - m.loc[has, "inst_dedup"])
    m.loc[has, "inst_total_count"] = m.loc[has, "inst_dedup"]
    path = OUT / "analysis_dataset_inst_dedup.csv"
    m.drop(columns=["inst_raw", "inst_dedup", "inst_distinct"]).to_csv(path, index=False)
    return path


def design(df: pd.DataFrame, outcome: str):
    sub = df.dropna(subset=[outcome, "persona_id"]).copy()
    cat, num = ml.pick_columns(sub, outcome)
    X = sub[cat + num].copy()
    for col in cat:
        X[col] = X[col].astype(str)
    return sub, X, sub[outcome].astype(float), ml.build_preprocessor(cat, num)


def table10(datasets: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for name, path in datasets.items():
        df = pd.read_csv(path, low_memory=False)
        for o in RUBRIC:
            sub, X, y, pre = design(df, o)
            info = ml.grouped_cv_score(ml.make_pipe(pre, classifier=False), X, y, sub["persona_id"], classifier=False, n_splits=5)
            rows.append({"data": name, "outcome": o, "cv_r2": info["cv_metric"], "cv_r2_sd": info["cv_metric_std"]})
            print(f"  {name:10s} {o:26s} R² {info['cv_metric']:.3f} ({info['cv_metric_std']:.3f})", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "table10_cv_r2_baseline_vs_dedup.csv", index=False)
    return t


def figure7(path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(path, low_memory=False)
    o = "korean_context_accuracy"
    sub, X, y, pre = design(df, o)
    pipe = ml.make_pipe(pre, classifier=False)
    pipe.fit(X, y)
    shap_dir = OUT / "ml_inst_dedup"
    shap_dir.mkdir(exist_ok=True)
    ml.shap_explain(pipe, X, outdir=shap_dir, outcome=o, classifier=False)
    tx = figs.TEXT["ko"]
    plt.rcParams["font.sans-serif"] = tx["font"]
    figs.shap_bars(shap_dir / f"shap_importance_{o}.csv", OUT / "fig7_shap_korean_context_inst_dedup.png", tx)


if __name__ == "__main__":
    dedup = build_dedup_dataset()
    print("[1] wrote", dedup.relative_to(ROOT))
    print("[2] table 10")
    table10({"baseline": ROOT / "results" / "analysis_dataset.csv", "inst_dedup": dedup})
    print("[3] figure 7")
    figure7(dedup)
    print("done")
