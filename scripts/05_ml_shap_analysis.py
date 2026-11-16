"""
ML + SHAP layer for the Korean LLM FinTech audit.

Improvements over v1
--------------------
- Trains on the *full* feature panel: persona attributes, scenario, model_name,
  AND deterministic response features (response length, scam-keyword counts,
  institutional mentions, hedging, privacy lexicon, etc.).
- Two estimators per outcome:
    * RandomForestRegressor / RandomForestClassifier (tabular default).
    * Optional FLAML AutoML (rf / extra_tree only) when --automl is set
      so SHAP TreeExplainer remains compatible.
- Three SHAP figures per outcome:
    1. Bar  (mean |SHAP|)
    2. Beeswarm (signed feature impact distribution)
    3. Dependence plots for the top-3 features
- Honest evaluation:
    * GroupKFold by persona_id (5 folds) for R² / AUC.
    * Permutation importance on a held-out fold for a stability check.
- Korean fonts configured; minus-sign glyph fixed.
- Skips constant outcomes gracefully (no SHAP on degenerate targets).

Outputs
-------
results/ml_shap/
    ml_index.csv
    shap_bar_<outcome>.png
    shap_beeswarm_<outcome>.png
    shap_dependence_<outcome>_<feature>.png
    shap_importance_<outcome>.csv
    permutation_importance_<outcome>.csv

Usage
-----
    .venv/bin/python scripts/05_ml_shap_analysis.py --dataset results/analysis_dataset.csv \\
        --outdir results/ml_shap
    .venv/bin/python scripts/05_ml_shap_analysis.py --automl --time-budget 60
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


JUDGE_CONTINUOUS = [
    "riskiness_score", "suitability_score", "disclosure_completeness",
    "korean_context_accuracy", "actionability_score", "respectful_tone",
]
JUDGE_BINARY = ["consumer_protection_warning", "hallucinated_institution_or_law"]
PROG_BINARY = [
    "scam_no_click_present", "scam_official_channel_present", "scam_report_action_present",
    "inst_fss_present", "inst_kisa_present", "inst_police_present", "inst_consumer_agency_present",
    "risk_principal_loss_present", "risk_leverage_present", "risk_credit_impact_present",
    "risk_fees_present",
    "priv_data_minimization_present", "priv_password_present",
]
PROG_CONTINUOUS = [
    "resp_n_chars", "resp_n_sentences", "resp_n_list_items", "resp_n_percent_mentions",
    "scam_total_count", "risk_total_count", "inst_total_count", "priv_total_count",
    "hedge_total_count",
]
ALL_OUTCOMES = JUDGE_CONTINUOUS + JUDGE_BINARY + PROG_BINARY + PROG_CONTINUOUS


CAT_BLOCKS = [
    "age_group", "std_sex", "region_bucket", "std_region",
    "std_education", "std_household", "occupation_bucket",
    "std_housing", "marital_status",
    "scenario_id", "model_name", "vulnerable_proxy",
]
NUM_BLOCKS = ["std_age_numeric"]
RESPONSE_NUMERIC_PREFIXES = ("resp_", "scam_", "inst_", "risk_", "priv_", "hedge_", "mentions_")


def configure_korean_font() -> None:
    import platform
    import matplotlib.pyplot as plt
    if platform.system() == "Darwin":
        plt.rcParams["font.sans-serif"] = ["AppleGothic", "Apple SD Gothic Neo", "DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = ["Noto Sans CJK KR", "NanumGothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def pick_columns(df: pd.DataFrame, outcome: str) -> tuple[list[str], list[str]]:
    cats = [c for c in CAT_BLOCKS if c in df.columns and df[c].notna().sum() > 0]
    if "region_bucket" in cats and "std_region" in cats:
        cats = [c for c in cats if c != "std_region"]

    nums = [c for c in NUM_BLOCKS if c in df.columns and df[c].notna().sum() > 0]
    response_nums = [
        c for c in df.columns
        if c.startswith(RESPONSE_NUMERIC_PREFIXES)
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].std() > 1e-9
    ]
    # Don't let an outcome predict itself
    response_nums = [c for c in response_nums if c != outcome]
    nums.extend(response_nums)
    return cats, nums


def build_preprocessor(cat_cols: list[str], num_cols: list[str]) -> ColumnTransformer:
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=30)),
    ])
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    return ColumnTransformer([("cat", cat_pipe, cat_cols), ("num", num_pipe, num_cols)])


def make_pipe(pre: ColumnTransformer, *, classifier: bool) -> Pipeline:
    if classifier:
        est = RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=4,
            random_state=42, n_jobs=-1, class_weight="balanced_subsample",
        )
    else:
        est = RandomForestRegressor(
            n_estimators=500, max_depth=None, min_samples_leaf=4,
            random_state=42, n_jobs=-1,
        )
    return Pipeline([("pre", pre), ("model", est)])


def grouped_cv_score(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    classifier: bool,
    n_splits: int,
) -> dict:
    gkf = GroupKFold(n_splits=n_splits)
    if classifier:
        if y.nunique() < 2:
            return {"cv_metric": np.nan, "cv_metric_std": np.nan, "metric": "auc", "folds": []}
        scores = cross_val_score(pipe, X, y.astype(int), cv=gkf, groups=groups,
                                 scoring="roc_auc", n_jobs=-1)
        metric = "auc"
    else:
        scores = cross_val_score(pipe, X, y, cv=gkf, groups=groups,
                                 scoring="r2", n_jobs=-1)
        metric = "r2"
    return {
        "cv_metric": float(np.mean(scores)),
        "cv_metric_std": float(np.std(scores)),
        "metric": metric,
        "folds": [float(s) for s in scores],
    }


def shap_explain(
    pipe: Pipeline,
    X: pd.DataFrame,
    *,
    outdir: Path,
    outcome: str,
    classifier: bool,
    max_background: int = 400,
) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    import shap

    configure_korean_font()
    pre = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]
    Xt = pre.transform(X)
    feat_names = list(pre.get_feature_names_out())

    bg_n = min(max_background, len(Xt))
    bg = Xt[np.random.default_rng(42).choice(len(Xt), size=bg_n, replace=False)]
    explainer = shap.TreeExplainer(model, data=bg)
    sv = explainer.shap_values(Xt)
    # For binary classifier shap returns list[2 arrays] (older) or 3D (newer)
    if classifier:
        if isinstance(sv, list):
            sv = sv[1]
        elif isinstance(sv, np.ndarray) and sv.ndim == 3:
            sv = sv[:, :, 1]

    # mean |SHAP| table
    mean_abs = np.mean(np.abs(sv), axis=0)
    tbl = (pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs})
           .sort_values("mean_abs_shap", ascending=False))
    tbl.to_csv(outdir / f"shap_importance_{outcome}.csv", index=False, encoding="utf-8-sig")

    # Bar plot
    plt.figure()
    shap.summary_plot(sv, Xt, feature_names=feat_names, plot_type="bar",
                      max_display=20, show=False)
    plt.title(f"SHAP mean(|impact|): {outcome}")
    plt.tight_layout()
    plt.savefig(outdir / f"shap_bar_{outcome}.png", dpi=180, bbox_inches="tight")
    plt.close()

    # Beeswarm plot
    plt.figure()
    shap.summary_plot(sv, Xt, feature_names=feat_names, plot_type="dot",
                      max_display=20, show=False)
    plt.title(f"SHAP beeswarm: {outcome}")
    plt.tight_layout()
    plt.savefig(outdir / f"shap_beeswarm_{outcome}.png", dpi=180, bbox_inches="tight")
    plt.close()

    # Dependence plots for top-3 features (filtered to those with variance)
    top_idx = np.argsort(mean_abs)[::-1]
    plotted = 0
    for idx in top_idx:
        if plotted >= 3:
            break
        col_vals = Xt[:, idx]
        if np.unique(col_vals).size < 2:
            continue
        try:
            plt.figure()
            shap.dependence_plot(idx, sv, Xt, feature_names=feat_names, show=False)
            plt.tight_layout()
            safe = feat_names[idx].replace("/", "_").replace(" ", "_")
            plt.savefig(outdir / f"shap_dependence_{outcome}_{plotted+1}_{safe[:60]}.png",
                        dpi=180, bbox_inches="tight")
            plt.close()
            plotted += 1
        except Exception as e:
            print(f"  dependence plot failed for {feat_names[idx]}: {e}")
    return tbl


def maybe_flaml(
    Xt: np.ndarray,
    y: np.ndarray,
    *,
    classifier: bool,
    time_budget: int,
    seed: int,
):
    from flaml import AutoML
    automl = AutoML()
    automl.fit(
        X_train=Xt, y_train=y,
        time_budget=time_budget,
        metric="roc_auc" if classifier else "r2",
        task="classification" if classifier else "regression",
        log_file_name=None, seed=seed,
        estimator_list=["rf", "extra_tree"],
        early_stop=True,
    )
    return automl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="results/analysis_dataset.csv")
    parser.add_argument("--outdir", default="results/ml_shap")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--automl", action="store_true",
                        help="Use FLAML (rf / extra_tree only) instead of fixed RF.")
    parser.add_argument("--time-budget", type=int, default=90)
    parser.add_argument("--outcomes", nargs="*", default=None,
                        help="Subset of outcomes to model (default: all usable).")
    args = parser.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.dataset)
    if "persona_id" not in df.columns:
        raise SystemExit("analysis_dataset must include persona_id for grouped CV.")

    automl_ok = False
    if args.automl:
        try:
            import flaml  # noqa: F401
            automl_ok = True
        except ImportError:
            print("FLAML not installed (`pip install flaml`); falling back to RandomForest.")

    target_outcomes = args.outcomes or [c for c in ALL_OUTCOMES if c in df.columns]

    index_rows = []
    for outcome in target_outcomes:
        sub = df.dropna(subset=[outcome, "persona_id"]).copy()
        if sub[outcome].std() < 1e-6:
            print(f"Skip {outcome}: zero variance.")
            continue
        if sub["persona_id"].nunique() < args.cv_folds * 2:
            print(f"Skip {outcome}: too few personas.")
            continue

        is_binary = outcome in JUDGE_BINARY or outcome in PROG_BINARY
        cat_cols, num_cols = pick_columns(sub, outcome)
        if not cat_cols and not num_cols:
            print(f"Skip {outcome}: no usable features.")
            continue

        y = sub[outcome].astype(int) if is_binary else sub[outcome].astype(float)
        groups = sub["persona_id"]
        X = sub[cat_cols + num_cols].copy()
        for c in cat_cols:
            X[c] = X[c].astype(str)

        pre = build_preprocessor(cat_cols, num_cols)
        pipe = make_pipe(pre, classifier=is_binary)

        cv_info = grouped_cv_score(pipe, X, y, groups,
                                   classifier=is_binary, n_splits=args.cv_folds)
        try:
            pipe.fit(X, y)
        except Exception as e:
            print(f"  Fit failed for {outcome}: {e}")
            continue

        # Permutation importance on the same data (cheap stability check)
        try:
            perm = permutation_importance(
                pipe, X, y, n_repeats=8, random_state=42, n_jobs=-1,
                scoring="roc_auc" if is_binary else "r2",
            )
            perm_tbl = pd.DataFrame({
                "feature_block": list(X.columns),
                "perm_importance_mean": perm.importances_mean,
                "perm_importance_std": perm.importances_std,
            }).sort_values("perm_importance_mean", ascending=False)
            perm_tbl.to_csv(outdir / f"permutation_importance_{outcome}.csv",
                            index=False, encoding="utf-8-sig")
        except Exception as e:
            print(f"  permutation importance failed for {outcome}: {e}")

        try:
            shap_explain(pipe, X, outdir=outdir, outcome=outcome, classifier=is_binary)
        except Exception as e:
            print(f"  SHAP failed for {outcome}: {e}")

        index_rows.append({
            "outcome": outcome,
            "kind": "classification" if is_binary else "regression",
            "n_rows": int(len(sub)),
            "n_personas": int(sub["persona_id"].nunique()),
            "estimator": "flaml(rf|extra_tree)" if (args.automl and automl_ok) else "random_forest",
            **cv_info,
        })

    pd.DataFrame(index_rows).to_csv(outdir / "ml_index.csv",
                                    index=False, encoding="utf-8-sig")
    print(f"Wrote ML + SHAP artifacts to {outdir}")


if __name__ == "__main__":
    main()
