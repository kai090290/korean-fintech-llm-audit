"""
Single-colour manuscript figures (Sungkyunkwan University green).

Regenerates the seven figures used in manuscript/manuscript_full.md from the
tables already produced by the pipeline (no models are refit):

    results/analysis_dataset.csv
    results/ml_shap/shap_importance_korean_context_accuracy.csv

Heatmaps use one hue running light -> dark (darker = larger value); bars use
the brand colour. Change BRAND to the university's official hex if it differs.

Axis labels show each scenario code together with a short name (e.g. "S5 / Smishing")
so that a figure can be read on its own. ``--lang ko`` writes Korean titles, axes, and
labels for the Korean manuscript.

Usage:
    python scripts/07_manuscript_figures.py --results results --out manuscript/figures
    python scripts/07_manuscript_figures.py --lang ko --out manuscript/figures_ko
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb

BRAND = "#0B6B3F"        # deep green used for bars and the dark end of the ramp
RAMP_LIGHT = "#EDF5F0"   # very light tint of the same hue (low values)
INK = "#1F2A24"          # text
MUTED = "#5B6A62"        # secondary text, axes
GRID = "#DCE5DF"         # recessive gridlines

CMAP = LinearSegmentedColormap.from_list("skku_green", [RAMP_LIGHT, BRAND])

MODEL_LABELS = {
    "openrouter/anthropic/claude-opus-4.6": "Claude Opus 4.6",
    "openrouter/deepseek/deepseek-chat": "DeepSeek Chat",
    "openrouter/google/gemini-3-flash-preview": "Gemini 3 Flash (preview)",
    "openrouter/moonshotai/kimi-k2.5": "Kimi K2.5",
    "openrouter/openai/gpt-4o": "GPT-4o",
}

TEXT = {
    "en": {
        "font": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "scenario": {"S1": "S1\nMobile\nloan", "S2": "S2\nRobo-\nadvisor", "S3": "S3\nCrypto/\nleverage",
                     "S4": "S4\nOpen\nbanking", "S5": "S5\nSmishing", "S6": "S6\nBNPL"},
        "age": {},
        "scenario_axis": "Scenario", "model_axis": "Advice model", "age_axis": "Age group",
        "fig1": "Korean contextual accuracy by advice model and scenario",
        "fig2": "Riskiness score by advice model and scenario",
        "fig3": "Suitability score by advice model and scenario",
        "fig4": "Hallucination flags by advice model and scenario",
        "fig5": "Korean contextual accuracy by scenario (all models, all personas)",
        "fig6": "Riskiness score by age group and scenario",
        "fig7": "Korean contextual accuracy: top random-forest features",
        "cb_high": "Mean score (1–5); darker = higher", "cb_risk": "Mean score (1–5); darker = riskier",
        "cb_pct": "Share of responses (%); darker = higher", "mean_score": "Mean score (1–5)",
        "shap_x": "Mean |SHAP value| (impact on predicted score)",
        "model_prefix": "Model: ", "scenario_prefix": "Scenario: ",
        "features": {
            "resp_avg_sentence_chars": "Avg. sentence length (chars)",
            "inst_total_count": "Institution mentions",
            "scam_report_action_count": "Scam-reporting action mentions",
            "resp_n_words": "Response words", "resp_n_chars": "Response characters",
            "mentions_persona_name": "Persona name mentioned",
            "resp_n_english_tokens": "English tokens", "resp_n_list_items": "List items",
            "risk_total_count": "Risk-term mentions", "scam_total_count": "Scam-term mentions",
            "std_age_numeric": "Persona age", "resp_n_sentences": "Response sentences",
        },
    },
    "ko": {
        "font": ["NanumGothic", "Apple SD Gothic Neo", "AppleGothic"],
        "scenario": {"S1": "S1\n모바일\n대출", "S2": "S2\n로보\n어드바이저", "S3": "S3\n암호화폐·\n레버리지",
                     "S4": "S4\n오픈뱅킹", "S5": "S5\n스미싱", "S6": "S6\nBNPL\n할부"},
        "age": {"19-29": "19–29세", "30-39": "30–39세", "40-49": "40–49세", "50-59": "50–59세",
                "60-69": "60–69세", "70+": "70세 이상"},
        "scenario_axis": "시나리오", "model_axis": "자문 모델", "age_axis": "연령대",
        "fig1": "자문 모델·시나리오별 한국 맥락 정확도",
        "fig2": "자문 모델·시나리오별 위험도",
        "fig3": "자문 모델·시나리오별 적합성",
        "fig4": "자문 모델·시나리오별 환각 판정 비율",
        "fig5": "시나리오별 한국 맥락 정확도(전체 모델·페르소나 평균)",
        "fig6": "연령대·시나리오별 위험도",
        "fig7": "한국 맥락 정확도: 랜덤포레스트 상위 특성",
        "cb_high": "평균 점수(1–5); 진할수록 높음", "cb_risk": "평균 점수(1–5); 진할수록 위험",
        "cb_pct": "응답 비율(%); 진할수록 높음", "mean_score": "평균 점수(1–5)",
        "shap_x": "평균 |SHAP 값|(예측 점수에 대한 영향)",
        "model_prefix": "모델: ", "scenario_prefix": "시나리오: ",
        "features": {
            "resp_avg_sentence_chars": "평균 문장 길이(자)",
            "inst_total_count": "기관 언급 횟수",
            "scam_report_action_count": "사기 신고 행동 언급 횟수",
            "resp_n_words": "응답 어절 수", "resp_n_chars": "응답 글자 수",
            "mentions_persona_name": "페르소나 이름 호명",
            "resp_n_english_tokens": "영어 토큰 수", "resp_n_list_items": "목록 항목 수",
            "risk_total_count": "위험 표현 횟수", "scam_total_count": "사기 관련 표현 횟수",
            "std_age_numeric": "페르소나 나이", "resp_n_sentences": "응답 문장 수",
        },
    },
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "text.color": INK,
    "axes.labelcolor": MUTED,
    "axes.edgecolor": GRID,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "axes.unicode_minus": False,
})


def _luminance(rgb) -> float:
    r, g, b = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _text_on(rgb) -> str:
    """White ink on dark cells, dark ink on light cells (contrast >= 4.5 either way)."""
    return "white" if _luminance(rgb) < 0.28 else INK


def heatmap(piv: pd.DataFrame, *, title: str, cbar_label: str, fmt: str, path: Path,
            xlabel: str, ylabel: str, figsize=(7.2, 4.1), col_labels: dict | None = None) -> None:
    vals = piv.values.astype(float)
    vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(vals, cmap=CMAP, aspect="auto", vmin=vmin, vmax=vmax)
    col_labels = col_labels or {}
    ax.set_xticks(range(piv.shape[1]), [col_labels.get(str(c), str(c)) for c in piv.columns])
    ax.set_yticks(range(piv.shape[0]), [str(i) for i in piv.index])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_xticks(np.arange(-0.5, piv.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, piv.shape[0]), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    span = (vmax - vmin) or 1.0
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            rgb = to_rgb(CMAP((vals[i, j] - vmin) / span))
            ax.text(j, i, fmt.format(vals[i, j]), ha="center", va="center",
                    fontsize=8.5, color=_text_on(rgb))
    ax.set_title(title, loc="left", fontsize=10.5, pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label(cbar_label, fontsize=9, color=MUTED)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=8.5)
    fig.savefig(path)
    plt.close(fig)


def model_scenario(df: pd.DataFrame, col: str) -> pd.DataFrame:
    piv = df.pivot_table(index="model_name", columns="scenario_id", values=col, aggfunc="mean")
    order = [m for m in MODEL_LABELS if m in piv.index]
    piv = piv.loc[order]
    piv.index = [MODEL_LABELS[m] for m in order]
    return piv


def scenario_bars(df: pd.DataFrame, col: str, *, ylabel: str, title: str, path: Path,
                  xlabel: str, col_labels: dict) -> None:
    means = df.groupby("scenario_id")[col].mean().sort_index()
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    bars = ax.bar([col_labels.get(c, c) for c in means.index], means.values, color=BRAND, width=0.58)
    ax.set_ylim(0, 5.4)  # headroom for the value labels
    ax.set_yticks(range(0, 6))
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)
    for b, v in zip(bars, means.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.2f}", ha="center",
                va="bottom", fontsize=8.5, color=INK)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, loc="left", fontsize=10.5, pad=8)
    fig.savefig(path)
    plt.close(fig)


def feature_label(raw: str, tx: dict) -> str:
    if raw.startswith("cat__model_name_"):
        m = raw[len("cat__model_name_"):]
        return tx["model_prefix"] + MODEL_LABELS.get(m, m.replace("openrouter/", ""))
    if raw.startswith("cat__scenario_id_"):
        sid = raw[len("cat__scenario_id_"):]
        return tx["scenario_prefix"] + tx["scenario"].get(sid, sid).replace("\n", " ")
    name = raw.replace("num__", "").replace("cat__", "")
    return tx["features"].get(name, name)


def shap_bars(csv: Path, path: Path, tx: dict, top: int = 15) -> None:
    t = pd.read_csv(csv, encoding="utf-8-sig").head(top).iloc[::-1]
    names = [feature_label(f, tx) for f in t["feature"]]
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.barh(names, t["mean_abs_shap"], color=BRAND, height=0.62)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0, labelsize=8.5)
    for y, v in enumerate(t["mean_abs_shap"]):
        ax.text(v + 0.003, y, f"{v:.3f}", va="center", fontsize=8, color=INK)
    ax.set_xlabel(tx["shap_x"], fontsize=9)
    ax.set_title(tx["fig7"], loc="left", fontsize=10.5, pad=8)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="manuscript/figures")
    ap.add_argument("--lang", choices=sorted(TEXT), default="en")
    args = ap.parse_args()
    res, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tx = TEXT[args.lang]
    plt.rcParams["font.sans-serif"] = tx["font"]
    sc = tx["scenario"]

    df = pd.read_csv(res / "analysis_dataset.csv")

    common = dict(xlabel=tx["scenario_axis"], ylabel=tx["model_axis"], col_labels=sc)
    heatmap(model_scenario(df, "korean_context_accuracy"), title=tx["fig1"],
            cbar_label=tx["cb_high"], fmt="{:.2f}", path=out / "fig1_korean_context_heatmap.png", **common)
    heatmap(model_scenario(df, "riskiness_score"), title=tx["fig2"],
            cbar_label=tx["cb_risk"], fmt="{:.2f}", path=out / "fig2_riskiness_heatmap.png", **common)
    heatmap(model_scenario(df, "suitability_score"), title=tx["fig3"],
            cbar_label=tx["cb_high"], fmt="{:.2f}", path=out / "fig3_suitability_heatmap.png", **common)
    heatmap(model_scenario(df, "hallucinated_institution_or_law") * 100, title=tx["fig4"],
            cbar_label=tx["cb_pct"], fmt="{:.0f}%", path=out / "fig4_hallucination_heatmap.png", **common)
    scenario_bars(df, "korean_context_accuracy", ylabel=tx["mean_score"], title=tx["fig5"],
                  xlabel=tx["scenario_axis"], col_labels=sc,
                  path=out / "fig5_korean_context_by_scenario.png")

    age_order = ["19-29", "30-39", "40-49", "50-59", "60-69", "70+"]
    piv = df.pivot_table(index="age_group", columns="scenario_id", values="riskiness_score", aggfunc="mean")
    piv = piv.loc[[a for a in age_order if a in piv.index]]
    piv.index = [tx["age"].get(a, a) for a in piv.index]
    heatmap(piv, title=tx["fig6"], cbar_label=tx["cb_risk"], fmt="{:.2f}",
            xlabel=tx["scenario_axis"], ylabel=tx["age_axis"], col_labels=sc,
            path=out / "fig6_riskiness_age_scenario.png", figsize=(7.2, 3.9))

    shap_bars(res / "ml_shap" / "shap_importance_korean_context_accuracy.csv",
              out / "fig7_shap_korean_context.png", tx)
    print(f"Saved 7 figures ({args.lang}) to {out}")


if __name__ == "__main__":
    main()
