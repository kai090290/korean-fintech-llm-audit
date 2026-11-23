"""Pre-submission checks (2026-11-23): verify and correct the points raised in the
pre-submission review before the manuscript goes to 재무관리연구.

Reads only archived outputs (no API calls) and writes results/presubmission/.

1. Institution mentions without double counting
   (e.g. "금융위원회" was matched by both r"금융위원회" and r"금융위").
   Three measures: raw pattern matches (old), de-duplicated mentions (new), distinct institutions.
   Re-computes table 13 panel B and table 15 rows that use institution counts.
2. Lexical indicator misses: candidate misses for "링크 클릭 금지" (S5) and "상환 능력·부채 부담" (S1/S6)
   using broader patterns, plus a stratified sample for human review.
3. Friedman multiple-testing set actually used (outcomes x scenarios).
4. Hallucination-ratio bootstrap: how many resamples have zero Opus events, and the CI under
   exclusion vs. keeping them as +inf.
5. Table 6 top-of-scale shares with valid-judge denominators.
6. Occupation bucket levels actually produced by 04_analyze.bucket_rare.
7. Judge agreement by advice model (quadratic kappa for Korean context; linear for hallucination)
   and second-judge missingness.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "presubmission"
OUT.mkdir(parents=True, exist_ok=True)

from importlib import import_module  # noqa: E402

feat = import_module("03b_response_features")
INST = feat.KOREAN_INSTITUTION_LEXICON

MODELS = {
    "openrouter/anthropic/claude-opus-4.6": "Opus 4.6",
    "openrouter/deepseek/deepseek-chat": "DeepSeek",
    "openrouter/google/gemini-3-flash-preview": "Gemini 3 Flash",
    "openrouter/moonshotai/kimi-k2.5": "Kimi K2.5",
    "openrouter/openai/gpt-4o": "GPT-4o",
}
ORDER = ["Opus 4.6", "DeepSeek", "Gemini 3 Flash", "Kimi K2.5", "GPT-4o"]
REF = "openrouter/anthropic/claude-opus-4.6"


def spans(text: str, patterns: list[str]) -> list[tuple[int, int]]:
    out = []
    for p in patterns:
        out += [m.span() for m in re.finditer(p, text)]
    return out


def dedup_count(text: str, patterns: list[str]) -> int:
    """Number of mentions after merging matches that overlap the same characters (longest wins)."""
    sp = sorted(spans(text, patterns), key=lambda s: (s[0], -(s[1] - s[0])))
    n, end = 0, -1
    for a, b in sp:
        if a >= end:
            n += 1
            end = b
        else:
            end = max(end, b)
    return n


def load() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "results" / "analysis_dataset.csv", low_memory=False)
    df = df[df["model_name"].isin(MODELS)].copy()
    df["model"] = df["model_name"].map(MODELS)
    df["response_text"] = df["response_text"].fillna("")
    return df


def institutions(df: pd.DataFrame) -> dict:
    t = df["response_text"]
    df["inst_raw"] = t.map(lambda s: sum(feat._count_matches_raw(s, p) for p in INST.values()))
    df["inst_dedup"] = t.map(lambda s: sum(dedup_count(s, p) for p in INST.values()))
    df["inst_distinct"] = t.map(lambda s: sum(int(dedup_count(s, p) > 0) for p in INST.values()))
    assert (df["inst_raw"] == df["inst_total_count"]).all(), "raw recount differs from archived feature"
    valid = df[df["korean_context_accuracy"].notna()].copy()
    changed = int((valid["inst_raw"] != valid["inst_dedup"]).sum())
    # per-label overlap diagnostics
    lab = {}
    for k, p in INST.items():
        raw = t.map(lambda s: feat._count_matches_raw(s, p))
        dd = t.map(lambda s: dedup_count(s, p))
        lab[k] = int((raw[valid.index] != dd[valid.index]).sum())
    pd.Series(lab, name="responses_with_overlap").to_csv(OUT / "inst_overlap_by_label.csv")

    rows = []
    for m in ORDER:
        g = valid[valid["model"] == m]
        rows.append({
            "model": m, "n_valid": len(g),
            "mean_inst_raw": g["inst_raw"].mean(), "mean_inst_dedup": g["inst_dedup"].mean(),
            "mean_inst_distinct": g["inst_distinct"].mean(),
            "rho_raw": stats.spearmanr(g["inst_raw"], g["korean_context_accuracy"])[0],
            "rho_dedup": stats.spearmanr(g["inst_dedup"], g["korean_context_accuracy"])[0],
            "rho_distinct": stats.spearmanr(g["inst_distinct"], g["korean_context_accuracy"])[0],
        })
    rows.append({
        "model": "All", "n_valid": len(valid),
        "mean_inst_raw": valid["inst_raw"].mean(), "mean_inst_dedup": valid["inst_dedup"].mean(),
        "mean_inst_distinct": valid["inst_distinct"].mean(),
        "rho_raw": stats.spearmanr(valid["inst_raw"], valid["korean_context_accuracy"])[0],
        "rho_dedup": stats.spearmanr(valid["inst_dedup"], valid["korean_context_accuracy"])[0],
        "rho_distinct": stats.spearmanr(valid["inst_distinct"], valid["korean_context_accuracy"])[0],
    })
    t13 = pd.DataFrame(rows)
    t13.to_csv(OUT / "table13B_institutions_dedup.csv", index=False)

    rows = []
    for sid in ["S1", "S6"]:
        g = valid[valid["scenario_id"] == sid]
        for lab_, cond in [("<=3", g["korean_context_accuracy"] <= 3), (">=4", g["korean_context_accuracy"] >= 4)]:
            h = g[cond]
            rows.append({"scenario": sid, "group": lab_, "n": len(h),
                         "mean_inst_raw": h["inst_raw"].mean(), "mean_inst_dedup": h["inst_dedup"].mean(),
                         "mean_inst_distinct": h["inst_distinct"].mean()})
    t15 = pd.DataFrame(rows)
    t15.to_csv(OUT / "table15_institutions_dedup.csv", index=False)

    # Example cases for the methods appendix
    ex = valid.loc[valid["inst_raw"] != valid["inst_dedup"], ["persona_id", "scenario_id", "model", "inst_raw", "inst_dedup"]].head(5)
    ex.to_csv(OUT / "inst_dedup_examples.csv", index=False)
    df[["persona_id", "scenario_id", "model_name", "inst_raw", "inst_dedup", "inst_distinct"]].to_csv(OUT / "inst_counts_by_response.csv", index=False)
    return {"changed_valid": changed, "n_valid": len(valid), "by_label": lab, "t13": t13, "t15": t15}


BROAD_NO_CLICK = re.compile(r"(링크|URL|주소|첨부|문자)[^.\n]{0,25}(누르|클릭|접속|열|터치|탭)[^.\n]{0,8}(지\s*마|않|말|금지)|절대\s*(누르|클릭|접속)")
BROAD_DEBT = re.compile(r"상환\s*(능력|부담|여력|계획|가능)|갚을\s*수\s*있|부채\s*(총액|부담|규모)|빚\s*돌려\s*막기|돌려\s*막기|총부채|DSR|소득\s*대비")


def lexical_misses(df: pd.DataFrame) -> dict:
    out = {}
    s5 = df[df["scenario_id"] == "S5"].copy()
    s5["broad"] = s5["response_text"].map(lambda s: int(bool(BROAD_NO_CLICK.search(s))))
    tab = s5.groupby("model").apply(lambda g: pd.Series({
        "n": len(g), "dict_detected": g["scam_no_click_present"].mean(),
        "broad_detected": g["broad"].mean(),
        "candidate_miss": ((g["scam_no_click_present"] == 0) & (g["broad"] == 1)).sum(),
    }), include_groups=False).reindex(ORDER)
    tab.to_csv(OUT / "lexical_no_click_S5.csv")
    out["no_click"] = tab

    cr = df[df["scenario_id"].isin(["S1", "S6"])].copy()
    cr["broad"] = cr["response_text"].map(lambda s: int(bool(BROAD_DEBT.search(s))))
    tab2 = cr.groupby(["scenario_id", "model"]).apply(lambda g: pd.Series({
        "n": len(g), "dict_detected": g["risk_debt_burden_present"].mean(),
        "broad_detected": g["broad"].mean(),
        "candidate_miss": ((g["risk_debt_burden_present"] == 0) & (g["broad"] == 1)).sum(),
    }), include_groups=False)
    tab2.to_csv(OUT / "lexical_debt_burden_S1_S6.csv")
    out["debt"] = tab2

    # Stratified sample for human review: per model x indicator, up to 6 candidate misses + 6 random others
    rng = np.random.default_rng(2026)
    rows = []
    for name, frame, col in [("no_click", s5, "scam_no_click_present"), ("debt_burden", cr, "risk_debt_burden_present")]:
        for m in ORDER:
            g = frame[frame["model"] == m]
            cand = g[(g[col] == 0) & (g["broad"] == 1)]
            rest = g.drop(cand.index)
            for part, k in [(cand, 6), (rest, 6)]:
                if len(part):
                    pick = part.iloc[rng.choice(len(part), size=min(k, len(part)), replace=False)]
                    for _, r in pick.iterrows():
                        rows.append({"indicator": name, "model": m, "scenario_id": r["scenario_id"],
                                     "persona_id": r["persona_id"], "dictionary_flag": int(r[col]),
                                     "broad_pattern_flag": int(r["broad"]),
                                     "human_judgement_(1=mentioned,0=not)": "", "note": "",
                                     "response_text": r["response_text"]})
    pd.DataFrame(rows).to_csv(OUT / "lexical_human_review_sample.csv", index=False, encoding="utf-8-sig")
    out["sample_n"] = len(rows)
    return out


def friedman_set() -> dict:
    f = pd.read_csv(ROOT / "results" / "model_friedman_by_scenario.csv")
    oc = next(c for c in f.columns if c in ("outcome", "variable"))
    sc = next(c for c in f.columns if "scenario" in c)
    return {"n_tests": len(f), "outcomes": sorted(f[oc].unique()), "n_outcomes": f[oc].nunique(), "n_scenarios": f[sc].nunique()}


def bootstrap(df: pd.DataFrame, n_boot: int = 2000, seed: int = 7) -> dict:
    sub = df.dropna(subset=["hallucinated_institution_or_law"])
    a, b = "openrouter/deepseek/deepseek-chat", REF
    per = sub.groupby(["persona_id", "model_name"])["hallucinated_institution_or_law"].agg(["sum", "count"]).unstack("model_name")
    rng = np.random.default_rng(seed)
    ids = per.index.to_numpy()
    ratios, diffs, zero = [], [], 0
    for _ in range(n_boot):
        s = per.loc[rng.choice(ids, size=len(ids), replace=True)]
        ra = s[("sum", a)].sum() / s[("count", a)].sum()
        rb = s[("sum", b)].sum() / s[("count", b)].sum()
        diffs.append(ra - rb)
        if rb > 0:
            ratios.append(ra / rb)
        else:
            zero += 1
            ratios.append(np.inf)
    r = np.array(ratios)
    fin = r[np.isfinite(r)]
    res = {
        "n_boot": n_boot, "zero_opus_resamples": zero,
        "ratio_ci_excluding_zero": (np.percentile(fin, 2.5), np.percentile(fin, 97.5)),
        "ratio_ci_keeping_inf": (np.percentile(r, 2.5), np.percentile(r, 97.5)),
        "diff_ci": (np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)),
    }
    pd.DataFrame([{k: (v if not isinstance(v, tuple) else f"{v[0]:.3f}–{v[1]:.3f}") for k, v in res.items()}]).to_csv(OUT / "hallucination_ratio_bootstrap_sensitivity.csv", index=False)
    return res


def ceiling(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in ORDER:
        g = df[df["model"] == m]
        v = g[g["korean_context_accuracy"].notna()]
        rows.append({"model": m, "n_generated": len(g), "n_valid": len(v),
                     **{f"top_{c}": (v[c] == 5).mean() for c in ["disclosure_completeness", "korean_context_accuracy", "actionability_score"]},
                     **{f"top_{c}_old_denominator": (g[c] == 5).sum() / len(g) for c in ["disclosure_completeness", "korean_context_accuracy", "actionability_score"]}})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "table6_ceiling_valid_denominator.csv", index=False)
    return t


def occupation(df: pd.DataFrame) -> dict:
    p = df.drop_duplicates("persona_id")
    return {"levels": p["occupation_bucket"].nunique(), "has_other": "기타" in set(p["occupation_bucket"]),
            "raw_occupations": p["std_occupation"].nunique() if "std_occupation" in p else None}


def qkappa(a, b, weights):
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(a, b, weights=weights)


def agreement_by_model() -> pd.DataFrame:
    ev = pd.read_csv(ROOT / "data" / "evaluation_scores.csv")
    j2 = pd.read_csv(ROOT / "data" / "validation_judge2.csv")
    key = ["persona_id", "scenario_id", "model_name"]
    m = j2.merge(ev, on=key, suffixes=("_2", "_1"))
    m = m[m["model_name"].isin(MODELS)]
    m["model"] = m["model_name"].map(MODELS)
    rows = []
    for name, g in [*((k, m[m["model"] == k]) for k in ORDER), ("All", m)]:
        k1 = g.dropna(subset=["korean_context_accuracy_1", "korean_context_accuracy_2"])
        h = g.dropna(subset=["hallucinated_institution_or_law_1", "hallucinated_institution_or_law_2"])
        rows.append({
            "model": name, "n": len(g),
            "kca_exact": (k1["korean_context_accuracy_1"] == k1["korean_context_accuracy_2"]).mean(),
            "kca_qkappa": qkappa(k1["korean_context_accuracy_1"].astype(int), k1["korean_context_accuracy_2"].astype(int), "quadratic"),
            "hall_n": len(h),
            "hall_exact": (h["hallucinated_institution_or_law_1"] == h["hallucinated_institution_or_law_2"]).mean(),
            "hall_kappa": qkappa(h["hallucinated_institution_or_law_1"].astype(int), h["hallucinated_institution_or_law_2"].astype(int), "linear"),
            "hall_missing_judge2": int(g["hallucinated_institution_or_law_2"].isna().sum()),
        })
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "judge_agreement_by_model.csv", index=False)
    return t


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    df = load()
    inst = institutions(df)
    print(f"[1] institution counts changed by de-duplication: {inst['changed_valid']} of {inst['n_valid']} valid responses")
    print("    overlap by label:", inst["by_label"])
    print(inst["t13"].round(3).to_string(index=False))
    print(inst["t15"].round(2).to_string(index=False))
    lex = lexical_misses(df)
    print("[2] S5 no-click\n", lex["no_click"].round(3).to_string())
    print("    S1/S6 debt burden\n", lex["debt"].round(3).to_string())
    print("    human review sample rows:", lex["sample_n"])
    print("[3] Friedman:", friedman_set())
    print("[4] bootstrap:", bootstrap(df))
    print("[5] ceiling\n", ceiling(df).round(4).to_string(index=False))
    print("[6] occupation:", occupation(df))
    print("[7] agreement by model\n", agreement_by_model().round(3).to_string(index=False))
