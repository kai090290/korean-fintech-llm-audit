"""
Create a manual validation sample for human coding.

The sample is stratified by model x scenario so every cell is covered, and the
coder-facing sheet omits the LLM-judge scores (blind coding). Judge scores are
written to a separate key file that is joined back only at analysis time.

Usage:
    python scripts/05_manual_validation_template.py \
        --responses data/model_outputs.csv --scores data/evaluation_scores.csv \
        --out data/manual_validation_sample.csv --per-cell 17 --pilot-per-cell 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

KEYS = ["persona_id", "scenario_id", "model_name"]
HUMAN_COLS = [
    "human_riskiness_score",
    "human_suitability_score",
    "human_disclosure_completeness",
    "human_consumer_protection_warning",
    "human_korean_context_accuracy",
    "human_hallucinated_institution_or_law",
    "human_actionability_score",
    "human_respectful_tone",
    "human_notes",
]
JUDGE_COLS = [
    "riskiness_score", "suitability_score", "disclosure_completeness",
    "consumer_protection_warning", "korean_context_accuracy",
    "hallucinated_institution_or_law", "actionability_score", "respectful_tone",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=str, default="data/model_outputs.csv")
    parser.add_argument("--scores", type=str, default="data/evaluation_scores.csv")
    parser.add_argument("--out", type=str, default="data/manual_validation_sample.csv")
    parser.add_argument("--per-cell", type=int, default=17, help="responses per model x scenario cell")
    parser.add_argument("--pilot-per-cell", type=int, default=4,
                        help="per-cell subset flagged for double coding (inter-rater reliability)")
    parser.add_argument("--exclude-models", type=str,
                        default="openrouter/anthropic/claude-sonnet-4.5",
                        help="comma-separated model_name values left out (default: the partial "
                             "earlier-run Sonnet 4.5 slate that is not in the main analysis dataset)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    responses = pd.read_csv(args.responses)
    scores = pd.read_csv(args.scores)
    scores = scores[scores["judge_status"].eq("ok")].drop_duplicates(KEYS)
    df = responses[responses["status"].eq("ok")].merge(
        scores[KEYS + JUDGE_COLS], on=KEYS, how="inner"
    )

    excluded = {m.strip() for m in args.exclude_models.split(",") if m.strip()}
    df = df[~df["model_name"].isin(excluded)]

    sample = pd.concat(
        [g.sample(n=min(args.per_cell, len(g)), random_state=args.seed)
         for _, g in df.groupby(["model_name", "scenario_id"])]
    )
    sample = sample.sample(frac=1, random_state=args.seed).reset_index(drop=True)  # shuffle row order
    sample.insert(0, "item_id", [f"V{i + 1:04d}" for i in range(len(sample))])
    sample["pilot_double_code"] = (
        sample.groupby(["model_name", "scenario_id"]).cumcount() < args.pilot_per_cell
    ).astype(int)
    for col in HUMAN_COLS:
        sample[col] = ""

    # Blind coder sheet: no judge scores, no model name.
    coder_cols = ["item_id", "pilot_double_code", "scenario_id", "scenario_title",
                  "user_prompt", "expected_safety_elements", "response_text"] + HUMAN_COLS
    coder_cols = [c for c in coder_cols if c in sample.columns]
    sample[coder_cols].to_csv(args.out, index=False, encoding="utf-8-sig")

    key_path = Path(args.out).with_name(Path(args.out).stem + "_key.csv")
    sample[["item_id"] + KEYS + JUDGE_COLS].to_csv(key_path, index=False, encoding="utf-8-sig")

    print(f"Saved blind coder sheet ({len(sample)} rows) to {args.out}")
    print(f"Saved judge-score key to {key_path}")
    print(f"Pilot double-coding subset: {int(sample['pilot_double_code'].sum())} rows")


if __name__ == "__main__":
    main()
