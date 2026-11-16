"""
Sample personas from nvidia/Nemotron-Personas-Korea.

The open HF split is a flat table (no nested dicts): age, sex, province,
district, education_level, occupation, family_type, housing_type, persona,
uuid, plus optional narrative columns. There is no income field.

An explicit column map is applied for this dataset; regex patterns fill
any remaining std_* slots for alternative schemas. persona_id is taken from
uuid when present.

Usage:
    python scripts/00_sample_personas.py --n 500 --out data/personas_sample.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import re
from typing import Optional, List

import numpy as np
import pandas as pd
from datasets import load_dataset


DATASET_NAME = "nvidia/Nemotron-Personas-Korea"

# nvidia/Nemotron-Personas-Korea uses a flat parquet schema with stable English
# column names. Apply these first so regex fallback does not mis-map (e.g.
# "persona" vs "arts_persona").
NEMOTRON_KOREA_EXPLICIT = {
    "std_age": "age",
    "std_sex": "sex",
    "std_region": "province",
    "std_district": "district",
    "std_education": "education_level",
    "std_occupation": "occupation",
    "std_household": "family_type",
    "std_housing": "housing_type",
    "std_persona_text": "persona",
}


def flatten_record(record: dict) -> dict:
    """Flatten one nested dictionary record into simple columns."""
    out = {}
    for k, v in record.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                out[f"{k}.{kk}"] = vv
        elif isinstance(v, list):
            out[k] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = v
    return out


def find_col(columns: List[str], patterns: List[str]) -> Optional[str]:
    """Find a likely column using regex patterns over lower-case column names."""
    lowered = {c: c.lower() for c in columns}
    for pat in patterns:
        rx = re.compile(pat, flags=re.IGNORECASE)
        for original, low in lowered.items():
            if rx.search(original) or rx.search(low):
                return original
    return None


def add_standard_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create standard names if likely matching columns are found."""
    cols = list(df.columns)

    for std_name, src in NEMOTRON_KOREA_EXPLICIT.items():
        if src in cols and std_name not in df.columns:
            df[std_name] = df[src]

    mapping = {
        "std_age": [r"\bage\b", r"나이", r"연령"],
        "std_sex": [r"\bsex\b", r"\bgender\b", r"성별"],
        "std_region": [r"region", r"province", r"시도", r"지역"],
        "std_district": [r"district", r"city", r"county", r"시군구", r"구$"],
        "std_education": [r"education", r"학력", r"교육"],
        "std_occupation": [r"occupation", r"job", r"직업"],
        "std_industry": [r"industry", r"산업"],
        "std_income": [r"income", r"소득"],
        "std_household": [r"household", r"family_type\b", r"가구", r"가족"],
        "std_housing": [r"housing", r"tenure", r"주거"],
        # Prefer exact column "persona" before other *persona* fields.
        "std_persona_text": [
            r"^persona$",
            r"profile",
            r"description",
            r"페르소나",
            r"프로필",
            r"persona",
        ],
    }

    for std_name, pats in mapping.items():
        if std_name in df.columns:
            continue
        col = find_col(cols, pats)
        if col is not None:
            df[std_name] = df[col]

    if "persona_id" not in df.columns:
        if "uuid" in df.columns:
            df.insert(0, "persona_id", df["uuid"].astype(str))
        else:
            df.insert(0, "persona_id", [f"P{i:07d}" for i in range(len(df))])

    if "std_age" in df.columns:
        numeric = pd.to_numeric(df["std_age"], errors="coerce")
        if numeric.notna().sum() < max(1, int(len(df) * 0.2)):
            numeric = pd.to_numeric(
                df["std_age"].astype(str).str.extract(r"(\d+)")[0],
                errors="coerce",
            )
        df["std_age_numeric"] = numeric
        df["age_group"] = pd.cut(
            df["std_age_numeric"],
            bins=[18, 29, 39, 49, 59, 69, 120],
            labels=["19-29", "30-39", "40-49", "50-59", "60-69", "70+"],
            include_lowest=True,
        )
    else:
        df["age_group"] = "unknown"

    return df


def stratified_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Prefer stratification by age/sex/region when available; fall back to random."""
    rng = np.random.default_rng(seed)
    strata_cols = [c for c in ["age_group", "std_sex", "std_region"] if c in df.columns]
    if not strata_cols or len(df) <= n:
        return df.sample(n=min(n, len(df)), random_state=seed)

    # Sample approximately equal proportion from each stratum, then top up.
    grouped = df.groupby(strata_cols, dropna=False, group_keys=False)
    sampled_parts = []
    target_frac = min(1.0, n / len(df))
    for _, group in grouped:
        k = max(1, int(round(len(group) * target_frac)))
        sampled_parts.append(group.sample(n=min(k, len(group)), random_state=int(rng.integers(1, 1_000_000))))

    sampled = pd.concat(sampled_parts, ignore_index=True).drop_duplicates(subset=["persona_id"])
    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=seed)
    elif len(sampled) < n:
        remainder = df.loc[~df["persona_id"].isin(sampled["persona_id"])]
        topup = remainder.sample(n=min(n - len(sampled), len(remainder)), random_state=seed)
        sampled = pd.concat([sampled, topup], ignore_index=True)

    return sampled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--out", type=str, default="data/personas_sample.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_load", type=int, default=None, help="Optional cap for faster testing.")
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(DATASET_NAME, split=args.split)
    if args.max_load:
        ds = ds.select(range(min(args.max_load, len(ds))))

    records = [flatten_record(r) for r in ds]
    df = pd.DataFrame(records)
    df = add_standard_columns(df)

    sample = stratified_sample(df, n=args.n, seed=args.seed)
    sample.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"Loaded {len(df):,} rows from {DATASET_NAME}.")
    print(f"Saved {len(sample):,} sampled personas to {args.out}.")
    print("Columns saved:")
    print(list(sample.columns))


if __name__ == "__main__":
    main()
