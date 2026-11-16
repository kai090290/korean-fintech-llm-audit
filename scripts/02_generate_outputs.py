"""
Generate model responses for each prompt (OpenAI, Anthropic, Google Gemini).

Requires API keys in the environment (see `.env.example`):
  OPENAI_API_KEY
  ANTHROPIC_API_KEY  (optional — if you use anthropic/...)
  GOOGLE_API_KEY or GEMINI_API_KEY (optional — if you use google/...)

Model selection
---------------
  --model gpt-4o-mini
      Shorthand for OpenAI (same as ``openai/gpt-4o-mini``).

  --models "openai/gpt-4o,clova/HCX-005,openrouter/openai/gpt-4o,anthropic/claude-sonnet-4-20250514,google/gemini-2.0-flash"
      Comma-separated; each grid row is called once per model. Keys live only in ``.env``.

  Providers: ``openai/``, ``anthropic/``, ``google/``, ``clova/`` (NAVER HyperCLOVA via CLOVA Studio),
  ``openrouter/`` (OpenRouter catalog; use ``openrouter/<openrouter-model-slug>``).

  Resolution order: ``--models`` > env ``AUDIT_ADVICE_MODELS`` >
  ``config/advice_models.txt`` (or ``--models-file``) > ``--model`` > default.

  ``--strict-models-file`` → only non-comment lines in ``--models-file`` (ignores env/CLI lists).

  Parallelism: ``--workers N`` (default 4) runs concurrent HTTP calls per model; use ``--workers 1`` if providers rate-limit (429).

Usage::

    .venv/bin/python scripts/02_generate_outputs.py \\
        --grid data/prompt_grid.csv --out data/model_outputs.csv \\
        --models "openai/gpt-4o,openai/gpt-4o-mini"
"""

from __future__ import annotations

import argparse
import hashlib
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from llm_providers import chat_completion, expand_models_arg

load_dotenv()


def _read_models_file(path: Path) -> str | None:
    """Return comma-separated model specs from a line-oriented config file."""
    if not path.is_file():
        return None
    specs: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        specs.append(line)
    return ",".join(specs) if specs else None


def _resolve_models_csv(
    *,
    cli_models: str | None,
    cli_model: str | None,
    models_file: Path,
    strict_models_file: bool,
) -> tuple[str | None, str | None]:
    """
    Decide which source feeds expand_models_arg(models_csv, single_model).

    With ``strict_models_file``: use only non-comment lines from ``models_file``;
    ``--models``, ``AUDIT_ADVICE_MODELS``, and ``--model`` are ignored.

    Otherwise priority: ``--models`` > ``AUDIT_ADVICE_MODELS`` > ``--models-file`` >
    ``--model`` > (file default path only if it exists).
    """
    if strict_models_file:
        from_file = _read_models_file(models_file)
        if not from_file:
            raise SystemExit(
                f"--strict-models-file: no models found in {models_file} "
                "(add one openrouter/... or provider/model per non-comment line)."
            )
        return from_file, None
    if cli_models and str(cli_models).strip():
        return str(cli_models).strip(), None
    env_block = (os.getenv("AUDIT_ADVICE_MODELS") or "").strip()
    if env_block:
        return env_block, None
    from_file = _read_models_file(models_file)
    if from_file:
        return from_file, None
    return None, cli_model


def _one_generation(
    r: pd.Series,
    provider: str,
    api_model: str,
    canonical: str,
    *,
    mock: bool,
    temperature: float,
    max_tokens: int,
) -> tuple[tuple, str, str, dict]:
    """Return (done_key, status, error, row_dict) for one grid row."""
    key = (r["persona_id"], r["scenario_id"], canonical)
    try:
        if mock:
            h = hashlib.sha256(
                f"{r['persona_id']}|{r['scenario_id']}|{canonical}".encode()
            ).hexdigest()[:4]
            text = (
                "[MOCK 응답 — API 미사용] "
                "공식 금융기관·앱 고객센터 등 신뢰할 수 있는 채널로 확인하시고, "
                "고액·급한 연락·링크 요청은 사기 가능성이 있으니 링크 클릭과 "
                "인증번호·비밀번호 공유를 피하세요. 본인의 상환능력·수수료에 "
                f"주의하시기 바랍니다. (표식: {h}, 모델: {canonical})"
            )
            status = "ok"
            err = ""
        else:
            text = chat_completion(
                provider,
                api_model,
                system_prompt=r["system_prompt"],
                user_prompt=r["user_prompt"],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            status = "ok"
            err = ""
    except Exception as e:
        text = ""
        status = "error"
        err = repr(e)

    rec = {
        "persona_id": r["persona_id"],
        "scenario_id": r["scenario_id"],
        "scenario_title": r["scenario_title"],
        "model_name": canonical,
        "temperature": temperature,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "system_prompt": r["system_prompt"],
        "user_prompt": r["user_prompt"],
        "expected_safety_elements": r["expected_safety_elements"],
        "response_text": text,
        "status": status,
        "error": err,
    }
    return key, status, err, rec


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate FinTech advice responses for each persona × scenario × model.",
    )
    parser.add_argument("--grid", type=str, default="data/prompt_grid.csv")
    parser.add_argument("--out", type=str, default="data/model_outputs.csv")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Single OpenAI-style shorthand (e.g. gpt-4o-mini). Ignored if --models is set.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help=(
            "Comma-separated provider/model list, e.g. "
            '"openai/gpt-4o,openrouter/openai/gpt-4o". '
            "Overrides AUDIT_ADVICE_MODELS and --models-file."
        ),
    )
    parser.add_argument(
        "--models-file",
        type=str,
        default="config/advice_models.txt",
        help="Line-oriented list of models (overridden by --models and AUDIT_ADVICE_MODELS).",
    )
    parser.add_argument(
        "--strict-models-file",
        action="store_true",
        help=(
            "Use only --models-file lines; ignore --models, --model, and AUDIT_ADVICE_MODELS. "
            "Ensures no extra models run from env/CLI."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel API calls per model (I/O bound). Use 1 to force sequential. Lower if you see 429/rate limits.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Max output tokens (Anthropic/Gemini); OpenAI uses provider default caps.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional first-N grid rows (pilot).")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between API calls.")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Skip API calls; write deterministic placeholder Korean text (testing only).",
    )
    args = parser.parse_args()

    models_csv, single = _resolve_models_csv(
        cli_models=args.models,
        cli_model=args.model,
        models_file=Path(args.models_file),
        strict_models_file=args.strict_models_file,
    )
    model_triples = expand_models_arg(models_csv, single)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    grid = pd.read_csv(args.grid)
    if args.limit:
        grid = grid.head(args.limit).copy()

    if Path(args.out).exists() and not args.mock:
        done = pd.read_csv(args.out)
        done_keys = set(zip(done["persona_id"], done["scenario_id"], done["model_name"]))
        rows = done.to_dict("records")
    else:
        done_keys = set()
        rows = []

    row_lock = threading.Lock()
    workers = max(1, int(args.workers))

    for provider, api_model, canonical in model_triples:
        label = canonical
        pending = [
            r
            for _, r in grid.iterrows()
            if (r["persona_id"], r["scenario_id"], canonical) not in done_keys
        ]

        def _run_row(r: pd.Series) -> None:
            _, _, _, rec = _one_generation(
                r,
                provider,
                api_model,
                canonical,
                mock=args.mock,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            if args.sleep:
                time.sleep(args.sleep)
            key = (rec["persona_id"], rec["scenario_id"], canonical)
            with row_lock:
                rows.append(rec)
                done_keys.add(key)
                pd.DataFrame(rows).to_csv(args.out, index=False, encoding="utf-8-sig")

        if workers == 1:
            for r in tqdm(pending, total=len(pending), desc=label):
                _run_row(r)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_run_row, r) for r in pending]
                for fut in tqdm(as_completed(futures), total=len(futures), desc=label):
                    fut.result()

        print(f"Finished model {canonical}")

    print(f"Saved responses to {args.out} (models: {[c for _, _, c in model_triples]})")


if __name__ == "__main__":
    main()
