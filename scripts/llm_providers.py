"""
Multi-provider chat completion for the audit pipeline.

Supported providers (prefix in ``--models``):

  openai/     — OpenAI Chat Completions (``OPENAI_API_KEY``; optional ``OPENAI_BASE_URL``)
  anthropic/  — Anthropic Messages API (``ANTHROPIC_API_KEY``; ``pip install anthropic``)
  google/     — Google Gemini (``GOOGLE_API_KEY`` or ``GEMINI_API_KEY``)
  clova/      — NAVER HyperCLOVA X via **OpenAI-compatible** CLOVA Studio
                (``CLOVA_STUDIO_API_KEY``; optional ``CLOVA_STUDIO_OPENAI_BASE_URL``)
  openrouter/ — OpenRouter unified API (``OPENROUTER_API_KEY``) — any model id in their catalog

Model specs
-----------
  - ``openai/gpt-4o``  /  bare ``gpt-4o-mini`` → ``openai/gpt-4o-mini``
  - ``clova/HCX-005``  (use the engine id from your CLOVA Studio console / docs)
  - ``openrouter/anthropic/claude-3.5-sonnet``  (exact slug from openrouter.ai/models)

The canonical ``model_name`` in CSV is always ``provider/api_model_id``.
"""

from __future__ import annotations

import os
from typing import Tuple


KNOWN_PROVIDERS = ("openai", "anthropic", "google", "clova", "openrouter")


def parse_model_spec(spec: str) -> Tuple[str, str, str]:
    """
    Return (provider, api_model_id, canonical_name).

    ``canonical_name`` is ``provider/api_model_id`` and is what we write to ``model_name``.
    """
    s = (spec or "").strip()
    if not s:
        raise ValueError("empty model spec")

    if "/" in s:
        prov, mid = s.split("/", 1)
        prov = prov.lower().strip()
        mid = mid.strip()
        if prov not in KNOWN_PROVIDERS:
            raise ValueError(
                f"Unknown provider {prov!r}. Use one of: {', '.join(KNOWN_PROVIDERS)}"
            )
        if not mid:
            raise ValueError(f"Missing model id after provider in {spec!r}")
        return prov, mid, f"{prov}/{mid}"

    # Bare id → OpenAI (backward compatible with older CLI)
    return "openai", s, f"openai/{s}"


def expand_models_arg(models_csv: str | None, single_model: str | None) -> list[Tuple[str, str, str]]:
    """Build ordered list of (provider, api_id, canonical) from CLI args."""
    if models_csv and str(models_csv).strip():
        specs = [p.strip() for p in str(models_csv).split(",") if p.strip()]
    elif single_model and str(single_model).strip():
        specs = [single_model.strip()]
    else:
        specs = ["gpt-4o-mini"]
    seen: set[str] = set()
    out: list[Tuple[str, str, str]] = []
    for sp in specs:
        prov, mid, canon = parse_model_spec(sp)
        if canon in seen:
            continue
        seen.add(canon)
        out.append((prov, mid, canon))
    return out


def chat_completion(
    provider: str,
    api_model: str,
    *,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int = 8192,
) -> str:
    if provider == "openai":
        return _openai_chat(api_model, system_prompt, user_prompt, temperature, max_tokens)
    if provider == "anthropic":
        return _anthropic_chat(api_model, system_prompt, user_prompt, temperature, max_tokens)
    if provider == "google":
        return _google_gemini_chat(api_model, system_prompt, user_prompt, temperature, max_tokens)
    if provider == "clova":
        return _clova_openai_compat_chat(api_model, system_prompt, user_prompt, temperature, max_tokens)
    if provider == "openrouter":
        return _openrouter_chat(api_model, system_prompt, user_prompt, temperature, max_tokens)
    raise ValueError(f"unsupported provider: {provider}")


def _openai_chat(
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
) -> str:
    from openai import OpenAI

    client = OpenAI()
    kwargs = dict(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    if max_tokens and max_tokens > 0:
        kwargs["max_tokens"] = max_tokens
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def _clova_openai_compat_chat(
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """
    NAVER CLOVA Studio — OpenAI-compatible Chat Completions.

    Docs: https://api.ncloud-docs.com/docs/en/clovastudio-openaicompatibility

    Env:
      CLOVA_STUDIO_API_KEY   (or NAVER_CLOVA_STUDIO_API_KEY)
      CLOVA_STUDIO_OPENAI_BASE_URL  (optional; default NAVER stream endpoint)
    """
    from openai import OpenAI

    api_key = os.getenv("CLOVA_STUDIO_API_KEY") or os.getenv("NAVER_CLOVA_STUDIO_API_KEY")
    if not api_key:
        raise RuntimeError(
            "clova/... requires CLOVA_STUDIO_API_KEY (NAVER Cloud CLOVA Studio API key)."
        )
    base = (
        os.getenv("CLOVA_STUDIO_OPENAI_BASE_URL")
        or "https://clovastudio.stream.ntruss.com/v1/openai"
    )
    base = base.rstrip("/")
    client = OpenAI(api_key=api_key, base_url=base)
    # CLOVA defaults max output fairly low; always pass max_tokens for long Korean answers.
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content or ""


def _openrouter_chat(
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """
    OpenRouter — OpenAI-compatible gateway to many providers (one API key).

    https://openrouter.ai/docs

    Env:
      OPENROUTER_API_KEY
      OPENROUTER_BASE_URL   (optional, default https://openrouter.ai/api/v1)
      OPENROUTER_HTTP_REFERER / OPENROUTER_APP_NAME — optional attribution headers
    """
    from openai import OpenAI

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("openrouter/... requires OPENROUTER_API_KEY in the environment.")
    base = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    default_headers: dict[str, str] = {}
    ref = os.getenv("OPENROUTER_HTTP_REFERER")
    if ref:
        default_headers["HTTP-Referer"] = ref
    title = os.getenv("OPENROUTER_APP_NAME")
    if title:
        default_headers["X-Title"] = title

    client_kwargs: dict = {"api_key": api_key, "base_url": base}
    if default_headers:
        client_kwargs["default_headers"] = default_headers

    client = OpenAI(**client_kwargs)
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content or ""


def _anthropic_chat(
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
) -> str:
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError(
            "Anthropic provider requires: pip install anthropic "
            "(and ANTHROPIC_API_KEY in the environment)."
        ) from e

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    parts: list[str] = []
    for block in msg.content:
        if block.type == "text":
            parts.append(block.text)
    return "".join(parts)


def _google_gemini_chat(
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
) -> str:
    try:
        import google.generativeai as genai
    except ImportError as e:
        raise RuntimeError(
            "Google provider requires: pip install google-generativeai "
            "(and GOOGLE_API_KEY or GEMINI_API_KEY)."
        ) from e

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY for Gemini.")
    genai.configure(api_key=api_key)

    gm = genai.GenerativeModel(
        model,
        system_instruction=system,
    )
    response = gm.generate_content(
        user,
        generation_config={
            "temperature": float(temperature),
            "max_output_tokens": int(max_tokens),
        },
    )
    try:
        text = (response.text or "").strip()
    except (ValueError, AttributeError):
        text = ""
    if text:
        return text
    # Blocked or empty — surface reason if available
    cand = getattr(response, "candidates", None) or []
    if cand and getattr(cand[0], "finish_reason", None):
        return f"[GEMINI_EMPTY_RESPONSE finish_reason={cand[0].finish_reason}]"
    return ""
