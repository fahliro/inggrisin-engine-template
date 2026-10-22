"""
groq_llm.py — terjemahan Inggris via Groq chat completion (gpt-oss).

docs/01 §36: arsitektur build pertama = Whisper 1× (transcribe → teks sumber +
timing kanonik) + gpt-oss (terjemah 1:1 per segmen). LLM bukan lagi fallback
(§32) melainkan PENERJEMAH PRIMER: text_en lahir langsung dari segmen sumber,
sehingga align lintas-pass (§29/§34) punah dan baris duplikat mustahil by
construction.

Ketahanan (cermin §31 di Worker):
  - rantai model: gpt-oss-120b → gpt-oss-20b (404 / model tidak tersedia);
  - video panjang: segmen di-chunk ≤60 segmen / ≤10K char; antar-chunk diberi
    konteks 3 baris terakhir agar istilah & nada konsisten;
  - 429 per-menit → hormati header retry-after, fallback backoff + jitter;
  - 429 kuota HARIAN → fail-fast `groq_daily_quota` (nol retry sia-sia);
  - 400 param ditolak → relaksasi 2 tahap (reasoning_effort, response_format).

Kontrak: output 1:1 per idx (timing manifest tidak disentuh — docs/00 §6).
Kegagalan LLM = pengecualian (pemanggil memutuskan gagal job — docs/01 §22).
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from typing import Any

import requests

from .log import Logger

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_PRIMARY = "openai/gpt-oss-120b"
MODEL_FALLBACK = "openai/gpt-oss-20b"
MAX_RETRIES = 4
MAX_SEGS_PER_CHUNK = 60
MAX_CHARS_PER_CHUNK = 10_000
CONTEXT_LINES = 3  # baris terjemahan sebelumnya dikirim sebagai konteks chunk


class GroqLlmError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _backoff(log: Logger, attempt: int, reason: str, wait: float | None = None) -> None:
    if wait is None:
        wait = 2.0**attempt + random.uniform(0, 1.0)
    log.warn(f"groq_llm: {reason} — retry {attempt}/{MAX_RETRIES} dalam {wait:.1f}s")
    time.sleep(min(wait, 60.0))


_SYSTEM_EN = "\n".join(
    [
        "You are a subtitle translator for dubbed videos (Indonesian → English).",
        "RULES:",
        "- Output ONLY English, natural spoken style.",
        "- The segments are CONSECUTIVE subtitle lines from one video; a segment may",
        "  start or end mid-sentence. Keep terminology and tone consistent.",
        "- Translate EVERY segment 1:1. Keep the given idx. Never split, merge, drop,",
        "  or repeat a segment.",
        "- Each translation MUST fit its character budget (max chars, spaces included).",
        '- Return STRICT JSON only: {"segments":[{"idx":<int>,"text_en":"<string>"}]}.',
    ]
)

_SYSTEM_ID = "\n".join(
    [
        "You are a subtitle translator for dubbed videos (English → Indonesian).",
        "RULES:",
        "- Output ONLY Indonesian, gaya lisan natural.",
        "- Translate EVERY segment 1:1. Keep the given idx. Never split or merge.",
        '- Return STRICT JSON only: {"segments":[{"idx":<int>,"text_id":"<string>"}]}.',
    ]
)


def _chunk_items(items: list[tuple[int, str, int]]) -> list[list[tuple[int, str, int]]]:
    """Pecah berurutan jadi chunk ≤ MAX_SEGS_PER_CHUNK segmen / ≤ MAX_CHARS_PER_CHUNK."""
    chunks: list[list[tuple[int, str, int]]] = []
    cur: list[tuple[int, str, int]] = []
    chars = 0
    for it in items:
        if cur and (len(cur) >= MAX_SEGS_PER_CHUNK or chars + len(it[1]) > MAX_CHARS_PER_CHUNK):
            chunks.append(cur)
            cur, chars = [], 0
        cur.append(it)
        chars += len(it[1])
    if cur:
        chunks.append(cur)
    return chunks


def translate_segments(
    log: Logger,
    api_key: str,
    items: list[tuple[int, str, int]],
    model: str | None = None,
    target: str = "en",
    on_chunk: Callable[[int, int], None] | None = None,
) -> dict[int, str]:
    """Terjemahkan [(idx, teks, budget_chars)] → {idx: teks hasil} via Groq chat.

    target="en": Indonesia→Inggris (penerjemah primer build pertama — §36).
    target="id": Inggris→Indonesia (back-translate; dipertahankan untuk alat uji).
    model=None → rantai MODEL_PRIMARY → MODEL_FALLBACK; model eksplisit → tunggal.
    Video panjang otomatis di-chunk; `on_chunk(i, n)` utk progress UI.
    """
    if not items:
        return {}
    chunks = _chunk_items(items)
    models = [model] if model else [MODEL_PRIMARY, MODEL_FALLBACK]
    mi = 0
    out: dict[int, str] = {}
    context: list[tuple[str, str]] = []
    for ci, chunk in enumerate(chunks, 1):
        while True:
            try:
                part = _translate_chunk(log, api_key, chunk, models[mi], target, context)
                break
            except GroqLlmError as e:
                if e.code == "llm_model_not_found" and mi + 1 < len(models):
                    log.warn(
                        f"groq_llm: model {models[mi]} tidak tersedia — "
                        f"beralih ke {models[mi + 1]}"
                    )
                    mi += 1
                    continue
                raise
        out.update(part)
        if target == "en":
            for idx, text, _ in chunk[-CONTEXT_LINES:]:
                if text.strip() and out.get(idx):
                    context.append((text, out[idx]))
            context = context[-CONTEXT_LINES:]
        if on_chunk is not None:
            on_chunk(ci, len(chunks))
    return out


def _translate_chunk(
    log: Logger,
    api_key: str,
    items: list[tuple[int, str, int]],
    model: str,
    target: str,
    context: list[tuple[str, str]],
) -> dict[int, str]:
    """Satu panggilan chat wajib-JSON utk satu chunk + retry/backoff (429/5xx/network)."""
    if target == "id":
        user = (
            "Translate these segments to Indonesian. Each line: idx, English text.\n"
            + "\n".join(f'- idx={idx} en="{text}"' for idx, text, _ in items)
        )
        system = _SYSTEM_ID
        out_field = "text_id"
    else:
        parts = ["Translate these segments to English. Each line: idx, budget, Indonesian text."]
        if context:
            parts.append(
                "Previous lines (already translated — context only, do NOT translate or repeat):"
            )
            parts.extend(f'- id="{src}" → en="{dst}"' for src, dst in context)
        parts.extend(f'- idx={idx} budget={budget} id="{text}"' for idx, text, budget in items)
        user = "\n".join(parts)
        system = _SYSTEM_EN
        out_field = "text_en"

    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.3,
        "reasoning_effort": "low",
        "response_format": {"type": "json_object"},
    }

    last = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(
                GROQ_CHAT_URL,
                headers={
                    "authorization": f"Bearer {api_key}",
                    "content-type": "application/json",
                    "user-agent": "inggrisin-engine",
                },
                json=body,
                timeout=(30, 300),
            )
        except requests.RequestException as e:
            last = f"network {type(e).__name__}"
            if attempt == MAX_RETRIES:
                raise GroqLlmError("llm_failed", f"Groq chat gagal: {last}") from e
            _backoff(log, attempt, last)
            continue

        if r.status_code == 200:
            content = _content_of(r)
            out = _parse_segments(content, out_field)
            if out is None:
                last = "jawaban bukan JSON sesuai skema"
                if attempt == MAX_RETRIES:
                    raise GroqLlmError("llm_bad_json", last)
                _backoff(log, attempt, last)
                continue
            missing = [idx for idx, _, _ in items if idx not in out]
            if missing:
                last = f"LLM tidak mengembalikan idx {missing[:5]}"
                if attempt == MAX_RETRIES:
                    raise GroqLlmError("llm_bad_json", last)
                _backoff(log, attempt, last)
                continue
            log.info(f"groq_llm: {len(out)} segmen diterjemahkan via {model}")
            return out

        last = f"HTTP {r.status_code}: {r.text[:200]}"
        if r.status_code == 429 and "daily" in (r.text or "").lower():
            # kuota harian habis — retry membakar waktu tanpa hasil (§31: fail-fast)
            raise GroqLlmError(
                "groq_daily_quota",
                "Kuota harian Groq untuk model ini habis — coba lagi besok "
                "(atau naikkan tier kunci Groq-mu di console.groq.com)",
            )
        if r.status_code == 404 or (r.status_code == 400 and "model_not_found" in r.text):
            raise GroqLlmError("llm_model_not_found", last)
        if r.status_code == 400 and "reasoning_effort" in body:
            del body["reasoning_effort"]
            continue
        if r.status_code == 400 and "response_format" in body:
            del body["response_format"]
            continue
        if r.status_code == 429 or r.status_code >= 500:
            if attempt == MAX_RETRIES:
                raise GroqLlmError("llm_rate_limit" if r.status_code == 429 else "llm_failed", last)
            ra = (getattr(r, "headers", None) or {}).get("retry-after")
            wait = None
            try:
                wait = float(ra) if ra else None
            except ValueError:
                wait = None
            _backoff(log, attempt, last, wait=wait)
            continue
        raise GroqLlmError("llm_failed", last)

    raise GroqLlmError("llm_failed", f"Groq chat gagal setelah {MAX_RETRIES} percobaan: {last}")


def _content_of(r: requests.Response) -> str:
    try:
        j = r.json()
        return str(j.get("choices", [{}])[0].get("message", {}).get("content") or "")
    except (ValueError, IndexError, AttributeError):
        return ""


def _parse_segments(content: str, out_field: str = "text_en") -> dict[int, str] | None:
    cleaned = (content or "").replace("```json", "").replace("```", "").strip()
    try:
        j = json.loads(cleaned)
    except ValueError:
        s, e = cleaned.find("{"), cleaned.rfind("}")
        if s < 0 or e <= s:
            return None
        try:
            j = json.loads(cleaned[s : e + 1])
        except ValueError:
            return None
    out: dict[int, str] = {}
    for seg in (j or {}).get("segments") or []:
        idx = seg.get("idx")
        text = seg.get(out_field)
        if isinstance(idx, int) and isinstance(text, str):
            out[idx] = " ".join(text.split())
    return out or None
