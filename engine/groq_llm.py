"""
groq_llm.py — terjemahan fallback via Groq chat completion (gpt-oss-20b).

Hanya dipakai bila dua pass Whisper bertentangan DAN pass sumber yang menang
(docs/01 §32): /audio/translations bisa BERHALUSINASI teks Inggris panjang di
atas BGM/keheningan, sementara transcribe (forced-id maupun auto-detect)
sepakat ucapannya pendek. Mempublikasikan halusinasi = subtitle dinding teks
yang tidak mengucapkan apa pun — maka teks sumber diterjemahkan via LLM:
murah, deterministik (temperature rendah), kunci Groq user yang sama (BYOK).

Kontrak: output 1:1 per idx (timing manifest tidak disentuh — docs/00 §6).
Kegagalan LLM = pengecualian (pemanggil memutuskan gagal job — docs/01 §22).
"""

from __future__ import annotations

import json
import random
import time
from typing import Any

import requests

from .log import Logger

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"
MAX_RETRIES = 4
MAX_CHARS_PER_CALL = 12_000  # satu panggilan; di atas ini caller wajib gagal jelas


class GroqLlmError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _backoff(log: Logger, attempt: int, reason: str) -> None:
    wait = 2.0**attempt + random.uniform(0, 1.0)
    log.warn(f"groq_llm: {reason} — retry {attempt}/{MAX_RETRIES} dalam {wait:.1f}s")
    time.sleep(min(wait, 30.0))


_SYSTEM_EN = "\n".join(
    [
        "You are a subtitle translator for dubbed videos (Indonesian → English).",
        "RULES:",
        "- Output ONLY English, natural spoken style.",
        "- Translate EVERY segment 1:1. Keep the given idx. Never split or merge.",
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


def translate_segments(
    log: Logger,
    api_key: str,
    items: list[tuple[int, str, int]],
    model: str = MODEL,
    target: str = "en",
) -> dict[int, str]:
    """Terjemahkan [(idx, text_id, budget_chars)] → {idx: text_en} via Groq chat.

    target="en": Indonesia→Inggris (fallback bila translate Whisper halusinasi).
    target="id": Inggris→Indonesia (back-translate untuk teks referensi bila
    segmentasi kanonik diambil dari pass translations — §32 v2).
    Satu panggilan chat completion wajib-JSON; retry + exponential backoff untuk
    429/5xx/network; 400 param ditolak → relaksasi (reasoning_effort lalu
    response_format). Semua idx WAJIB hadir di jawaban, selain itu GroqLlmError.
    """
    total = sum(len(t) for _, t, _ in items)
    if total > MAX_CHARS_PER_CALL:
        raise GroqLlmError(
            "llm_payload_too_large",
            f"teks sumber {total} char melebihi batas fallback LLM {MAX_CHARS_PER_CALL}",
        )
    if target == "id":
        user = (
            "Translate these segments to Indonesian. Each line: idx, English text.\n"
            + "\n".join(f'- idx={idx} en="{text}"' for idx, text, _ in items)
        )
        system = _SYSTEM_ID
        out_field = "text_id"
    else:
        user = (
            "Translate these segments to English. Each line: idx, budget, Indonesian text.\n"
            + "\n".join(f'- idx={idx} budget={budget} id="{text}"' for idx, text, budget in items)
        )
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
        if r.status_code == 400 and "reasoning_effort" in body:
            del body["reasoning_effort"]
            continue
        if r.status_code == 400 and "response_format" in body:
            del body["response_format"]
            continue
        if r.status_code == 429 or r.status_code >= 500:
            if attempt == MAX_RETRIES:
                raise GroqLlmError("llm_rate_limit" if r.status_code == 429 else "llm_failed", last)
            _backoff(log, attempt, last)
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
