"""
translate.py — terjemahan dubbing id→en lewat OpenRouter (JSON strict).

Menggantikan polish.py (era konsep lama). Perbedaan prinsip:
  * Output 1:1 dengan segmen input — idx/start/end TIDAK boleh dipecah/digabung
    (manifest mengunci timing; docs/00 §6 aturan 1).
  * Terjemahan harus MUAT di slot: anggaran karakter per segmen dihitung
    eksplisit dan dikirim di prompt; segmen yang masih kelebihan diperbaiki
    satu kali (repair pass "ringkas lagi").
  * Kegagalan LLM = job GAGAL (docs/01 §22) — tidak ada hasil parsial.
  * Istilah/nama diri dipertahankan otomatis (NER oleh LLM, dikembalikan
    sebagai `terms` untuk kartu baca-saja di halaman job — bukan input user).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from .log import Logger
from .srt import CPS, TEMPO_MAX, budget_of, normalize_text

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_RETRIES = 3


class TranslateError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(slots=True)
class InSegment:
    idx: int
    start: float
    end: float
    text: str  # bahasa Indonesia (dari Whisper)


@dataclass(slots=True)
class TranslatedSegment:
    idx: int
    start: float
    end: float
    text_id: str
    text_en: str
    fit: str = "ok"  # ok | tempo | short
    tempo: float = 1.0
    note: str = ""


@dataclass(slots=True)
class TranslateResult:
    segments: list[TranslatedSegment] = field(default_factory=list)
    terms: list[dict[str, str]] = field(default_factory=list)  # [{from,to}] baca-saja
    model_used: str = ""
    elapsed_s: float = 0.0
    tokens_prompt: int = 0
    tokens_completion: int = 0


SYSTEM_PROMPT = """\
You are a professional film dubbing translator (Indonesian → English).
You receive numbered segments with durations. Return STRICT JSON only — no
prose, no markdown fences.

Rules:
1. Translate each segment to natural spoken English. Keep the register of the
   speaker (casual stays casual). Do NOT add information, do NOT summarize
   away meaning.
2. HARD LENGTH BUDGET: each translation must fit its spoken slot. Budget in
   characters is given per segment (slot_seconds × 15 chars/s × 1.25). If the
   natural translation exceeds the budget, CONDENSE it (shorter synonyms,
   drop fillers) — never truncate mid-sentence, never drop the core meaning.
3. Preserve proper nouns as spoken: names of people, places, foods, brands
   (e.g. "rendang" stays "rendang", "Padang" stays "Padang"). Translate common
   nouns normally (e.g. "santan" → "coconut milk").
4. Keep the segment mapping EXACTLY 1:1 — same idx values, never split, merge,
   reorder, or drop segments. Output one object per input segment.
5. Collect preserved/translated notable terms into "terms".

Output schema:
{
  "segments": [ {"idx": 0, "text_en": "..."} ],
  "terms": [ {"from": "rendang", "to": "rendang"} ]
}
"""

REPAIR_PROMPT = """\
These dubbing lines still exceed their character budget. Condense EACH one
further while keeping meaning and natural spoken English. Return STRICT JSON:
{"segments": [{"idx": <int>, "text_en": "..."}]} — exactly one entry per input
line, same idx values.
"""


def translate(
    log: Logger,
    api_key: str,
    segments: list[InSegment],
    model: str,
    *,
    max_batch_chars: int = 6000,
) -> TranslateResult:
    if not api_key:
        raise TranslateError("openrouter_auth", "OpenRouter API key tidak ada")
    if not segments:
        raise TranslateError("llm_bad_json", "Tidak ada segmen untuk diterjemahkan")

    t0 = time.monotonic()
    tokens = {"prompt": 0, "completion": 0}
    out: dict[int, str] = {}
    terms: list[dict[str, str]] = []

    for batch in _batch(segments, max_batch_chars):
        user = _render_batch(batch)
        content, usage = _chat(log, api_key, model, SYSTEM_PROMPT, user, True, "translate")
        tokens["prompt"] += usage["prompt_tokens"]
        tokens["completion"] += usage["completion_tokens"]
        data = _extract_json(log, content)
        if not isinstance(data, dict) or not isinstance(data.get("segments"), list):
            log.warn("jawaban LLM tak sesuai skema — coba repair", raw=repr(content[:600]))
            # satu kali repair attempt untuk JSON rusak
            content, usage = _chat(
                log,
                api_key,
                model,
                "Return ONLY valid JSON matching the requested schema. Fix this response:",
                content[:6000],
                False,
                "translate-repair-json",
            )
            tokens["prompt"] += usage["prompt_tokens"]
            tokens["completion"] += usage["completion_tokens"]
            data = _extract_json(log, content)
        if not isinstance(data, dict) or not isinstance(data.get("segments"), list):
            log.warn("repair JSON juga gagal — job dihentikan", raw=repr(content[:600]))
            raise TranslateError(
                "llm_bad_json", "LLM tidak mengembalikan JSON sesuai skema (2 percobaan)"
            )
        got: set[int] = set()
        for item in data["segments"]:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("idx"))
            except (TypeError, ValueError):
                continue
            text = normalize_text(str(item.get("text_en", "")))
            if idx in got:
                continue
            got.add(idx)
            out[idx] = text
        for t in data.get("terms") or []:
            if isinstance(t, dict) and t.get("from"):
                terms.append(
                    {"from": str(t.get("from"))[:80], "to": str(t.get("to") or t.get("from"))[:80]}
                )
        missing = [s.idx for s in batch if s.idx not in got]
        if missing:
            raise TranslateError(
                "llm_bad_json",
                f"LLM menghilangkan {len(missing)} segmen (idx {missing[:5]}…) — mapping harus 1:1",
            )

    # ---- pass perbaikan untuk yang masih melebihi budget -------------------
    over = [s for s in segments if len(out.get(s.idx, "")) > budget_of(s.start, s.end)]
    if over:
        log.info(f"{len(over)} segmen melebihi budget — repair pass (ringkas lagi)")
        lines = "\n".join(
            f'- idx {s.idx} | budget {budget_of(s.start, s.end)} chars | current: "{out[s.idx]}"'
            for s in over
        )
        try:
            content, usage = _chat(
                log, api_key, model, REPAIR_PROMPT, lines, True, "translate-condense"
            )
            tokens["prompt"] += usage["prompt_tokens"]
            tokens["completion"] += usage["completion_tokens"]
            data = _extract_json(log, content)
            if isinstance(data, dict) and isinstance(data.get("segments"), list):
                for item in data["segments"]:
                    if isinstance(item, dict):
                        try:
                            idx = int(item.get("idx"))
                        except (TypeError, ValueError):
                            continue
                        text = normalize_text(str(item.get("text_en", "")))
                        if text and idx in out and len(text) < len(out[idx]):
                            out[idx] = text
        except TranslateError as e:
            log.warn(
                f"repair pass gagal ({e.message}) — pakai hasil pertama, engine akan clip di 1,25×"
            )

    # ---- rakit hasil + hitung fit/tempo secara deterministik ---------------
    result: list[TranslatedSegment] = []
    for s in segments:
        text_en = out.get(s.idx, "")
        if not text_en:
            raise TranslateError("llm_bad_json", f"Segmen {s.idx} tidak punya terjemahan")
        slot = max(0.1, s.end - s.start)
        est_tempo = len(text_en) / (slot * CPS)  # 1.0 = pas di tempo natural
        budget = budget_of(s.start, s.end)
        note = ""
        if len(text_en) > budget:
            fit, tempo = "tempo", TEMPO_MAX
            note = f"melebihi budget {budget} karakter — dipercepat maksimal, sisa dipotong engine"
        elif est_tempo > 1.0:
            fit, tempo = "tempo", round(min(est_tempo, TEMPO_MAX), 2)
        else:
            fit, tempo = "ok", 1.0
        result.append(
            TranslatedSegment(
                idx=s.idx,
                start=s.start,
                end=s.end,
                text_id=s.text,
                text_en=text_en,
                fit=fit,
                tempo=tempo,
                note=note,
            )
        )

    return TranslateResult(
        segments=result,
        terms=terms[:50],
        model_used=model,
        elapsed_s=round(time.monotonic() - t0, 2),
        tokens_prompt=tokens["prompt"],
        tokens_completion=tokens["completion"],
    )


def _render_batch(batch: list[InSegment]) -> str:
    lines = []
    for s in batch:
        slot = max(0.0, s.end - s.start)
        lines.append(
            f'- idx {s.idx} | slot {slot:.1f}s | budget {budget_of(s.start, s.end)} chars | id: "{s.text}"'
        )
    return "Translate to English:\n" + "\n".join(lines)


def _batch(segments: list[InSegment], max_chars: int) -> list[list[InSegment]]:
    out: list[list[InSegment]] = []
    cur: list[InSegment] = []
    size = 0
    for s in segments:
        need = len(s.text) + 40
        if cur and size + need > max_chars:
            out.append(cur)
            cur, size = [], 0
        cur.append(s)
        size += need
    if cur:
        out.append(cur)
    return out


def _error_detail(r: requests.Response, limit: int = 300) -> str:
    """Body error OpenRouter dalam bentuk ringkas — `error.message` JSON diprioritaskan.

    Tanpa ini, job gagal hanya menampilkan `OpenRouter HTTP 404` di halaman job —
    tidak cukup untuk diagnosis; pesan error harus actionable.
    """
    raw = (r.text or "").strip().replace("\n", " ")
    try:
        j = r.json()
    except ValueError:
        j = None
    if isinstance(j, dict):
        msg = (j.get("error") or {}).get("message")
        if isinstance(msg, str) and msg.strip():
            raw = msg.strip().replace("\n", " ")
    return raw[:limit] or "(body kosong)"


def _data_policy_hint(model: str, detail: str) -> str:
    """Petunjuk konkret bila model `:free` ditolak data policy user OpenRouter.

    Gejala: HTTP 404 `No endpoints found matching your data policy` — terjadi
    karena toggle privacy FREE endpoints di akun user masih OFF. Retry seberapa
    pun tidak akan berhasil sebelum toggle dinyalakan, jadi pesan harus
    menyebut persis ke mana user harus pergi.
    """
    low = detail.lower()
    if model.endswith(":free") and ("data policy" in low or "no endpoints" in low):
        return (
            " | Solusi: model :free butuh izin training di akun OpenRouter — buka"
            " https://openrouter.ai/settings/privacy lalu nyalakan 'Enable prompt logging'"
            " + 'Enable completion logging' di bagian FREE ENDPOINTS, kemudian Coba lagi."
        )
    return ""


def _chat(
    log: Logger,
    api_key: str,
    model: str,
    system: str,
    user: str,
    json_mode: bool,
    tag: str,
) -> tuple[str, dict[str, int]]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "max_tokens": 16000,
        # Model reasoning (mis. minimax-m3.1) bisa menghabiskan max_tokens untuk
        # token reasoning sehingga `content` kosong/terpotong → llm_bad_json.
        # Matikan; model tanpa dukungan reasoning mengabaikan parameter ini
        # (OpenRouter men-drop param yang tak didukung provider).
        "reasoning": {"enabled": False},
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    last = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(
                OPENROUTER_URL,
                headers={
                    "authorization": f"Bearer {api_key}",
                    "content-type": "application/json",
                    "http-referer": "https://github.com/inggrisin",
                    "x-title": "inggrisin",
                    "user-agent": "inggrisin-engine",
                },
                json=body,
                timeout=(15, 300),
            )
        except requests.RequestException as e:
            last = f"network {type(e).__name__}"
            if attempt == MAX_RETRIES:
                raise TranslateError("openrouter_failed", last) from e
            time.sleep(2**attempt)
            continue

        if r.status_code == 200:
            j = r.json()
            choice = (j.get("choices") or [{}])[0]
            content = (choice.get("message") or {}).get("content") or ""
            finish = choice.get("finish_reason")
            if finish not in (None, "stop"):
                log.warn(f"LLM finish_reason={finish} — jawaban mungkin terpotong", tag=tag)
            usage = j.get("usage") or {}
            return str(content), {
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
            }
        if (
            r.status_code == 400
            and "reasoning" in body
            and "reasoning is mandatory" in _error_detail(r).lower()
        ):
            # Endpoint tertentu (mis. minimax :free) MEWAJIBKAN reasoning dan
            # menolak dimatikan → buang param-nya dan coba ulang segera.
            body.pop("reasoning")
            log.warn("endpoint mewajibkan reasoning — ulangi tanpa mematikannya", tag=tag)
            continue
        if r.status_code in (401, 402, 403):
            detail = _error_detail(r)
            code = "openrouter_no_credit" if r.status_code == 402 else "openrouter_auth"
            raise TranslateError(
                code,
                f"OpenRouter HTTP {r.status_code} (key/kredit): {detail}"
                f"{_data_policy_hint(model, detail)}",
            )
        if r.status_code == 429 or 500 <= r.status_code < 600:
            detail = _error_detail(r)
            last = f"HTTP {r.status_code}: {detail}"
            if attempt == MAX_RETRIES:
                raise TranslateError(
                    "openrouter_failed",
                    f"OpenRouter {last} setelah {MAX_RETRIES} percobaan"
                    f"{_data_policy_hint(model, detail)}",
                )
            time.sleep(2**attempt)
            continue
        detail = _error_detail(r)
        raise TranslateError(
            "openrouter_failed",
            f"OpenRouter HTTP {r.status_code}: {detail}{_data_policy_hint(model, detail)}",
        )

    raise TranslateError("openrouter_failed", last or "gagal tanpa alasan")


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _extract_json(log: Logger, text: str) -> Any | None:
    cleaned = _FENCE.sub("", (text or "").strip()).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as e:
            log.warn(f"JSON extraction gagal: {e}")
    return None
