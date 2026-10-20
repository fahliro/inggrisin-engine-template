"""
Transkripsi lewat Groq Whisper API.

Batasan yang menentukan desain:
  - ukuran file maks 25 MB (free tier) / 100 MB (dev tier)
  - format didukung: flac, mp3, m4a, mpeg, mpga, ogg, opus, wav, webm
  - `whisper-large-v3-turbo` ≈ $0.04/jam, jauh lebih cepat dari large-v3

Karena audio bisa di-chunk (lihat audio.py), setiap chunk ditranskrip terpisah
lalu digabung dengan OFFSET WAKU. Offset dihitung dari durasi chunk sebelumnya
— bukan dari index × chunk_seconds — supaya akurat walau chunk terakhir pendek.

Retry: hanya untuk 429/5xx, dengan exponential backoff + jitter. 401/400
tidak di-retry (buang waktu & kuota user).
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

from .audio import probe
from .log import Logger

GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
MAX_RETRIES = 4
# batas yang kita pakai sendiri (paling ketat) — lihat audio.TARGET_AUDIO_BYTES
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


class TranscribeError(Exception):
    def __init__(self, code: str, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


@dataclass(slots=True)
class Word:
    start: float
    end: float
    word: str


@dataclass(slots=True)
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass(slots=True)
class Transcript:
    text: str
    language: str | None
    duration_s: float
    segments: list[Segment] = field(default_factory=list)
    elapsed_s: float = 0.0
    audio_seconds_billed: float = 0.0


def transcribe_file(
    log: Logger,
    api_key: str,
    path: Path,
    model: str = "whisper-large-v3-turbo",
    language: str | None = "en",
    prompt: str | None = None,
    temperature: float = 0.0,
) -> Transcript:
    """Transkrip SATU file audio yang ukurannya sudah di bawah batas."""
    size = path.stat().st_size
    if size > MAX_UPLOAD_BYTES:
        raise TranscribeError(
            "groq_file_too_large",
            f"File {size / 1e6:.1f} MB melebihi batas Groq 25 MB. Pipeline chunking seharusnya "
            "menangani ini — periksa audio.needs_chunking().",
        )
    if size == 0:
        raise TranscribeError("groq_file_too_large", "File audio kosong")

    t0 = time.monotonic()
    mime = _mime_for(path)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with path.open("rb") as fh:
                files = {"file": (path.name, fh, mime)}
                data: dict[str, str | float] = {
                    "model": model,
                    "response_format": "verbose_json",
                    "timestamp_granularities[]": "segment",
                    "temperature": temperature,
                }
                if language:
                    data["language"] = language
                if prompt:
                    # prompt memandu gaya & istilah; maks 244 token
                    data["prompt"] = prompt[:800]

                r = requests.post(
                    GROQ_URL,
                    headers={"authorization": f"Bearer {api_key}", "user-agent": "inggrisin-engine"},
                    files=files,
                    data=data,
                    timeout=(30, 600),
                )
        except requests.RequestException as e:
            if attempt == MAX_RETRIES:
                raise TranscribeError("groq_failed", f"network error: {type(e).__name__}", True) from e
            _backoff(log, attempt, f"network {type(e).__name__}")
            continue

        if r.status_code == 200:
            elapsed = time.monotonic() - t0
            return _parse(log, r, elapsed)

        # --- pemetaan error yang bisa ditindaklanjuti ---
        if r.status_code in (401, 403):
            raise TranscribeError("groq_auth", f"Groq menolak API key (HTTP {r.status_code}). Perbarui key di dashboard.")
        if r.status_code == 413:
            raise TranscribeError("groq_file_too_large", "Groq menolak: file terlalu besar (HTTP 413)")
        if r.status_code == 422:
            raise TranscribeError("groq_failed", f"Groq tidak bisa memproses audio (HTTP 422): {_safe_detail(r)}")
        if r.status_code == 429:
            # rate limit → retryable, tapi jangan buang banyak percobaan
            if attempt == MAX_RETRIES:
                raise TranscribeError("groq_rate_limit", "Groq rate limit tercapai setelah beberapa percobaan", True)
            _backoff(log, attempt, "429 rate limit", base=4.0)
            continue
        if 500 <= r.status_code < 600:
            if attempt == MAX_RETRIES:
                raise TranscribeError("groq_failed", f"Groq error server (HTTP {r.status_code})", True)
            _backoff(log, attempt, f"{r.status_code} server error")
            continue

        raise TranscribeError("groq_failed", f"Groq HTTP {r.status_code}: {_safe_detail(r)}")

    raise TranscribeError("groq_failed", "transkripsi gagal tanpa status jelas")


def _parse(log: Logger, r: requests.Response, elapsed: float) -> Transcript:
    j = r.json()
    segments: list[Segment] = []
    for s in j.get("segments") or []:
        words = [
            Word(start=float(w.get("start", 0)), end=float(w.get("end", 0)), word=str(w.get("word", "")))
            for w in (s.get("words") or [])
        ]
        segments.append(
            Segment(
                start=float(s.get("start", 0.0)),
                end=float(s.get("end", 0.0)),
                text=str(s.get("text", "")).strip(),
                words=words,
            )
        )

    text = str(j.get("text") or "").strip()
    duration = float(j.get("duration") or 0.0)
    lang = j.get("language")
    log.info(
        f"transkrip: {len(text)} char, {len(segments)} segmen, {elapsed:.1f}s "
        f"(RTF {elapsed / max(duration, 0.001):.2f})",
        language=lang,
    )
    return Transcript(
        text=text,
        language=lang,
        duration_s=duration,
        segments=segments,
        elapsed_s=round(elapsed, 2),
        audio_seconds_billed=duration,
    )


def transcribe_chunks(
    log: Logger,
    api_key: str,
    chunks: list[Path],
    model: str,
    language: str | None = "en",
    prompt: str | None = None,
    on_chunk: callable | None = None,
) -> Transcript:
    """
    Transkrip banyak chunk lalu gabung dengan offset waktu yang benar.

    Offset diambil dari durasi aktual tiap chunk (ffprobe), bukan asumsi
    chunk_seconds, supaya tidak terjadi drift pada chunk terakhir.
    """
    t0 = time.monotonic()
    all_segments: list[Segment] = []
    text_parts: list[str] = []
    offset = 0.0
    billed = 0.0
    lang: str | None = language

    for i, chunk in enumerate(chunks, start=1):
        dur = 0.0
        try:
            dur = probe(log, chunk).duration_s
        except Exception as e:  # noqa: BLE001 - probe gagal bukan alasan berhenti total
            log.warn(f"probe chunk {i} gagal: {e}")

        log.info(f"chunk {i}/{len(chunks)} ({dur:.0f}s, offset {offset:.0f}s)")
        tr = transcribe_file(log, api_key, chunk, model=model, language=language, prompt=prompt)
        if tr.language:
            lang = tr.language

        for s in tr.segments:
            all_segments.append(
                Segment(
                    start=round(s.start + offset, 3),
                    end=round(s.end + offset, 3),
                    text=s.text,
                    words=[Word(start=w.start + offset, end=w.end + offset, word=w.word) for w in s.words],
                )
            )
        if tr.text:
            text_parts.append(tr.text)

        billed += tr.duration_s or dur
        offset += dur if dur > 0 else tr.duration_s
        if on_chunk:
            on_chunk(i, len(chunks))

    return Transcript(
        text="\n".join(text_parts).strip(),
        language=lang,
        duration_s=offset,
        segments=all_segments,
        elapsed_s=round(time.monotonic() - t0, 2),
        audio_seconds_billed=billed,
    )


def _backoff(log: Logger, attempt: int, reason: str, base: float = 2.0) -> None:
    wait = base**attempt + random.uniform(0, 1.0)
    log.warn(f"{reason} — retry {attempt}/{MAX_RETRIES} dalam {wait:.1f}s")
    time.sleep(min(wait, 60.0))


def _mime_for(path: Path) -> str:
    return {
        ".opus": "audio/opus",
        ".ogg": "audio/ogg",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".flac": "audio/flac",
        ".webm": "audio/webm",
    }.get(path.suffix.lower(), "application/octet-stream")


def _safe_detail(r: requests.Response) -> str:
    """Ambil pesan error tanpa membocorkan header/token."""
    try:
        j = r.json()
        msg = j.get("error", {}).get("message") if isinstance(j, dict) else None
        return str(msg or j)[:300]
    except Exception:  # noqa: BLE001
        return (r.text or "")[:300]
