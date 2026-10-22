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
# Terjemahan LANGSUNG audio → Inggris (docs/01 §29). CATATAN: hanya didukung
# `whisper-large-v3` (BUKAN -turbo — tabel dukungan Groq: Translation=No).
GROQ_TRANSLATE_URL = "https://api.groq.com/openai/v1/audio/translations"
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
    # metrik keyakinan Whisper (verbose_json OpenAI-style; mungkin absen di provider)
    no_speech_prob: float | None = None
    avg_logprob: float | None = None


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
    task: str = "transcribe",
) -> Transcript:
    """Proses SATU file audio (di bawah batas ukuran) via Groq Whisper.

    task="transcribe" → /audio/transcriptions (teks bahasa sumber).
    task="translate"  → /audio/translations (LANGSUNG ke Inggris; parameter
    `language` diabaikan — endpoint translate hanya menerima 'en'/kosong).
    """
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
                    "temperature": temperature,
                }
                # CATATAN (§32 v2): JANGAN kirim timestamp_granularities[] —
                # verbose_json sudah mengembalikan segments, dan param ini
                # dicurigai memicu transkrip degenerat pada audio tertentu
                # (translate tanpa param ini sehat pada audio yang sama).
                if language and task != "translate":
                    data["language"] = language
                if prompt:
                    # prompt memandu gaya & istilah; maks 244 token
                    data["prompt"] = prompt[:800]

                r = requests.post(
                    GROQ_TRANSLATE_URL if task == "translate" else GROQ_URL,
                    headers={
                        "authorization": f"Bearer {api_key}",
                        "user-agent": "inggrisin-engine",
                    },
                    files=files,
                    data=data,
                    timeout=(30, 600),
                )
        except requests.RequestException as e:
            if attempt == MAX_RETRIES:
                raise TranscribeError(
                    "groq_failed", f"network error: {type(e).__name__}", True
                ) from e
            _backoff(log, attempt, f"network {type(e).__name__}")
            continue

        if r.status_code == 200:
            elapsed = time.monotonic() - t0
            return _parse(
                log, r, elapsed, label="terjemahan whisper" if task == "translate" else "transkrip"
            )

        # --- pemetaan error yang bisa ditindaklanjuti ---
        if r.status_code in (401, 403):
            raise TranscribeError(
                "groq_auth",
                f"Groq menolak API key (HTTP {r.status_code}). Perbarui key di dashboard.",
            )
        if r.status_code == 413:
            raise TranscribeError(
                "groq_file_too_large", "Groq menolak: file terlalu besar (HTTP 413)"
            )
        if r.status_code == 422:
            raise TranscribeError(
                "groq_failed", f"Groq tidak bisa memproses audio (HTTP 422): {_safe_detail(r)}"
            )
        if r.status_code == 429:
            # rate limit → retryable, tapi jangan buang banyak percobaan
            if attempt == MAX_RETRIES:
                raise TranscribeError(
                    "groq_rate_limit", "Groq rate limit tercapai setelah beberapa percobaan", True
                )
            _backoff(log, attempt, "429 rate limit", base=4.0)
            continue
        if 500 <= r.status_code < 600:
            if attempt == MAX_RETRIES:
                raise TranscribeError(
                    "groq_failed", f"Groq error server (HTTP {r.status_code})", True
                )
            _backoff(log, attempt, f"{r.status_code} server error")
            continue

        raise TranscribeError("groq_failed", f"Groq HTTP {r.status_code}: {_safe_detail(r)}")

    raise TranscribeError("groq_failed", "transkripsi gagal tanpa status jelas")


def _parse(
    log: Logger, r: requests.Response, elapsed: float, label: str = "transkrip"
) -> Transcript:
    j = r.json()
    segments: list[Segment] = []
    for s in j.get("segments") or []:
        words = [
            Word(
                start=float(w.get("start", 0)),
                end=float(w.get("end", 0)),
                word=str(w.get("word", "")),
            )
            for w in (s.get("words") or [])
        ]
        nsp = s.get("no_speech_prob")
        alp = s.get("avg_logprob")
        segments.append(
            Segment(
                start=float(s.get("start", 0.0)),
                end=float(s.get("end", 0.0)),
                text=str(s.get("text", "")).strip(),
                words=words,
                no_speech_prob=float(nsp) if isinstance(nsp, int | float) else None,
                avg_logprob=float(alp) if isinstance(alp, int | float) else None,
            )
        )

    text = str(j.get("text") or "").strip()
    duration = float(j.get("duration") or 0.0)
    lang = j.get("language")
    log.info(
        f"{label}: {len(text)} char, {len(segments)} segmen, {elapsed:.1f}s "
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
    task: str = "transcribe",
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
        tr = transcribe_file(
            log, api_key, chunk, model=model, language=language, prompt=prompt, task=task
        )
        if tr.language:
            lang = tr.language

        for s in tr.segments:
            all_segments.append(
                Segment(
                    start=round(s.start + offset, 3),
                    end=round(s.end + offset, 3),
                    text=s.text,
                    words=[
                        Word(start=w.start + offset, end=w.end + offset, word=w.word)
                        for w in s.words
                    ],
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


def align_translation(id_segments: list[Segment], en_segments: list[Segment]) -> list[str]:
    """Petakan hasil translate Whisper (timing-nya SENDIRI) ke segmen transkrip asli.

    docs/01 §29/§34: timing kanonik = transkrip bahasa sumber (manifest).
    Pada audio PANJANG pass translate menghasilkan segmen JAUH lebih kasar
    daripada transcribe (satu kalimat Inggris bisa membentang di atas beberapa
    slot asli). Algoritma lama menumpuk seluruh teks ke satu slot terbaik lalu
    mengisi slot kosong dengan teks slot terdekat → baris DUPLIKAT dan TTS
    mengetim dobel (insiden job_GbFcCQT41RZ5WeL4: 48 baris duplikat, 141 baris
    over-budget, audio bertabrakan).

    v2: segmen Inggris DISEBAR proporsional ke semua slot asli yang overlap
    dengannya — kata-kata dibagi menurut porsi waktu overlap, sehingga tiap
    slot mendapat fragmen miliknya sendiri (sinkron dengan jendela waktunya)
    dan tidak ada duplikasi. Perilaku 1:1 untuk video pendek tidak berubah
    (overlap tunggal → teks utuh ke slot itu).
    """
    out = [""] * len(id_segments)
    if not id_segments or not en_segments:
        return out

    def _center(a: float, b: float) -> float:
        return (a + b) / 2.0

    for e in en_segments:
        text = (e.text or "").strip()
        if not text:
            continue
        overlaps: list[tuple[int, float]] = []
        for i, s in enumerate(id_segments):
            ov = min(e.end, s.end) - max(e.start, s.start)
            if ov > 0:
                overlaps.append((i, ov))
        if not overlaps:
            # tidak ada overlap sama sekali → pusat waktu terdekat
            ce = _center(e.start, e.end)
            best = min(
                range(len(id_segments)),
                key=lambda i: abs(_center(id_segments[i].start, id_segments[i].end) - ce),
            )
            out[best] = f"{out[best]} {text}".strip() if out[best] else text
            continue
        if len(overlaps) == 1:
            i = overlaps[0][0]
            out[i] = f"{out[i]} {text}".strip() if out[i] else text
            continue
        words = text.split()
        total_ov = sum(ov for _, ov in overlaps)
        if len(words) < 2 or total_ov <= 0:
            i = max(overlaps, key=lambda t: t[1])[0]
            out[i] = f"{out[i]} {text}".strip() if out[i] else text
            continue
        # sebar kata proporsional terhadap porsi waktu overlap per slot
        cursor = 0
        for k, (i, ov) in enumerate(overlaps):
            if k == len(overlaps) - 1:
                take = len(words) - cursor
            else:
                take = round(len(words) * ov / total_ov)
                take = max(0, min(take, len(words) - cursor))
            frag = " ".join(words[cursor : cursor + take])
            cursor += take
            if not frag:
                continue
            out[i] = f"{out[i]} {frag}".strip() if out[i] else frag

    for i, s in enumerate(id_segments):
        if not out[i]:
            cs = _center(s.start, s.end)
            j = min(
                range(len(en_segments)),
                key=lambda k: abs(_center(en_segments[k].start, en_segments[k].end) - cs),
            )
            out[i] = (en_segments[j].text or "").strip()
    return out


def source_degenerate(
    id_tr: Transcript,
    en_tr: Transcript,
    ratio: float = 0.35,
    min_en_chars: int = 60,
) -> bool:
    """True bila pass sumber "mendengar" JAUH lebih sedikit daripada pass translate.

    Indikasi bahasa audio ≠ bahasa sumber yang dipaksa (docs/01 §32): Whisper
    yang dipaksa `language=id` atas ucapan bahasa lain menghasilkan transkrip
    degenerat (sedikit segmen/char, kadang halusinasi pendek), sementara
    /audio/translations tetap sehat karena memang mendengar bahasa aslinya.
    Tanpa guard ini, align_translation menumpuk SELURUH teks Inggris ke satu
    slot sumber → satu baris subtitle raksasa.
    """
    id_chars = sum(len((s.text or "").strip()) for s in id_tr.segments) or len(
        (id_tr.text or "").strip()
    )
    en_chars = sum(len((s.text or "").strip()) for s in en_tr.segments) or len(
        (en_tr.text or "").strip()
    )
    if en_chars < min_en_chars:
        return False
    return id_chars < ratio * en_chars


def speech_confidence(tr: Transcript) -> float | None:
    """Rata-rata no_speech_prob segmen (0=yakin ucapan, 1=yakin bukan ucapan).

    None bila provider tidak mengirim metriknya. Pelengkap source_degenerate:
    halusinasi Whisper atas BGM/keheningan biasanya ber-no_speech_prob tinggi,
    sementara ucapan riil rendah (kasus nyata §32: riil ≈ 0.24).
    """
    vals = [s.no_speech_prob for s in tr.segments if s.no_speech_prob is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)
