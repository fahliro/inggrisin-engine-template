"""
Persiapan audio dengan ffmpeg.

Masalah inti: Groq membatasi ukuran file audio — 25 MB (free tier) sampai
100 MB (dev tier). Video mentah hampir selalu lebih besar. Jadi audio harus
di-extract dan DIKOMPRES sampai di bawah batas, dengan strategi:

  1. probe durasi
  2. hitung bitrate yang membuat total ≈ TARGET_BYTES untuk durasi tsb
  3. kalau bitrate hasil hitungan < 8 kbps (durasi sangat panjang), turunkan
     sample rate ke 8 kHz mono — Whisper tetap akurat untuk speech
  4. kalau MASIH terlalu besar, potong per segmen (silence-aware) dan
     transkrip per segmen, lalu gabungkan dengan offset waktu

Output selalu 16 kHz mono untuk speech recognition; itu sudah lebih dari cukup
untuk Whisper dan memperkecil file 4-6x dibanding 44.1 kHz stereo.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .log import Logger

# 24 MB: menyisakan margin di bawah batas 25 MB free tier Groq
TARGET_AUDIO_BYTES = 24 * 1024 * 1024
MAX_BITRATE_KBPS = 128
# Lantai kualitas. Di bawah ini Opus mulai merusak konsonan halus (th/s/f/sh),
# dan justru di situlah nilai produk ini berada (latihan pelafalan).
# Kalau bitrate hasil hitungan < lantai, kita MEMILIH CHUNKING daripada
# menurunkan kualitas — chunking gratis, kualitas yang hilang tidak kembali.
QUALITY_FLOOR_KBPS = 24
MIN_BITRATE_KBPS = QUALITY_FLOOR_KBPS
CHUNK_SECONDS = 600  # 10 menit per chunk


class AudioError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(slots=True)
class Probe:
    duration_s: float
    has_audio: bool
    format_name: str
    video_codec: str | None
    size_bytes: int
    audio_codec: str | None = None


def _require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise AudioError("ffmpeg_failed", f"`{tool}` not found on the runner")


def _run(log: Logger, cmd: list[str], step: str) -> subprocess.CompletedProcess[str]:
    log.info(" ".join(cmd[:3]) + f" ... ({step})")
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        # stderr ffmpeg panjang; ambil ekor saja dan jangan sampai memuat path rahasia
        tail = "\n".join(proc.stderr.strip().splitlines()[-12:])
        raise AudioError(
            "ffmpeg_failed", f"ffmpeg failed during {step} (exit {proc.returncode}):\n{tail}"
        )
    return proc


def probe(log: Logger, path: Path) -> Probe:
    _require_ffmpeg()
    proc = _run(
        log,
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        "probe",
    )
    data = json.loads(proc.stdout or "{}")
    fmt = data.get("format", {}) or {}
    streams = data.get("streams", []) or []

    duration = float(fmt.get("duration") or 0.0)
    if duration <= 0:
        # beberapa container tidak mengisi format.duration; ambil dari stream
        for s in streams:
            d = float(s.get("duration") or 0.0)
            if d > 0:
                duration = d
                break
    if duration <= 0:
        raise AudioError("ffmpeg_failed", "Could not determine the media duration")

    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    if not audio_streams:
        raise AudioError("ffmpeg_failed", "The file has no audio track")

    return Probe(
        duration_s=duration,
        has_audio=True,
        format_name=str(fmt.get("format_name", "unknown")),
        video_codec=(video_streams[0].get("codec_name") if video_streams else None),
        size_bytes=int(fmt.get("size") or path.stat().st_size),
        audio_codec=str(audio_streams[0].get("codec_name") or "") or None,
    )


def _choose_settings(duration_s: float) -> tuple[int, int]:
    """
    Kembalikan (bitrate_kbps, sample_rate_hz).

    Bitrate dihitung supaya hasil ≈ TARGET_AUDIO_BYTES, tapi TIDAK pernah di
    bawah QUALITY_FLOOR_KBPS. Konsekuensinya: audio panjang menghasilkan file
    di atas target, dan `needs_chunking()` akan memicu pemotongan. Ini pilihan
    yang disengaja — lihat komentar QUALITY_FLOOR_KBPS.

    Sample rate selalu 16 kHz: Whisper bekerja pada 16 kHz internal, jadi
    sample rate lebih tinggi hanya memboroskan byte tanpa menambah akurasi.
    """
    # bytes = kbps * 1000 / 8 * duration  →  kbps = bytes * 8 / (1000 * duration)
    kbps = int(TARGET_AUDIO_BYTES * 8 / (1000 * max(duration_s, 1.0)))
    kbps = max(QUALITY_FLOOR_KBPS, min(MAX_BITRATE_KBPS, kbps))
    return kbps, 16000


def will_need_chunking(duration_s: float) -> bool:
    """Prediksi apakah durasi ini akan memicu chunking (untuk log/estimasi)."""
    kbps, _ = _choose_settings(duration_s)
    return (kbps * 1000 / 8 * duration_s) > TARGET_AUDIO_BYTES


def extract(
    log: Logger, src: Path, dst: Path, duration_hint: float | None = None
) -> tuple[Path, float]:
    """
    Extract audio ke satu file di bawah TARGET_AUDIO_BYTES.
    Mengembalikan (path, durasi). Path bisa = dst, atau tetap dst (selalu dst).
    """
    _require_ffmpeg()
    p = probe(log, src)
    duration = duration_hint if duration_hint is not None else p.duration_s

    # docs/01 §38: prioritas pertama = bitstream IDENTIK sumber (remux tanpa
    # re-encode): decoding Whisper jadi sama dengan Groq Playground yang
    # menerima file asli, sehingga potongan segmen lebih tepat (eksperimen
    # A/B job_8eYM1iEFdbdtslhG: re-encode opus menggeser batas 20-140 ms dan
    # menggabung tiga cut yang dibuat Playground). Fallback: opus adaptif bila
    # codec bukan aac atau hasil > TARGET_AUDIO_BYTES (video panjang).
    if p.audio_codec == "aac":
        m4a = dst.with_suffix(".m4a")
        try:
            _run(
                log,
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(src),
                    "-vn",
                    "-sn",
                    "-dn",
                    "-c:a",
                    "copy",
                    "-map_metadata",
                    "-1",
                    str(m4a),
                ],
                "remux-m4a",
            )
            size = m4a.stat().st_size if m4a.exists() else 0
            if size > 0:
                bps = size / max(duration, 1.0)
                sec = chunk_seconds_for(size, duration)
                if size <= TARGET_AUDIO_BYTES or bps * sec <= TARGET_AUDIO_BYTES:
                    mode = (
                        "satu request" if size <= TARGET_AUDIO_BYTES else f"di-chunk copy @ {sec}s"
                    )
                    log.info(
                        f"extract audio: remux aac→m4a tanpa re-encode ({size / 1e6:.1f} MB) — "
                        f"bitstream identik sumber (§38), {mode}"
                    )
                    return m4a, duration
                log.info(
                    f"remux m4a {size / 1e6:.1f} MB @ {bps / 1000:.0f} kbps — chunk copy tak "
                    "muat batas request Groq; fallback opus adaptif"
                )
            else:
                log.info("remux m4a kosong/gagal — fallback opus adaptif")
        except AudioError as e:
            log.warn(f"remux m4a gagal ({e.message[:120]}) — fallback opus adaptif")

    kbps, sr = _choose_settings(duration)
    log.info(f"extract audio: {duration:.0f}s → {kbps}kbps @ {sr}Hz mono opus")

    dst.parent.mkdir(parents=True, exist_ok=True)
    _run(
        log,
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-vn",
            "-sn",
            "-dn",  # buang video/subtitle/data
            "-ac",
            "1",
            "-ar",
            str(sr),
            "-c:a",
            "libopus",
            "-b:a",
            f"{kbps}k",
            "-map_metadata",
            "-1",
            str(dst),
        ],
        "extract",
    )

    size = dst.stat().st_size
    if size > TARGET_AUDIO_BYTES:
        # SENGAJA tidak diturunkan bitrate-nya di sini. Menurunkan kualitas demi
        # muat dalam satu request akan merusak konsonan halus yang justru mau
        # dilatih user. Sebagai gantinya main.py memanggil split_chunks() dan
        # mengirim beberapa request ke Groq (biaya total sama, kualitas utuh).
        est_chunks = max(2, int(size / TARGET_AUDIO_BYTES) + 1)
        log.info(
            f"audio {size / 1e6:.1f} MB > target {TARGET_AUDIO_BYTES / 1e6:.0f} MB "
            f"→ akan dipecah jadi ±{est_chunks} chunk"
        )

    return dst, duration


def chunk_seconds_for(size_bytes: int, duration_s: float) -> int:
    """Durasi chunk terpanjang sehingga tiap chunk copy ≤ TARGET_AUDIO_BYTES (§38).

    Chunk dipotong stream-copy (lossless), jadi ukuran chunk = bitrate × durasi.
    Di-clamp [60, CHUNK_SECONDS].
    """
    if size_bytes <= 0 or duration_s <= 0:
        return CHUNK_SECONDS
    bps = size_bytes / duration_s
    if bps <= 0:
        return CHUNK_SECONDS
    sec = int((TARGET_AUDIO_BYTES * 0.9) / bps)
    return max(60, min(CHUNK_SECONDS, sec))


def needs_chunking(path: Path) -> bool:
    return path.stat().st_size > TARGET_AUDIO_BYTES


def split_chunks(
    log: Logger, src: Path, out_dir: Path, chunk_seconds: int | None = None
) -> list[Path]:
    """
    Potong audio jadi beberapa file. `segment` dengan `-c copy` tidak
    encode ulang → cepat dan lossless. Kontainer chunk mengikuti sumber
    (m4a remux §38 tetap aac; fallback opus tetap opus) dan durasi chunk
    otomatis via chunk_seconds_for supaya tiap chunk ≤ TARGET_AUDIO_BYTES.
    """
    _require_ffmpeg()
    if chunk_seconds is None:
        chunk_seconds = chunk_seconds_for(src.stat().st_size, probe(log, src).duration_s)
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / f"chunk_%04d{src.suffix}")
    _run(
        log,
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-f",
            "segment",
            "-segment_time",
            str(chunk_seconds),
            "-c",
            "copy",
            "-reset_timestamps",
            "1",
            pattern,
        ],
        "split",
    )
    chunks = sorted(out_dir.glob("chunk_*.opus"))
    if not chunks:
        raise AudioError("ffmpeg_failed", "segmentation produced no chunks")
    log.info(f"audio dipecah jadi {len(chunks)} chunk @ {chunk_seconds}s")
    return chunks


def loudnorm(log: Logger, src: Path, dst: Path, target_i: float = -16.0) -> Path:
    """
    Normalisasi loudness ke -16 LUFS (standar podcast). Level yang konsisten
    antar segmen membuat hasil dubbing nyaman didengar. Two-pass supaya akurat.
    """
    _require_ffmpeg()
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Pass 1: ffmpeg menulis statistik loudnorm ke STDERR (bukan stdout/file),
    # jadi `_parse_loudnorm` menjalankan ffmpeg sekali dengan `-f null`.
    measured = _parse_loudnorm(log, src, target_i)

    _run(
        log,
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-af",
            measured,
            "-ar",
            "44100",
            "-ac",
            "1",
            "-c:a",
            "libmp3lame",
            "-q:a",
            "4",
            "-map_metadata",
            "-1",
            str(dst),
        ],
        "loudnorm-apply",
    )
    return dst


def _parse_loudnorm(log: Logger, src: Path, target_i: float) -> str:
    """Pass 1: ambil measured_I/TP/LRA/thresh untuk dipakai di pass 2."""
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(src),
            "-af",
            f"loudnorm=I={target_i}:TP=-1.5:LRA=11:print_format=json",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    err = proc.stderr or ""
    start = err.rfind("{")
    end = err.rfind("}")
    if start == -1 or end == -1 or end <= start:
        log.warn("loudnorm stats tidak terbaca — pakai single-pass")
        return f"loudnorm=I={target_i}:TP=-1.5:LRA=11"
    try:
        j = json.loads(err[start : end + 1])
    except json.JSONDecodeError:
        log.warn("loudnorm stats bukan JSON valid — pakai single-pass")
        return f"loudnorm=I={target_i}:TP=-1.5:LRA=11"

    return (
        f"loudnorm=I={target_i}:TP=-1.5:LRA=11:"
        f"measured_I={j.get('input_i', -23.0)}:"
        f"measured_TP={j.get('input_tp', -2.0)}:"
        f"measured_LRA={j.get('input_lra', 7.0)}:"
        f"measured_thresh={j.get('input_thresh', -33.0)}:"
        f"offset={j.get('target_offset', 0.0)}:linear=true:print_format=summary"
    )


def to_mp3(log: Logger, src: Path, dst: Path) -> Path:
    """Konversi final ke MP3 44.1 kHz mono (paling kompatibel untuk pemutar HP)."""
    _require_ffmpeg()
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run(
        log,
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-ar",
            "44100",
            "-ac",
            "1",
            "-c:a",
            "libmp3lame",
            "-q:a",
            "4",
            "-map_metadata",
            "-1",
            str(dst),
        ],
        "to-mp3",
    )
    return dst


def duration_of(log: Logger, path: Path) -> float:
    p = probe(log, path)
    return p.duration_s


def estimate_chunks(duration_s: float, chunk_seconds: int = CHUNK_SECONDS) -> int:
    return max(1, math.ceil(duration_s / chunk_seconds))


# ---------------------------------------------------------------------------
# v2 DUBBING: voice track yang SELARAS TIMELINE
# ---------------------------------------------------------------------------
# Dubbing berbeda dari pipeline lama: audio tidak digabung berurutan dengan
# jeda tetap, melainkan ditempatkan pada timestamp aslinya (start tiap
# kalimat dari manifest). Celah diisi silence; hasil = satu WAV sepanjang
# video, siap di-mux oleh vertical.py.

VOICE_SR = 22050  # sample rate native Piper; di-resample ke 44100 saat render


def plan_voice_track(
    placements: list[tuple[float, float]], total_s: float, min_gap: float = 0.02
) -> list[tuple[str, float]]:
    """
    Fungsi murni (bisa diuji tanpa ffmpeg): rencana susunan track suara.
    placements = [(start_s, dur_s)] terurut naik. Mengembalikan daftar
    ('gap'|'seg'|'trim', detik). 'trim' menandakan segmen sebelumnya molor
    melewati start segmen ini (overlap di-prevent dengan memotong celah).
    """
    plan: list[tuple[str, float]] = []
    cursor = 0.0
    for start, dur in placements:
        gap = start - cursor
        if gap > min_gap:
            plan.append(("gap", round(gap, 3)))
        elif gap < -min_gap:
            plan.append(("trim", round(-gap, 3)))
        if dur > 0:
            plan.append(("seg", round(dur, 3)))
        cursor = max(cursor, start) + dur
    tail = total_s - cursor
    if tail > min_gap:
        plan.append(("gap", round(tail, 3)))
    return plan


def apply_tempo(log: Logger, src: Path, dst: Path, tempo: float) -> Path:
    """Percepat/perlambat WAV ke tempo (0.5–2.0; kontrak kita maks 1.25).
    Sekaligus menormalkan format: 22050 Hz mono pcm_s16le."""
    tempo = min(2.0, max(0.5, float(tempo)))
    if abs(tempo - 1.0) < 0.01:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-ar",
            str(VOICE_SR),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(dst),
        ]
    else:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-filter:a",
            f"atempo={tempo:.4f}",
            "-ar",
            str(VOICE_SR),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(dst),
        ]
    _run(log, cmd, f"tempo {tempo:.2f}x")
    return dst


def _make_silence(log: Logger, seconds: float, dst: Path) -> Path:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"anullsrc=r={VOICE_SR}:cl=mono",
        "-t",
        f"{max(0.01, seconds):.3f}",
        "-c:a",
        "pcm_s16le",
        str(dst),
    ]
    _run(log, cmd, "silence")
    return dst


DEMUCS_MODEL = "htdemucs"


def demucs_available() -> bool:
    """True bila paket demucs ter-install di interpreter yang sama."""
    import importlib.util

    try:
        return importlib.util.find_spec("demucs") is not None
    except (ImportError, ValueError):
        return False


def separate_accompaniment(log: Logger, src_video: Path, work: Path, out: Path) -> Path | None:
    """Pisahkan vokal asli dari BGM/SFX memakai demucs (htdemucs, two-stems).

    Mengembalikan path stem `no_vocals` (musik tanpa suara narator) atau None
    bila gagal — PEMANGGIL WAJIB fallback ke dub tanpa musik (docs/01 §27:
    mempertahankan BGM itu peningkatan, bukan syarat hasil yang valid).

    Alur: ekstrak audio kualitas penuh (44.1 kHz stereo — BGM butuh high-end
    yang sudah dibuang di audio 16 kHz mono untuk Whisper) → demucs → stem.
    """
    if not demucs_available():
        log.warn("demucs tidak tersedia — BGM/SFX asli tidak dipertahankan")
        return None
    work.mkdir(parents=True, exist_ok=True)
    src_audio = work / "src_audio.wav"
    try:
        _run(
            log,
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(src_video),
                "-vn",
                "-ac",
                "2",
                "-ar",
                "44100",
                "-c:a",
                "pcm_s16le",
                str(src_audio),
            ],
            "extract audio penuh (utk demucs)",
        )
        _run(
            log,
            [
                sys.executable,
                "-m",
                "demucs",
                "--two-stems=vocals",
                "-n",
                DEMUCS_MODEL,
                "-d",
                "cpu",
                "-o",
                str(work),
                str(src_audio),
            ],
            "pemisahan vokal (demucs htdemucs)",
        )
    except AudioError as e:
        log.warn(f"pemisahan vokal gagal — lanjut tanpa BGM: {e.message.splitlines()[0]}")
        return None
    no_vocals = work / DEMUCS_MODEL / src_audio.stem / "no_vocals.wav"
    if not no_vocals.exists() or no_vocals.stat().st_size < 1024:
        log.warn(f"stem demucs tidak ditemukan: {no_vocals.name} — lanjut tanpa BGM")
        return None
    shutil.copyfile(no_vocals, out)
    log.info(f"BGM/SFX asli dipertahankan ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def mix_voice_with_music(
    log: Logger,
    voice: Path,
    music: Path,
    total_s: float,
    out: Path,
    music_gain: float = 0.6,
) -> Path:
    """Campur track dub (voice) + iringan asli (music) → out, lalu loudnorm.

    - voice: 22050 Hz mono (build_voice_track) → di-resample ke 44.1 kHz stereo
    - music di-attenuasi (`music_gain`) agar dub tetap dominan, dipotong ke
      total_s (amix duration=first memakai panjang voice sebagai acuan)
    - loudnorm two-pass menjaga hasil konsisten -16 LUFS (standar podcast)
    """
    raw = out.with_suffix(".raw.wav")
    out.parent.mkdir(parents=True, exist_ok=True)
    fc = (
        "[0:a]aresample=44100,aformat=channel_layouts=stereo[v];"
        f"[1:a]volume={music_gain:.3f},atrim=0:{max(0.1, total_s):.3f},asetpts=PTS-STARTPTS[m];"
        "[v][m]amix=inputs=2:duration=first:normalize=0[mix]"
    )
    _run(
        log,
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(voice),
            "-i",
            str(music),
            "-filter_complex",
            fc,
            "-map",
            "[mix]",
            "-ar",
            "44100",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(raw),
        ],
        "mix voice+music",
    )
    loudnorm(log, raw, out)
    raw.unlink(missing_ok=True)
    return out


def build_voice_track(
    log: Logger,
    items: list[tuple[float, Path]],
    total_s: float,
    out: Path,
) -> Path:
    """
    Susun satu WAV sepanjang `total_s` dari [(start_s, wav_path)].
    Semua wav masukan HARUS sudah 22050 Hz mono pcm_s16le (lihat apply_tempo).
    Implementasi: silence per celah + concat demuxer (lossless, -c copy).
    """
    if not items:
        _make_silence(log, total_s, out)
        return out
    items = sorted(items, key=lambda x: x[0])
    work = out.parent / "voice_parts"
    work.mkdir(parents=True, exist_ok=True)
    list_file = work / "concat.txt"
    lines: list[str] = []
    cursor = 0.0
    gap_n = 0
    for start, wav in items:
        gap = start - cursor
        if gap > 0.02:
            gp = work / f"gap_{gap_n:04d}.wav"
            _make_silence(log, gap, gp)
            lines.append(f"file '{gp}'")
            gap_n += 1
        elif gap < -0.02:
            log.warn(
                f"overlap {-gap:.2f}s sebelum {wav.name} — suara sebelumnya molor, celah dipotong"
            )
        lines.append(f"file '{wav.resolve()}'")
        cursor = max(cursor, start) + duration_of(log, wav)
    tail = total_s - cursor
    if tail > 0.02:
        gp = work / f"gap_{gap_n:04d}.wav"
        _make_silence(log, tail, gp)
        lines.append(f"file '{gp}'")
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _run(
        log,
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c",
            "copy",
            str(out),
        ],
        "voice track concat",
    )
    return out
