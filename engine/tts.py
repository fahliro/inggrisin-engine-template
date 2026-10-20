"""
Piper TTS — mesin suara inti. Jalan di runner GitHub Actions milik user,
tanpa biaya API, tanpa GPU.

Dua jalur eksekusi (fallback otomatis):

  1. PYTHON API  — `piper.voice.PiperVoice` dari paket `piper-tts`.
     Dipakai bila `pip install piper-tts` berhasil (ada wheel untuk
     Python 3.11 di ubuntu-latest).

  2. BINARY      — rilis `piper_linux_x86_64.tar.gz` yang self-contained
     (sudah termasuk onnxruntime + espeak-ng data). Dipakai bila wheel pip
     tidak tersedia / gagal import. Jalur ini paling tahan banting terhadap
     perubahan image `ubuntu-latest`.

Voice model TIDAK di-commit ke repo (puluhan MB). Di-download sekali dari
HuggingFace ke `PIPERS_VOICE_DIR` dan di-cache oleh `actions/cache` di
workflow — jadi run kedua dan seterusnya tidak mengunduh ulang.

Kenapa sintesis per segmen, bukan satu teks panjang?
  - UI belajar butuh "klik kalimat → dengar kalimat itu".
  - Piper lebih stabil pada teks pendek (jeda & intonasi lebih natural).
  - Kalau satu segmen gagal, segmen lain tetap jadi (partial success).
  Biayanya: penggabungan audio. Kita pakai ffmpeg concat, murah.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tarfile
import time
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path

from .log import Logger

# registry ringkas voice yang kita dukung. URL mengikuti pola resmi
# rhasspy/piper-voices di HuggingFace.
VOICE_HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

VOICES: dict[str, dict[str, str]] = {
    "en_US-lessac-medium": {
        "path": "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "sample_rate": "22050",
    },
    "en_US-ryan-high": {
        "path": "en/en_US/ryan/high/en_US-ryan-high.onnx",
        "sample_rate": "22050",
    },
    "en_GB-alba-medium": {
        "path": "en/en_GB/alba/medium/en_GB-alba-medium.onnx",
        "sample_rate": "22050",
    },
    "en_GB-northern_english_male-medium": {
        "path": "en/en_GB/northern_english_male/medium/en_GB-northern_english_male-medium.onnx",
        "sample_rate": "22050",
    },
    "en_AU-athena-medium": {
        "path": "en/en_AU/athena/medium/en_AU-athena-medium.onnx",
        "sample_rate": "22050",
    },
}

BINARY_RELEASE_URL = (
    "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz"
)


class TtsError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(slots=True)
class TtsResult:
    wav_dir: Path
    files: list[Path]
    merged_wav: Path | None
    elapsed_s: float
    backend: str  # 'python' | 'binary'
    failed_segments: int


def voice_dir() -> Path:
    d = Path(os.environ.get("PIPERS_VOICE_DIR", str(Path.home() / ".piper-voices")))
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_voice(log: Logger, voice_id: str) -> tuple[Path, Path]:
    """
    Pastikan model + config voice tersedia lokal. Mengembalikan (onnx, onnx.json).
    Download idempoten: kalau file sudah ada dengan ukuran wajar, dipakai ulang.
    """
    meta = VOICES.get(voice_id)
    if not meta:
        raise TtsError(
            "piper_voice_missing",
            f"Voice '{voice_id}' tidak dikenal. Pilihan: {', '.join(sorted(VOICES))}",
        )

    dest = voice_dir() / Path(meta["path"]).name
    cfg = dest.with_suffix(dest.suffix + ".json")

    for target, suffix in ((dest, ""), (cfg, ".json")):
        if target.exists() and target.stat().st_size > 1024:
            continue
        url = f"{VOICE_HF_BASE}/{meta['path']}{suffix}"
        log.info(f"unduh voice: {url}")
        _download_to(url, target)

    if not dest.exists() or dest.stat().st_size < 1024:
        raise TtsError("piper_voice_missing", f"Model voice gagal diunduh: {dest.name}")
    return dest, cfg


def _download_to(url: str, dest: Path, retries: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"user-agent": "inggrisin-engine"})
            with urllib.request.urlopen(req, timeout=120) as r, tmp.open("wb") as f:  # noqa: S310
                shutil.copyfileobj(r, f, 1024 * 256)
            tmp.replace(dest)
            return
        except Exception as e:  # noqa: BLE001 - banyak kemungkinan error jaringan
            last = e
            tmp.unlink(missing_ok=True)
            time.sleep(2**attempt)
    raise TtsError("piper_voice_missing", f"gagal mengunduh {url}: {last}")


# ---------------------------------------------------------------------------
# backend 1: Python API
# ---------------------------------------------------------------------------


def _try_python_backend(log: Logger) -> object | None:
    try:
        from piper import PiperVoice  # type: ignore

        return PiperVoice
    except Exception as e:  # noqa: BLE001
        log.warn(f"piper python API tidak tersedia ({type(e).__name__}); coba jalur binary")
        return None


def _synth_one(voice: object, text: str, dst: Path) -> None:
    """Sintesis satu teks → WAV `dst`, kompatibel antar-versi piper-tts.

    piper-tts 1.2.0 (versi yang di-pin requirements) hanya punya
    `synthesize(text, wav_file: wave.Wave_write)`; `synthesize_wav(text, fh)`
    baru ada di 1.3+. Tanpa wrapper ini, setiap baris AttributeError dan
    seluruh job gagal di TTS (padahal load voice sukses, jadi fallback
    binary tidak pernah jalan).
    """
    if hasattr(voice, "synthesize_wav"):
        with dst.open("wb") as fh:
            voice.synthesize_wav(text, fh)  # type: ignore[attr-defined]
    else:
        with wave.open(str(dst), "wb") as wav_file:
            voice.synthesize(text, wav_file)  # type: ignore[attr-defined]


def _synth_python(
    log: Logger,
    voice_cls: object,
    model: Path,
    texts: list[str],
    out_dir: Path,
) -> tuple[list[Path], int]:
    voice = voice_cls.load(str(model))  # type: ignore[attr-defined]
    out_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    failed = 0
    for i, text in enumerate(texts):
        dst = out_dir / f"seg_{i:05d}.wav"
        if not text.strip():
            continue
        try:
            _synth_one(voice, text, dst)
            if dst.stat().st_size > 1000:
                files.append(dst)
            else:
                dst.unlink(missing_ok=True)
                failed += 1
        except Exception as e:  # noqa: BLE001
            failed += 1
            log.warn(f"segmen {i} gagal (python): {type(e).__name__}")
    return files, failed


# ---------------------------------------------------------------------------
# backend 2: binary release
# ---------------------------------------------------------------------------


def _ensure_binary(log: Logger) -> Path | None:
    bin_dir = voice_dir() / "piper-bin"
    exe = bin_dir / "piper"
    if exe.exists():
        return exe

    tgz = bin_dir / "piper.tar.gz"
    try:
        _download_to(BINARY_RELEASE_URL, tgz)
    except TtsError as e:
        log.warn(f"binary piper gagal diunduh: {e.message}")
        return None

    bin_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(tgz, "r:gz") as tf:
            # proteksi path traversal (S202): hanya ekstrak anggota aman
            for member in tf.getmembers():
                target = (bin_dir / member.name).resolve()
                if not str(target).startswith(str(bin_dir.resolve())):
                    raise TtsError(
                        "piper_install_failed", f"path tidak aman di arsip: {member.name}"
                    )
            tf.extractall(bin_dir, filter="data")  # noqa: S202
    except tarfile.TarError as e:
        raise TtsError("piper_install_failed", f"ekstraksi binary piper gagal: {e}") from e
    finally:
        tgz.unlink(missing_ok=True)

    if not exe.exists():
        # beberapa rilis menaruh piper di subfolder
        found = list(bin_dir.rglob("piper"))
        exe = found[0] if found else exe
    if not exe.exists():
        raise TtsError("piper_install_failed", "binary `piper` tidak ditemukan setelah ekstraksi")
    exe.chmod(0o755)
    log.info(f"binary piper siap: {exe}")
    return exe


def _synth_binary(log: Logger, exe: Path, model: Path, config: Path, text: str, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            str(exe),
            "--model",
            str(model),
            "--config",
            str(config),
            "--output_file",
            str(dst),
        ],
        input=text.encode("utf-8"),
        capture_output=True,
        check=False,
        timeout=180,
    )
    if proc.returncode != 0:
        log.warn(f"piper binary gagal: {(proc.stderr or b'').decode('utf-8', 'replace')[-200:]}")
        return False
    return dst.exists() and dst.stat().st_size > 1000


# ---------------------------------------------------------------------------
# API publik
# ---------------------------------------------------------------------------


def synthesize_segments(
    log: Logger,
    texts: list[str],
    voice_id: str,
    out_dir: Path,
    gap_ms: int = 250,
) -> TtsResult:
    """
    Sintesis tiap segmen jadi WAV terpisah, lalu gabungkan jadi satu file
    dengan jeda antar segmen (penting untuk shadowing: ada ruang bernapas).
    """
    t0 = time.monotonic()
    model, config = ensure_voice(log, voice_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = "python"
    voice_cls = _try_python_backend(log)
    files: list[Path] = []
    failed = 0

    if voice_cls is not None:
        files, failed = _synth_python(log, voice_cls, model, texts, out_dir)
        if not files:
            log.warn("python backend tidak menghasilkan file — pindah ke binary")
            voice_cls = None

    if voice_cls is None:
        exe = _ensure_binary(log)
        if exe is None:
            raise TtsError(
                "piper_install_failed",
                "Piper tidak bisa dipakai: wheel Python gagal dan binary release tidak tersedia.",
            )
        backend = "binary"
        files, failed = [], 0
        for i, text in enumerate(texts):
            if not text.strip():
                continue
            dst = out_dir / f"seg_{i:05d}.wav"
            if _synth_binary(log, exe, model, config, text, dst):
                files.append(dst)
            else:
                failed += 1

    if not files:
        raise TtsError("piper_failed", "Semua segmen gagal disintesis")

    merged = concat_with_gaps(log, files, out_dir / "merged.wav", gap_ms=gap_ms)
    elapsed = time.monotonic() - t0
    log.info(f"TTS selesai: {len(files)} segmen, {failed} gagal, backend={backend}, {elapsed:.1f}s")
    return TtsResult(
        wav_dir=out_dir,
        files=files,
        merged_wav=merged,
        elapsed_s=round(elapsed, 2),
        backend=backend,
        failed_segments=failed,
    )


def concat_with_gaps(log: Logger, wavs: list[Path], dest: Path, gap_ms: int = 250) -> Path | None:
    """
    Gabungkan WAV dengan jeda hening di antaranya.
    Memakai ffmpeg concat demuxer + filter untuk sisipkan silence.
    """
    if not wavs:
        return None
    if len(wavs) == 1:
        shutil.copyfile(wavs[0], dest)
        return dest

    list_file = dest.with_suffix(".list.txt")
    silence = dest.with_suffix(".silence.wav")

    try:
        # buat satu file silence yang dipakai berulang
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=22050:cl=mono",
                "-t",
                f"{gap_ms / 1000:.3f}",
                "-c:a",
                "pcm_s16le",
                str(silence),
            ],
            capture_output=True,
            check=True,
        )
        lines: list[str] = []
        for i, w in enumerate(wavs):
            lines.append(f"file '{w.resolve()}'")
            if i < len(wavs) - 1:
                lines.append(f"file '{silence.resolve()}'")
        list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        proc = subprocess.run(
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
                "-c:a",
                "pcm_s16le",
                "-ar",
                "22050",
                "-ac",
                "1",
                str(dest),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0 or not dest.exists():
            log.warn("concat dengan jeda gagal — gabung tanpa jeda")
            list_file.write_text(
                "\n".join(f"file '{w.resolve()}'" for w in wavs) + "\n", encoding="utf-8"
            )
            proc2 = subprocess.run(
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
                    str(dest),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc2.returncode != 0 or not dest.exists():
                raise TtsError("piper_failed", "Penggabungan audio gagal")
        return dest
    except subprocess.CalledProcessError as e:
        raise TtsError("piper_failed", f"ffmpeg gagal saat concat: {e}") from e
    finally:
        list_file.unlink(missing_ok=True)
        silence.unlink(missing_ok=True)


def selfcheck(log: Logger, voice_id: str = "en_US-lessac-medium") -> dict[str, object]:
    """
    Dipanggil workflow `selfcheck.yml`. Memverifikasi seluruh rantai TTS tanpa
    menghabiskan kuota: pastikan ffmpeg ada, piper bisa di-import/diunduh,
    dan satu kalimat pendek benar-benar jadi WAV.
    """
    info: dict[str, object] = {"voice": voice_id}

    for tool in ("ffmpeg", "ffprobe"):
        info[tool] = bool(shutil.which(tool))
        if not info[tool]:
            info["ok"] = False
            info["reason"] = f"{tool} tidak ditemukan"
            return info

    proc = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=False)
    info["ffmpeg_version"] = (proc.stdout or "").split("\n")[0][:60]

    try:
        model, config = ensure_voice(log, voice_id)
        info["model_bytes"] = model.stat().st_size
        info["model_sha256"] = hashlib.sha256(model.read_bytes()).hexdigest()[:16]
    except TtsError as e:
        info["ok"] = False
        info["reason"] = f"voice: {e.message}"
        return info

    voice_cls = _try_python_backend(log)
    if voice_cls is not None:
        info["backend"] = "python"
        try:
            import piper  # type: ignore

            info["piper_version"] = getattr(piper, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            info["piper_version"] = "unknown"
    else:
        exe = _ensure_binary(log)
        info["backend"] = "binary" if exe else "none"
        if not exe:
            info["ok"] = False
            info["reason"] = "tidak ada backend Piper yang bisa dipakai"
            return info

    # runner Actions = container ephemeral; RUNNER_TEMP selalu ada di lingkungan nyata
    out = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "selfcheck.wav"  # noqa: S108
    try:
        res = synthesize_segments(log, ["This is an inggrisin self check."], voice_id, out.parent)
        info["synth_files"] = len(res.files)
        info["ok"] = len(res.files) == 1
    except TtsError as e:
        info["ok"] = False
        info["reason"] = f"synth: {e.message}"
        return info
    finally:
        out.unlink(missing_ok=True)

    info.setdefault("ok", True)
    return info


# ---------------------------------------------------------------------------
# v2 DUBBING: sintesis per baris, SEJAJAR INDEKS (untuk penempatan timeline)
# ---------------------------------------------------------------------------


def synthesize_lines(
    log: Logger,
    voice_id: str,
    lines: list[str],
    out_dir: Path,
    *,
    max_failure_ratio: float = 0.10,
) -> list[Path | None]:
    """
    Satu WAV per baris (line_%05d.wav), daftar hasil sejajar indeks `lines`
    (None = baris kosong atau gagal). Berbeda dengan synthesize_segments
    (pipeline lama) yang menggabungkan berurutan — di dubbing, penempatan
    waktu dilakukan audio.build_voice_track, bukan di sini.

    Gagal ≤ max_failure_ratio dari baris berisi → lanjut (celah jadi hening,
    jumlah dilaporkan di callback). Lebih dari itu → TtsError (job gagal;
    dubbing dengan banyak kalimat bisu bukan produk).
    """
    model, config = ensure_voice(log, voice_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[Path | None] = [None] * len(lines)

    voice_obj = None
    voice_cls = _try_python_backend(log)
    if voice_cls is not None:
        try:
            voice_obj = voice_cls.load(str(model))  # type: ignore[attr-defined]
        except Exception as e:  # noqa: BLE001
            log.warn(f"load voice python gagal ({type(e).__name__}); coba binary")
            voice_obj = None

    exe = None
    if voice_obj is None:
        exe = _ensure_binary(log)
        if exe is None:
            raise TtsError(
                "piper_install_failed",
                "Piper tidak bisa dipakai: wheel Python gagal dan binary tidak tersedia.",
            )

    for i, text in enumerate(lines):
        if not (text or "").strip():
            continue
        dst = out_dir / f"line_{i:05d}.wav"
        ok = False
        if voice_obj is not None:
            try:
                _synth_one(voice_obj, text, dst)
                ok = dst.exists() and dst.stat().st_size > 1000
                if not ok:
                    dst.unlink(missing_ok=True)
            except Exception as e:  # noqa: BLE001
                log.warn(f"baris {i} gagal (python): {type(e).__name__}")
        if not ok and exe is not None:
            ok = _synth_binary(log, exe, model, config, text, dst)
        if ok:
            results[i] = dst

    spoken = sum(1 for t in lines if (t or "").strip())
    missing = sum(1 for t, r in zip(lines, results, strict=True) if (t or "").strip() and r is None)
    if spoken and missing == spoken:
        raise TtsError("piper_failed", "Semua baris gagal disintesis")
    if spoken and missing / spoken > max_failure_ratio:
        raise TtsError(
            "piper_failed",
            f"{missing}/{spoken} baris gagal disintesis (> {max_failure_ratio:.0%}) — hasil akan penuh keheningan",
        )
    if missing:
        log.warn(
            f"{missing} baris gagal disintesis — celahnya dibiarkan hening (dilaporkan di callback)"
        )
    return results
