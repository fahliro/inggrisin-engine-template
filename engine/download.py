"""
Unduh sumber dari proxy app (bukan langsung dari Google Drive).

Kenapa proxy?
  Access token Google tidak boleh masuk ke runner. App men-stream file dari
  Drive lewat `/api/dl/<token-sekali-pakai>`; runner hanya melihat URL itu.

Yang dilakukan modul ini:
  - stream ke disk (tidak pernah load penuh ke memory — video bisa GB-an)
  - verifikasi ukuran & sha256 bila app mengirimnya
  - retry dengan backoff untuk error jaringan transient
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from .log import Logger

CHUNK = 1024 * 1024  # 1 MiB
MAX_RETRIES = 3
# kalau sumber lebih besar dari ini, hampir pasti bukan video pendek yang wajar
HARD_LIMIT_BYTES = 6 * 1024 * 1024 * 1024


class DownloadError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(slots=True)
class DownloadResult:
    path: Path
    bytes: int
    sha256: str | None
    content_type: str | None
    elapsed_s: float


def download(
    log: Logger,
    url: str,
    dest: Path,
    expected_bytes: int = 0,
    expected_sha256: str = "",
) -> DownloadResult:
    dest.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    last_err: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return _attempt(log, url, dest, expected_bytes, expected_sha256, t0, attempt)
        except DownloadError as e:
            # 4xx dari proxy = token kadaluarsa/salah. Retry tidak akan menolong.
            if e.code in {"source_download_failed", "source_too_large"}:
                raise
            last_err = e
        except requests.RequestException as e:
            last_err = e
            log.warn(f"download gagal (attempt {attempt}/{MAX_RETRIES}): {type(e).__name__}")

        if attempt < MAX_RETRIES:
            wait = 2**attempt
            log.info(f"retry download dalam {wait}s")
            time.sleep(wait)

    raise DownloadError("source_download_failed", f"download gagal setelah {MAX_RETRIES} percobaan: {last_err}")


def _attempt(
    log: Logger,
    url: str,
    dest: Path,
    expected_bytes: int,
    expected_sha256: str,
    t0: float,
    attempt: int,
) -> DownloadResult:
    tmp = dest.with_suffix(dest.suffix + ".part")
    sha = hashlib.sha256()
    total = 0

    with requests.get(url, stream=True, timeout=(15, 600), allow_redirects=True) as r:
        if r.status_code == 404:
            raise DownloadError(
                "source_download_failed",
                "Link sumber tidak ditemukan atau sudah dipakai (sekali pakai). Jalankan ulang job dari dashboard.",
            )
        if r.status_code == 410:
            raise DownloadError("source_download_failed", "Link sumber kadaluarsa. Jalankan ulang job.")
        if r.status_code >= 400:
            raise DownloadError("source_download_failed", f"Proxy mengembalikan HTTP {r.status_code}")

        content_type = r.headers.get("content-type")
        declared = r.headers.get("content-length")
        if declared:
            declared_i = int(declared)
            if declared_i > HARD_LIMIT_BYTES:
                raise DownloadError("source_too_large", f"File terlalu besar: {declared_i} byte")

        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=CHUNK):
                if not chunk:
                    continue
                f.write(chunk)
                sha.update(chunk)
                total += len(chunk)
                if total > HARD_LIMIT_BYTES:
                    f.close()
                    tmp.unlink(missing_ok=True)
                    raise DownloadError("source_too_large", "File melebihi batas 6 GB")

    if total == 0:
        tmp.unlink(missing_ok=True)
        raise DownloadError("source_download_failed", "File sumber kosong (0 byte)")

    if expected_bytes and total != expected_bytes:
        # bukan fatal — proxy bisa saja men-chunk berbeda; tapi patut dicatat
        log.warn(f"ukuran tidak cocok: dapat {total}, diharapkan {expected_bytes}")

    digest = sha.hexdigest()
    if expected_sha256 and digest != expected_sha256.lower():
        tmp.unlink(missing_ok=True)
        raise DownloadError("source_download_failed", "Checksum sha256 tidak cocok — file korup saat transfer")

    tmp.replace(dest)
    elapsed = time.monotonic() - t0
    log.info(
        f"download selesai: {total / 1e6:.1f} MB dalam {elapsed:.1f}s "
        f"({total / max(elapsed, 0.001) / 1e6:.1f} MB/s)",
        attempt=attempt,
        content_type=content_type,
    )
    return DownloadResult(
        path=dest,
        bytes=total,
        sha256=digest,
        content_type=content_type,
        elapsed_s=round(elapsed, 2),
    )
