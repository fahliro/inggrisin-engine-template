"""
Unggah hasil ke tujuan akhir.

Dua mode, dipilih otomatis oleh app lewat env:

  A. `INGGRISIN_RESULT_UPLOAD_URL` diisi  → POST multipart ke proxy app.
     App yang meneruskan ke Drive dengan token Google-nya sendiri.
     KEUNTUNGAN: tidak ada kredensial Google di runner sama sekali.
     Ini mode default & paling aman.

  B. `GOOGLE_ACCESS_TOKEN` diisi (jarang, hanya bila user memilih "direct")
     → runner mengunggah langsung ke Drive.
     RISIKO: token muncul di lingkungan runner. Harus di-mask dan TTL pendek.
     Mode ini DINONAKTIFKAN secara default; lihat docs/00-ARCHITECTURE.md §3.

Untuk file besar, mode A memakai chunked/resumable di sisi app; engine cukup
mengirim satu multipart request dengan stream.
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from .log import Logger

MAX_RETRIES = 3


class UploadError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(slots=True)
class UploadResult:
    drive_file_id: str | None
    drive_web_url: str | None
    bytes: int
    elapsed_s: float
    mode: str


def upload_result(
    log: Logger,
    path: Path,
    *,
    proxy_url: str = "",
    filename: str | None = None,
    content_type: str = "audio/mpeg",
    job_id: str = "",
    extra_fields: dict[str, str] | None = None,
) -> UploadResult:
    if not path.exists() or path.stat().st_size == 0:
        raise UploadError("drive_upload_failed", f"File hasil tidak ada atau kosong: {path}")

    t0 = time.monotonic()
    size = path.stat().st_size
    name = filename or path.name

    if proxy_url:
        res = _upload_via_proxy(log, proxy_url, path, name, content_type, size, job_id, extra_fields or {})
    else:
        token = os.environ.get("GOOGLE_ACCESS_TOKEN", "").strip()
        if not token:
            raise UploadError(
                "drive_auth",
                "Tidak ada jalur unggah: INGGRISIN_RESULT_UPLOAD_URL kosong dan GOOGLE_ACCESS_TOKEN tidak disetel.",
            )
        res = _upload_direct_drive(log, token, path, name, content_type, size, extra_fields or {})

    res.elapsed_s = round(time.monotonic() - t0, 2)
    log.info(f"upload selesai: {size / 1e6:.2f} MB dalam {res.elapsed_s}s (mode={res.mode})")
    return res


def _upload_via_proxy(
    log: Logger,
    url: str,
    path: Path,
    filename: str,
    content_type: str,
    size: int,
    job_id: str,
    extra: dict[str, str],
) -> UploadResult:
    sha = hashlib.sha256(path.read_bytes()).hexdigest() if size < 200 * 1024 * 1024 else ""
    data = {"job_id": job_id, "filename": filename, "sha256": sha, **extra}

    last = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with path.open("rb") as fh:
                r = requests.post(
                    url,
                    data=data,
                    files={"file": (filename, fh, content_type)},
                    timeout=(30, 1800),
                    headers={"user-agent": "inggrisin-engine"},
                )
            if r.status_code == 200:
                j = r.json()
                return UploadResult(
                    drive_file_id=j.get("drive_file_id"),
                    drive_web_url=j.get("drive_web_url"),
                    bytes=int(j.get("bytes") or size),
                    elapsed_s=0.0,
                    mode="proxy",
                )
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if 400 <= r.status_code < 500 and r.status_code != 429:
                raise UploadError("drive_upload_failed", f"Proxy menolak unggahan ({last})")
        except requests.RequestException as e:
            last = f"network {type(e).__name__}"
        if attempt < MAX_RETRIES:
            wait = 3 * attempt
            log.warn(f"upload gagal ({last}) — retry {attempt}/{MAX_RETRIES} dalam {wait}s")
            time.sleep(wait)

    raise UploadError("drive_upload_failed", f"Upload gagal setelah {MAX_RETRIES} percobaan: {last}")


def _upload_direct_drive(
    log: Logger,
    token: str,
    path: Path,
    filename: str,
    content_type: str,
    size: int,
    extra: dict[str, str],
) -> UploadResult:
    """
    Mode langsung ke Drive (multipart upload). Hanya untuk file < 5 MB;
    di atas itu wajib resumable. Karena mode ini default-off, kita tolak
    file besar daripada mengimplementasikan resumable upload di runner.
    """
    if size > 5 * 1024 * 1024:
        raise UploadError(
            "drive_upload_failed",
            "Mode direct Drive hanya untuk file kecil (<5MB). Gunakan INGGRISIN_RESULT_UPLOAD_URL (proxy app).",
        )

    folder = extra.get("drive_folder_id") or ""
    metadata = {"name": filename, "mimeType": content_type}
    if folder:
        metadata["parents"] = [folder]

    import json

    boundary = f"inggrisin_{int(time.time() * 1000)}"
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            b'Content-Type: application/json; charset=UTF-8\r\n\r\n',
            json.dumps(metadata).encode(),
            b"\r\n",
            f"--{boundary}\r\n".encode(),
            f'Content-Type: {content_type}\r\n\r\n'.encode(),
            path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )

    r = requests.post(
        "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
        headers={
            "authorization": f"Bearer {token}",
            "content-type": f"multipart/related; boundary={boundary}",
            "user-agent": "inggrisin-engine",
        },
        data=body,
        timeout=(30, 600),
    )
    if r.status_code not in (200, 201):
        raise UploadError("drive_upload_failed", f"Drive multipart gagal (HTTP {r.status_code})")
    j = r.json()
    return UploadResult(
        drive_file_id=j.get("id"),
        drive_web_url=j.get("webViewLink"),
        bytes=size,
        elapsed_s=0.0,
        mode="direct",
    )
