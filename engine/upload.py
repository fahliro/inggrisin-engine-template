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

Untuk file BESAR, mode A memakai PROTOKOL 2 (chunked resumable, docs/01 §33):
engine POST init (JSON + signature) → menerima `session` ber-HMAC dari app →
meng-PUT chunk ≤ 8 MiB per request (Content-Range) → app meneruskan (stream)
ke sesi resumable Drive. Satu request tidak pernah mendekati limit body edge
Cloudflare (100 MB), jadi video panjang tidak lagi mati 413. Resume setelah
kegagalan network: PUT `content-range: bytes */total` untuk menanyakan byte
yang sudah diterima Drive.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from .log import Logger

MAX_RETRIES = 3
# 8 MiB per request: jauh di bawah limit body edge Cloudflare (100 MB) dan
# kelipatan 256 KiB (syarat chunk Drive resumable).
CHUNK_BYTES = 8 * 1024 * 1024


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
    auth_secret: str = "",
) -> UploadResult:
    if not path.exists() or path.stat().st_size == 0:
        raise UploadError("drive_upload_failed", f"Result file missing or empty: {path}")

    t0 = time.monotonic()
    size = path.stat().st_size
    name = filename or path.name

    if proxy_url:
        res = _upload_via_proxy(
            log, proxy_url, path, name, content_type, size, job_id, extra_fields or {}, auth_secret
        )
    else:
        token = os.environ.get("GOOGLE_ACCESS_TOKEN", "").strip()
        if not token:
            raise UploadError(
                "drive_auth",
                "No upload path: INGGRISIN_RESULT_UPLOAD_URL is empty and GOOGLE_ACCESS_TOKEN is not set.",
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
    auth_secret: str = "",
) -> UploadResult:
    sha = hashlib.sha256(path.read_bytes()).hexdigest() if size < 200 * 1024 * 1024 else ""

    # Signature HMAC atas `${job_id}.${sha256_file}` — skema sama dengan
    # callback (lihat callback.py). App memverifikasi HMAC; session chunk
    # berikutnya ber-HMAC pula dengan secret yang sama.
    headers = {"user-agent": "inggrisin-engine"}
    if auth_secret:
        digest = hmac.new(
            auth_secret.encode(), f"{job_id}.{sha}".encode(), hashlib.sha256
        ).hexdigest()
        headers["x-inggrisin-signature"] = f"sha256={digest}"

    # --- 1) init sesi resumable di app -------------------------------------
    init_body = {
        "job_id": job_id,
        "filename": filename,
        "folder_id": extra.get("drive_folder_id", ""),
        "overwrite_file_id": extra.get("overwrite_file_id", ""),
        "total_bytes": size,
        "sha256": sha,
    }
    last = ""
    session = ""
    chunk = CHUNK_BYTES
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(
                url,
                json=init_body,
                headers={**headers, "x-inggrisin-protocol": "2"},
                timeout=(30, 120),
            )
            if r.status_code == 200:
                j = r.json()
                session = str(j.get("session") or "")
                chunk = int(j.get("chunk_bytes") or CHUNK_BYTES)
                break
            last = f"init HTTP {r.status_code}: {r.text[:200]}"
            if 400 <= r.status_code < 500 and r.status_code != 429:
                raise UploadError("drive_upload_failed", f"The proxy rejected the upload init ({last})")
        except requests.RequestException as e:
            last = f"init network {type(e).__name__}"
        if attempt < MAX_RETRIES:
            wait = 3 * attempt
            log.warn(f"init upload gagal ({last}) — retry {attempt}/{MAX_RETRIES} dalam {wait}s")
            time.sleep(wait)
    if not session:
        raise UploadError(
            "drive_upload_failed", f"Upload init failed after {MAX_RETRIES} attempts: {last}"
        )

    log.info(f"sesi unggah dimulai; {size / 1e6:.1f} MB dalam chunk {chunk / 1e6:.0f} MB")

    # --- 2) kirim chunk per chunk, resume bila gagal ------------------------
    sent = 0
    while sent < size:
        end = min(sent + chunk, size)
        rng = f"bytes {sent}-{end - 1}/{size}"
        ok = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                with path.open("rb") as fh:
                    fh.seek(sent)
                    data = fh.read(end - sent)
                rp = requests.put(
                    url,
                    data=data,
                    headers={
                        **headers,
                        "x-inggrisin-session": session,
                        "content-range": rng,
                        "content-type": content_type,
                    },
                    timeout=(30, 600),
                )
                if rp.status_code in (200, 201):
                    j = rp.json()
                    return UploadResult(
                        drive_file_id=j.get("drive_file_id"),
                        drive_web_url=j.get("drive_web_url"),
                        bytes=int(j.get("bytes") or size),
                        elapsed_s=0.0,
                        mode="proxy-chunked",
                    )
                if rp.status_code == 204:
                    sent = end
                    ok = True
                    break
                last = f"chunk HTTP {rp.status_code}: {rp.text[:150]}"
            except requests.RequestException as e:
                last = f"chunk network {type(e).__name__}"
            # sebelum retry: tanyakan ke Drive berapa byte yang sudah masuk
            received = _query_received(log, url, headers, session, size)
            if received is not None:
                sent = received
                if sent >= size:
                    break
            if attempt < MAX_RETRIES:
                wait = 3 * attempt
                log.warn(
                    f"chunk {rng} gagal ({last}) — retry {attempt}/{MAX_RETRIES} dalam {wait}s"
                )
                time.sleep(wait)
        if sent >= size and not ok:
            # semua byte masuk tetapi respons final belum dilihat: tanyakan status
            fin = _query_received(log, url, headers, session, size, expect_final=True)
            if isinstance(fin, dict):
                return UploadResult(
                    drive_file_id=fin.get("drive_file_id"),
                    drive_web_url=fin.get("drive_web_url"),
                    bytes=int(fin.get("bytes") or size),
                    elapsed_s=0.0,
                    mode="proxy-chunked",
                )
        if not ok and sent < size:
            raise UploadError(
                "drive_upload_failed", f"Chunk upload failed after {MAX_RETRIES} attempts: {last}"
            )

    raise UploadError("drive_upload_failed", f"The upload did not return a final result: {last}")


def _query_received(
    log: Logger,
    url: str,
    headers: dict[str, str],
    session: str,
    size: int,
    expect_final: bool = False,
) -> int | dict[str, str] | None:
    """PUT `content-range: bytes */total` → {received} atau hasil final (dict)."""
    try:
        rs = requests.put(
            url,
            headers={**headers, "x-inggrisin-session": session, "content-range": f"bytes */{size}"},
            timeout=(30, 120),
        )
        if rs.status_code == 200:
            j = rs.json()
            if "drive_file_id" in j:
                return {k: str(v) for k, v in j.items()}
            return int(j.get("received") or 0)
        log.warn(f"status query upload -> HTTP {rs.status_code}")
    except requests.RequestException as e:
        log.warn(f"status query upload gagal: {type(e).__name__}")
    return None if not expect_final else None


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
            "Direct Drive mode is only for small files (<5MB). Use INGGRISIN_RESULT_UPLOAD_URL (the app proxy).",
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
            b"Content-Type: application/json; charset=UTF-8\r\n\r\n",
            json.dumps(metadata).encode(),
            b"\r\n",
            f"--{boundary}\r\n".encode(),
            f"Content-Type: {content_type}\r\n\r\n".encode(),
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
