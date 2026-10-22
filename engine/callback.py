"""
Kirim hasil ke app dengan signature HMAC.

Format signature (harus cocok dengan Worker — lihat apps/web/src/lib/crypto/hmac.ts):
    X-Inggrisin-Signature: sha256=HMAC_SHA256(secret, f"{job_id}.{raw_body}")

Kenapa raw_body ikut ditandatangani (bukan cuma field tertentu)?
  Supaya tidak ada bagian payload yang bisa diubah di tengah jalan.
  Worker memverifikasi terhadap body mentah yang ia terima, jadi serialisasi
  di sini HARUS byte-identik dengan yang dikirim. Karena itu kita serialize
  sekali, lalu mengirim bytes yang sama persis.

Ukuran: Worker membatasi 1 MB. Kalau segmen terlalu banyak, payload dipotong
dan `overflow` diisi referensi ke file JSON lengkap di Drive.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any

import requests

from .log import Logger

MAX_BODY_BYTES = 900 * 1024  # di bawah batas 1 MB Worker, beri margin
MAX_RETRIES = 4


class CallbackError(Exception):
    pass


def sign(secret: str, job_id: str, raw_body: bytes) -> str:
    digest = hmac.new(
        secret.encode(), job_id.encode() + b"." + raw_body, hashlib.sha256
    ).hexdigest()
    return f"sha256={digest}"


def _headers(
    secret: str, job_id: str, raw_body: bytes, event: str, run_id: str, run_url: str
) -> dict[str, str]:
    h = {
        "content-type": "application/json",
        "x-inggrisin-event": event,
        "x-inggrisin-signature": sign(secret, job_id, raw_body),
        "user-agent": "inggrisin-engine",
    }
    if run_id:
        h["x-github-run-id"] = run_id
    if run_url:
        h["x-github-run-url"] = run_url
    return h


def send(
    log: Logger,
    *,
    url: str,
    secret: str,
    payload: dict[str, Any],
    event: str = "result",
    run_id: str = "",
    run_url: str = "",
) -> bool:
    """
    Kirim payload. Mengembalikan True bila app menjawab 2xx.

    Retry hanya untuk 429/5xx dan error jaringan. 4xx berarti payload kita
    salah — retry tidak akan menolong dan hanya memperlama run.
    """
    if not url:
        log.warn("callback_url kosong — hasil tidak bisa dikirim balik")
        return False

    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(body) > MAX_BODY_BYTES:
        raise CallbackError(
            f"payload {len(body)} byte melebihi batas {MAX_BODY_BYTES}. "
            "Potong `segments` dan isi `overflow` sebelum mengirim."
        )

    headers = _headers(secret, str(payload.get("job_id", "")), body, event, run_id, run_url)

    last = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(url, data=body, headers=headers, timeout=(15, 120))
        except requests.RequestException as e:
            last = f"network {type(e).__name__}"
            log.warn(f"callback attempt {attempt}: {last}")
            time.sleep(min(2**attempt, 30))
            continue

        if 200 <= r.status_code < 300:
            log.info(f"callback terkirim ({r.status_code}, {len(body)} byte)")
            return True

        last = f"HTTP {r.status_code} {r.text[:200]}"
        if 400 <= r.status_code < 500 and r.status_code != 429:
            log.error(f"callback ditolak permanen: {last}")
            return False

        log.warn(f"callback attempt {attempt} gagal: {last}")
        time.sleep(min(2**attempt, 30))

    log.error(f"callback gagal setelah {MAX_RETRIES} percobaan: {last}")
    return False


def shrink_to_fit(payload: dict[str, Any], log: Logger) -> dict[str, Any]:
    """
    Potong `segments` sampai payload muat di bawah MAX_BODY_BYTES.
    Mengembalikan payload baru (tidak memutasi yang asli) dengan
    `truncated=True` dan jumlah total di `segments_total`.
    """
    result = payload.get("result")
    if not isinstance(result, dict):
        return payload

    segments = result.get("segments") or []
    total = len(segments)
    out = json.loads(json.dumps(payload))  # deep copy sederhana & aman
    out["result"]["segments_total"] = total

    def size(p: dict[str, Any]) -> int:
        return len(json.dumps(p, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))

    if size(out) <= MAX_BODY_BYTES:
        return out

    # Strategi bertahap, dari yang paling murah dikorbankan:
    #   1. buang `hints` (bisa diambil lagi dari file overflow di Drive)
    #   2. baru potong jumlah `segments`
    for s in out["result"].get("segments", []):
        s.pop("hints", None)
    out["result"]["hints_dropped"] = True
    log.warn("payload kebesaran — hints dibuang dari callback (tersedia di file overflow Drive)")
    if size(out) <= MAX_BODY_BYTES:
        return out

    lo, hi = 0, total
    while lo < hi:
        mid = (lo + hi + 1) // 2
        trial = json.loads(json.dumps(out))
        trial["result"]["segments"] = trial["result"]["segments"][:mid]
        if size(trial) <= MAX_BODY_BYTES:
            lo = mid
        else:
            hi = mid - 1

    out["result"]["segments"] = out["result"]["segments"][:lo]
    out["result"]["truncated"] = True
    log.warn(f"payload dipotong: {lo}/{total} segmen dikirim lewat callback")
    return out
