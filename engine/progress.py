"""
Heartbeat progres ke app.

Mengapa perlu? Tanpa ini, UI hanya menunjukkan "running" selama 5-15 menit.
Dengan heartbeat, user melihat tahapan nyata: downloading → transcribing →
translating → synthesizing → rendering → uploading.
(mode revision melewati transcribing/translating — docs/01 §20)

Desain:
  - best-effort. Kegagalan mengirim progres TIDAK boleh menggagalkan job.
  - di-throttle: minimal 5 detik antar kiriman untuk status yang sama.
  - memakai HMAC yang sama dengan callback hasil.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any

import requests

from .log import Logger

# urutan tahap → rentang persen, dipakai supaya progress bar monoton naik
_STAGE_PERCENT: dict[str, int] = {
    "starting": 2,
    "downloading": 10,
    "transcribing": 30,
    "translating": 50,
    "synthesizing": 65,
    "rendering": 80,
    "uploading": 92,
    "done": 100,
}


class Progress:
    def __init__(
        self,
        log: Logger,
        callback_url: str,
        secret: str,
        job_id: str,
        contract_version: str,
        run_id: str = "",
        run_url: str = "",
        enabled: bool = True,
    ) -> None:
        self._log = log
        self._url = callback_url
        self._secret = secret.encode()
        self._job_id = job_id
        self._contract = contract_version
        self._run_id = run_id
        self._run_url = run_url
        self._enabled = enabled
        self._last_sent: dict[str, float] = {}
        self._max_percent = 0

    def _sign(self, body: str) -> str:
        digest = hmac.new(self._secret, f"{self._job_id}.{body}".encode(), hashlib.sha256).hexdigest()
        return f"sha256={digest}"

    def send(self, status: str, message: str = "", percent: int | None = None) -> None:
        if not self._enabled or not self._url:
            return
        pct = percent if percent is not None else _STAGE_PERCENT.get(status, 0)
        # progress tidak boleh mundur
        pct = max(pct, self._max_percent)
        self._max_percent = pct

        now = time.monotonic()
        key = f"{status}:{pct // 10}"
        if now - self._last_sent.get(key, 0) < 5:
            return
        self._last_sent[key] = now

        payload = {
            "contract_version": self._contract,
            "job_id": self._job_id,
            "status": status,
            "percent": pct,
            "message": message[:500],
            "at": int(time.time() * 1000),
        }
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        headers = {
            "content-type": "application/json",
            "x-inggrisin-event": "progress",
            "x-inggrisin-signature": self._sign(body),
            "user-agent": f"inggrisin-engine/{self._contract}",
        }
        if self._run_id:
            headers["x-github-run-id"] = self._run_id
        if self._run_url:
            headers["x-github-run-url"] = self._run_url

        try:
            r = requests.post(self._url, data=body.encode(), headers=headers, timeout=15)
            if r.status_code >= 400:
                self._log.warn(f"progress ditolak ({r.status_code})")
        except requests.RequestException as e:
            # jangan pernah gagal karena heartbeat
            self._log.warn(f"progress gagal dikirim: {type(e).__name__}")

    def stage(self, status: str, message: str = "", **extra: Any) -> None:
        self._log.info(message or status, stage=status, **extra)
        self.send(status, message)
