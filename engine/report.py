"""
Penulis report.json — satu-satunya tempat engine mencatat status untuk
dibaca `callback_guard` dan di-upload sebagai artifact debug.

Ditulis INCREMENTAL: setiap tahap selesai, file di-update. Kalau runner
mati mendadak, report terakhir tetap berguna untuk diagnosis.

PENTING: jangan pernah menulis secret ke report. Field yang bisa memuat
nilai sensitif (URL dengan token, api key) disanitasi lewat log.sanitize().
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .log import sanitize


class Report:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict[str, Any] = {
            "schema": 1,
            "started_at": int(time.time() * 1000),
            "status": "running",
            "callback_delivered": False,
            "stages": {},
        }
        self._flush()

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # tulis atomik: file sementara lalu rename
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def stage_done(self, name: str, seconds: float, **extra: Any) -> None:
        self._data["stages"][name] = {"seconds": round(seconds, 2), **extra}
        self._flush()

    def set(self, **kw: Any) -> None:
        for k, v in kw.items():
            self._data[k] = v
        self._flush()

    def fail(self, code: str, message: str, step: str) -> None:
        self._data.update(
            {
                "status": "failed",
                "error_code": code,
                "error_message": sanitize(str(message))[:2000],
                "error_step": step,
                "finished_at": int(time.time() * 1000),
            }
        )
        self._flush()

    def succeed(self) -> None:
        self._data.update({"status": "done", "finished_at": int(time.time() * 1000)})
        self._flush()

    def callback_delivered(self, ok: bool) -> None:
        self._data["callback_delivered"] = bool(ok)
        self._flush()

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    @property
    def timings(self) -> dict[str, float]:
        return {f"{k}_s": float(v.get("seconds", 0.0)) for k, v in self._data.get("stages", {}).items()}
