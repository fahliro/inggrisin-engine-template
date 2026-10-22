"""
Logging yang aman untuk GitHub Actions.

Dua aturan yang tidak boleh dilanggar:
  1. Jangan pernah menulis secret ke log, walau sudah di-mask. Masking GitHub
     bersifat best-effort dan bisa bocor lewat transformasi string
     (mis. secret yang di-base64 atau dipotong).
  2. Jangan menulis URL lengkap yang mengandung token. Selalu disanitasi.

Selain ke stdout, log juga ditulis ke file yang di-upload sebagai artifact
debug (retention 3 hari).
"""

from __future__ import annotations

import re
import sys
import time
from collections.abc import Iterable
from pathlib import Path

_REDACT_PATTERNS: list[re.Pattern[str]] = [
    # query string token
    re.compile(r"([?&](?:token|key|api[_-]?key|access_token|sig)=)[^&\s]+", re.I),
    # bearer token
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]+"),
    # sk- style keys
    re.compile(r"\b(?:sk|gsk|pk)-[A-Za-z0-9_\-]{12,}\b"),
]

_SECRETS: set[str] = set()


def register_secrets(*values: str | None) -> None:
    """Daftarkan nilai yang harus disensor total dari log."""
    for v in values:
        if v and len(v) >= 6:
            _SECRETS.add(v)


def sanitize(text: str) -> str:
    out = text
    for s in _SECRETS:
        if s in out:
            out = out.replace(s, "***REDACTED***")
    for pat in _REDACT_PATTERNS:
        out = pat.sub(lambda m: (m.group(1) if m.groups() else "") + "***", out)
    return out


class Logger:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._t0 = time.monotonic()
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def _emit(self, level: str, msg: str, **kw: object) -> None:
        elapsed = time.monotonic() - self._t0
        extra = " ".join(f"{k}={v}" for k, v in kw.items() if v is not None)
        line = f"[{elapsed:7.2f}s] {level.upper():5s} {msg}" + (f" | {extra}" if extra else "")
        line = sanitize(line)
        print(line, flush=True)
        if self._path:
            try:
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:
                pass

    def info(self, msg: str, **kw: object) -> None:
        self._emit("info", msg, **kw)

    def warn(self, msg: str, **kw: object) -> None:
        self._emit("warn", msg, **kw)
        # annotation GitHub supaya terlihat di UI Actions
        print(f"::warning::{sanitize(msg)}", file=sys.stderr, flush=True)

    def error(self, msg: str, **kw: object) -> None:
        self._emit("error", msg, **kw)
        print(f"::error::{sanitize(msg)}", file=sys.stderr, flush=True)

    def group(self, title: str) -> None:
        print(f"::group::{sanitize(title)}", flush=True)
        self.info(title)

    def endgroup(self) -> None:
        print("::endgroup::", flush=True)

    def mask(self, values: Iterable[str | None]) -> None:
        """Minta GitHub me-mask nilai-nilai ini di seluruh log run."""
        for v in values:
            if v:
                print(f"::add-mask::{v}", flush=True)
                register_secrets(v)
