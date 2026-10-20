"""
Jaring pengaman terakhir: pastikan app SELALU menerima kabar, bahkan ketika
`engine.main` crash sebelum sempat mengirim callback.

Dipanggil dari step workflow dengan `if: always()`.

Perilaku:
  - kalau report.json sudah berisi status terminal → main sudah mengirim
    callback; kita hanya memverifikasi ukurannya dan keluar diam-diam
  - kalau tidak → susun payload kegagalan dari informasi yang tersedia
    (conclusion GitHub + report parsial) dan kirim
  - kegagalan di sini TIDAK boleh membuat step merah (sudah `always()`),
    tapi kita tetap catat sebagai warning
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _load_report(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--conclusion", default="unknown")
    args = ap.parse_args()

    job_id = os.environ.get("INGGRISIN_JOB_ID", "").strip()
    url = os.environ.get("INGGRISIN_CALLBACK_URL", "").strip()
    secret = os.environ.get("INGGRISIN_CALLBACK_SECRET", "").strip()
    contract = os.environ.get("INGGRISIN_CONTRACT_VERSION", "unknown")
    engine_version = os.environ.get("INGGRISIN_ENGINE_VERSION", "unknown")
    run_id = os.environ.get("GITHUB_RUN_ID_OUT") or os.environ.get("GITHUB_RUN_ID", "")
    run_url = os.environ.get("GITHUB_RUN_URL_OUT", "")

    if not (job_id and url and secret):
        print("::warning::callback_guard: env tidak lengkap, lewati", file=sys.stderr)
        return 0

    report = _load_report(Path(args.report))
    if report.get("callback_delivered"):
        print(f"callback sudah terkirim oleh engine (status={report.get('status')})")
        return 0

    # susun payload kegagalan
    code = report.get("error_code") or (
        "cancelled_by_user" if args.conclusion == "cancelled" else "unknown"
    )
    message = report.get("error_message") or (
        f"Workflow berakhir dengan status '{args.conclusion}' tanpa mengirim hasil. "
        "Lihat log Actions untuk detail."
    )

    payload = {
        "contract_version": contract,
        "job_id": job_id,
        "status": "cancelled" if args.conclusion == "cancelled" else "failed",
        "error": {"code": code, "message": str(message)[:2000], "step": report.get("error_step", "unknown")},
        "engine_version": engine_version,
        "finished_at": int(time.time() * 1000),
    }

    # import di sini supaya modul ini tetap bisa dijalankan walau dep belum terpasang
    try:
        from .callback import send
        from .log import Logger

        log = Logger()
        ok = send(
            log,
            url=url,
            secret=secret,
            payload=payload,
            event="result",
            run_id=run_id,
            run_url=run_url,
        )
        print("callback_guard: terkirim" if ok else "callback_guard: GAGAL terkirim")
        return 0 if ok else 1
    except Exception as e:  # noqa: BLE001
        print(f"::warning::callback_guard error: {type(e).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
