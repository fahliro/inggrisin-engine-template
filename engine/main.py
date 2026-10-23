"""
inggrisin engine v2 — orkestrator DUBBING dengan dua mode.

  --mode first     pipeline penuh (docs/01 §36 — Whisper 1× + gpt-oss 1×):
                   download → extract audio → Groq Whisper 1×
                   (transcribe id = teks asli + timing kanonik) → terjemahan
                   gpt-oss 1:1 per segmen (groq_llm, chunked) → Piper TTS
                   (satu suara) → voice track + BGM asli (demucs) →
                   vertical.py SATU pass (9:16 blur + burn ASS + mux) → Drive

  --mode revision  SRT pass-through (docs/00 §6, docs/01 §20):
                   fetch SRT+manifest (tervalidasi di Worker) → download video
                   asli → TTS → voice track → render → TIMPA file Drive.
                   TIDAK memanggil Groq maupun OpenRouter sama sekali.

Kontrak keluaran: SATU file MP4 9:16 subtitle menempel. Tidak ada audio
terpisah, tidak ada SRT sebagai hasil user.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests

from . import audio, callback, download, groq_llm, srt, transcribe, tts, vertical
from .config import Config, ConfigError
from .log import Logger, register_secrets
from .progress import Progress
from .report import Report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="inggrisin-engine")
    p.add_argument("--workspace", type=Path, default=Path("work"))
    p.add_argument("--out", type=Path, default=Path("out"))
    p.add_argument("--report", type=Path, default=Path("out/report.json"))
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--selfcheck-voice", default="en_US-lessac-medium")
    return p.parse_args()


def _ffmpeg_version() -> str:
    """Token versi ringkas ('6.1.1-3ubuntu5') — kontrak callback membatasi 32 char.

    Baris pertama `ffmpeg -version` utuh ("ffmpeg version 6.1.1-3ubuntu5
    Copyright ...") bisa ~80 char dan membuat callback hasil DITOLAK skema.
    """
    try:
        out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=False)
        line = (out.stdout or "").splitlines()[0] if out.returncode == 0 else ""
        parts = line.split()
        return (parts[2] if len(parts) > 2 else line)[:32] or "?"
    except OSError:
        return "?"


# ---------------------------------------------------------------------------
# payload callback (bentuk = CallbackPayload di packages/shared)
# ---------------------------------------------------------------------------


def _subtitle_dicts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        d = {
            "idx": int(r["idx"]),
            "start": round(float(r["start"]), 3),
            "end": round(float(r["end"]), 3),
            "textId": str(r.get("text_id", "") or ""),
            "textEn": str(r.get("text_en", "") or ""),
            "fit": r.get("fit", "ok"),
            "tempo": round(float(r.get("tempo", 1.0)), 2),
        }
        if r.get("note"):
            d["note"] = str(r["note"])[:200]
        out.append(d)
    return out


def _done_payload(cfg: Config, result: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "contract_version": cfg.contract_version,
        "job_id": cfg.job_id,
        "status": "done",
        "result": result,
        "engine_version": cfg.engine_version,
        "runner": {
            "os": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
            "ffmpeg": _ffmpeg_version(),
        },
        "finished_at": int(time.time() * 1000),
    }
    if cfg.mode == "revision":
        payload["revision_number"] = cfg.revision_number
    return payload


def _fail_payload(
    cfg: Config | None,
    code: str,
    message: str,
    step: str,
    contract_version: str = "unknown",
    job_id: str = "",
) -> dict[str, Any]:
    p: dict[str, Any] = {
        "contract_version": contract_version,
        "job_id": job_id,
        "status": "failed",
        "error": {"code": code, "message": message[:2000], "step": step[:64]},
        "finished_at": int(time.time() * 1000),
    }
    if cfg is not None and cfg.mode == "revision":
        p["revision_number"] = cfg.revision_number
    return p


# ---------------------------------------------------------------------------
# tahap bersama: TTS → tempo → voice track → ASS → render → upload
# ---------------------------------------------------------------------------


def _tts_to_render(
    cfg: Config,
    log: Logger,
    prog: Progress,
    report: Report,
    timings: dict[str, float],
    segs: list[dict[str, Any]],  # {idx,start,end,text_en}
    total_duration: float,
    aspect: str,
    step_ref: list[str],
    target_wh: tuple[int, int] = (vertical.TARGET_W, vertical.TARGET_H),
) -> None:
    """Sintesis per baris → pasakan durasi (atempo ≤ 1,25) → rakit track → render."""
    step_ref[0] = "tts"
    prog.stage("synthesizing", "Merekam suara Inggris (satu suara otomatis)")
    t = time.monotonic()
    lines = [s["text_en"] for s in segs]
    files = tts.synthesize_lines(log, cfg.voice, lines, cfg.lines_dir)
    timings["tts_s"] = round(time.monotonic() - t, 2)
    spoken = sum(1 for f in files if f is not None)
    report.stage_done("tts", timings["tts_s"], lines=len(lines), synthesized=spoken)

    step_ref[0] = "tempo"
    placements: list[tuple[float, Path]] = []
    missing = 0
    for s, wav in zip(segs, files, strict=True):
        if wav is None:
            missing += 1
            continue
        slot = max(0.2, float(s["end"]) - float(s["start"]))
        measured = audio.duration_of(log, wav)
        need = measured / slot
        if need > 1.01:
            tempo = min(need, srt.TEMPO_MAX)
            dst = cfg.tempo_dir / wav.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            audio.apply_tempo(log, wav, dst, tempo)
            s["tempo"] = round(tempo, 2)
            if tempo >= srt.TEMPO_MAX - 1e-6 and need > srt.TEMPO_MAX:
                s["fit"] = "tempo"
                s["note"] = (
                    s.get("note") or ""
                ) + " · dipercepat maksimal; sisa dibiarkan mengalir"
            wav = dst
        placements.append((float(s["start"]), wav))
    if missing:
        log.warn(f"{missing} baris tanpa suara — celahnya hening (dilaporkan di callback)")

    step_ref[0] = "voice_track"
    prog.stage("synthesizing", "Menyusun suara ke timeline asli")
    t = time.monotonic()
    audio.build_voice_track(log, placements, total_duration, cfg.voice_track_path)
    timings["voice_track_s"] = round(time.monotonic() - t, 2)
    report.stage_done("voice_track", timings["voice_track_s"], placements=len(placements))

    # BGM/SFX asli dipertahankan (docs/01 §27): pisahkan vokal asli (demucs) →
    # mix dengan track dub. GAGAL DI MANA PUN → lanjut dengan audio dub bersih
    # (musik adalah peningkatan, bukan syarat hasil valid).
    final_audio = cfg.voice_track_path
    if cfg.options.get("music"):
        step_ref[0] = "music"
        prog.stage("synthesizing", "Mempertahankan BGM/SFX asli")
        t = time.monotonic()
        with prog.heartbeat(
            "synthesizing",
            "Mempertahankan BGM/SFX asli (audio panjang bisa beberapa menit)",
            60,
        ):
            accom = audio.separate_accompaniment(
                log, cfg.source_path, cfg.music_dir, cfg.accompaniment_path
            )
            if accom is not None:
                audio.mix_voice_with_music(
                    log, cfg.voice_track_path, accom, total_duration, cfg.final_audio_path
                )
                final_audio = cfg.final_audio_path
                timings["music_s"] = round(time.monotonic() - t, 2)
                report.stage_done("music", timings["music_s"])

    step_ref[0] = "render"
    prog.stage("rendering", "Menyusun video vertikal + menempel subtitle (satu pass)")
    t = time.monotonic()
    cues = [
        srt.Cue(idx=int(s["idx"]), start=float(s["start"]), end=float(s["end"]), text=s["text_en"])
        for s in segs
    ]
    vertical.build_ass(
        cues,
        cfg.ass_path,
        w=target_wh[0],
        h=target_wh[1],
        bottom_pct=int(cfg.options.get("subtitle_bottom_pct", 19)),
    )
    cmd = vertical.build_render_cmd(
        cfg.source_path,
        final_audio,
        cfg.ass_path,
        cfg.final_video_path,
        aspect,
        duration_s=total_duration,
        w=target_wh[0],
        h=target_wh[1],
    )
    with prog.heartbeat(
        "rendering", "Merender video vertikal + subtitle (video panjang bisa lama)", 60
    ):
        vertical.render(log, cmd)
    timings["render_s"] = round(time.monotonic() - t, 2)
    report.stage_done(
        "render", timings["render_s"], aspect=aspect, bytes=cfg.final_video_path.stat().st_size
    )


def _upload(
    cfg: Config,
    log: Logger,
    prog: Progress,
    report: Report,
    timings: dict[str, float],
    overwrite_file_id: str = "",
) -> dict[str, Any]:
    from . import upload as upload_mod

    prog.stage("uploading", "Menyimpan ke folder Drive-mu")
    t = time.monotonic()
    extra = {"folder_id": cfg.drive_folder_id, "source_name": cfg.source_name}
    if overwrite_file_id:
        extra["overwrite_file_id"] = overwrite_file_id
    res = upload_mod.upload_result(
        log,
        cfg.final_video_path,
        proxy_url=cfg.result_upload_url,
        filename=cfg.dubbed_file_name,
        content_type="video/mp4",
        job_id=cfg.job_id,
        extra_fields=extra,
        auth_secret=cfg.callback_secret,
    )
    timings["upload_s"] = round(time.monotonic() - t, 2)
    report.stage_done("upload", timings["upload_s"], bytes=res.bytes, mode=res.mode)
    return {
        "drive_file_id": res.drive_file_id or "",
        "drive_file_name": cfg.dubbed_file_name,
        "drive_web_url": res.drive_web_url or "",
        "bytes": int(res.bytes),
        "format": "mp4",
        "aspect": "9:16",
        "overwritten": bool(overwrite_file_id),
    }


# ---------------------------------------------------------------------------
# mode: first
# ---------------------------------------------------------------------------


def run_first(
    cfg: Config,
    log: Logger,
    prog: Progress,
    report: Report,
    timings: dict[str, float],
    step_ref: list[str],
) -> dict[str, Any]:
    step_ref[0] = "download"
    prog.stage("downloading", f"Mengunduh {cfg.source_name}")
    t = time.monotonic()
    dl = download.download(
        log, cfg.source_url, cfg.source_path, cfg.source_bytes, cfg.source_sha256
    )
    timings["download_s"] = round(time.monotonic() - t, 2)
    report.stage_done("download", timings["download_s"], bytes=dl.bytes)

    step_ref[0] = "probe"
    vp = vertical.probe_video(log, cfg.source_path)
    aspect = vertical.classify_aspect(vp.width, vp.height)
    total_duration = vp.duration_s or 0.0
    log.info(f"video: {vp.width}x{vp.height} ({aspect}), {total_duration:.1f}s")
    target_wh = vertical.pick_target_size(vp.width, vp.height)
    log.info(f"kanvas render: {target_wh[0]}x{target_wh[1]} (§35)")

    step_ref[0] = "extract"
    prog.stage("downloading", "Memisahkan suara dari gambar")
    t = time.monotonic()
    audio_path, dur2 = audio.extract(log, cfg.source_path, cfg.audio_path, total_duration)
    total_duration = max(total_duration, dur2)
    timings["extract_s"] = round(time.monotonic() - t, 2)
    report.stage_done("extract", timings["extract_s"], duration_s=round(total_duration, 1))

    step_ref[0] = "transcribe"
    prog.stage("transcribing", "Mendengarkan ucapan Indonesia")
    t = time.monotonic()
    # docs/01 §36: SATU pass Groq Whisper (transcribe saja) — teks asli +
    # timing kanonik (manifest). Terjemahan Inggris lahir 1:1 per segmen via
    # groq_llm (gpt-oss): tidak ada lagi dua segmentasi yang harus didamaikan,
    # align lintas-pass (§29/§34) punah, baris duplikat mustahil by construction.
    chunks: list[Path] = []
    if audio.needs_chunking(audio_path):
        chunks = audio.split_chunks(log, audio_path, cfg.chunks_dir)

    def _transcribe(language: str | None, on_chunk: Any = None) -> transcribe.Transcript:
        if chunks:
            return transcribe.transcribe_chunks(
                log,
                cfg.groq_api_key,
                chunks,
                model=cfg.whisper_model,
                language=language,
                on_chunk=on_chunk,
                task="transcribe",
            )
        return transcribe.transcribe_file(
            log,
            cfg.groq_api_key,
            audio_path,
            model=cfg.whisper_model,
            language=language,
            task="transcribe",
        )

    tr = _transcribe(
        None,  # auto-detect primer (§38-eksperimen: potongan segmen lebih tepat
        # sesuai Groq Playground; forced-id turun jadi telinga cadangan guard)
        on_chunk=lambda i, n: prog.send(
            "transcribing", f"Mendengarkan bagian {i}/{n}", percent=20 + int(10 * i / max(n, 1))
        ),
    )
    timings["transcribe_s"] = round(time.monotonic() - t, 2)
    report.stage_done("transcribe", timings["transcribe_s"], segments=len(tr.segments))
    if not tr.text.strip():
        raise RuntimeError("Transkrip kosong — video mungkin hanya musik/keheningan")

    # Guard degenerasi single-pass (§36): tanpa telinga kedua, heuristik
    # keyakinan-ucapan + volume teks yang memutuskan (transcript_degenerate).
    # Transkrip forced-id mencurigakan → satu retry auto-detect; dua-duanya
    # degenerat → gagal jelas, karangan tidak dipublikasikan (§22).
    if transcribe.transcript_degenerate(tr, total_duration):
        log.warn(
            f"transkrip auto-detect dicurigai degenerat ({len(tr.text.strip())} char "
            f"untuk {total_duration:.0f}s) — retry transcribe forced-id"
        )
        tr_auto = _transcribe("id")
        if tr_auto.text.strip() and not transcribe.transcript_degenerate(tr_auto, total_duration):
            log.warn(
                f"transkrip forced-id dipakai: language={tr_auto.language}, "
                f"{len(tr_auto.segments)} segmen"
            )
            tr = tr_auto
        else:
            raise RuntimeError(
                "Transkrip Whisper degenerat di kedua varian (forced-id & auto-detect) — "
                "decoder Groq gagal membaca audio ini (quirk §32). Dengan 1 pass (§36) "
                "tidak ada telinga kedua; job digagalkan jelas daripada menerbitkan "
                "teks karangan (§22)."
            )

    # jejak mentah utk diagnosis pemotongan (§38): batas segmen persis Whisper
    log.info(
        "segmen transkrip: "
        + " | ".join(f"{s.start:.2f}-{s.end:.2f} {(s.text or '')[:48]}" for s in tr.segments)
    )
    id_segs = [s for s in tr.segments if (s.text or "").strip()]
    shutil.rmtree(cfg.chunks_dir, ignore_errors=True)

    # docs/01 §37: pemecahan keterbacaan dilakukan ATAS TEKS SUMBER lebih dulu —
    # batas baris mengikuti kalimat bahasa sumber, lalu terjemahan lahir 1:1
    # per baris sehingga isi EN tiap baris PADAN dengan ID-nya. (Urutan lama —
    # terjemah dulu, pecah paralel proporsional kemudian — menggeser konten EN
    # satu klausa dari ID-nya: batas kalimat dua bahasa tak berimpit; insiden
    # job_pCQawh1rDMasdCeP.)
    rows = [
        {"idx": i, "start": s.start, "end": s.end, "text_id": s.text, "text_en": ""}
        for i, s in enumerate(id_segs)
    ]
    n_rows = len(rows)
    rows = srt.split_long_rows(rows)  # driver otomatis = teks sumber (EN kosong)
    if len(rows) != n_rows:
        log.warn(f"subtitle dipecah demi keterbacaan: {n_rows} → {len(rows)} baris")
    n_before_tail = len(rows)
    rows = srt.merge_short_tails(rows)  # fragmen ekor ("terukur.", "met.") ikut baris sebelumnya
    if len(rows) != n_before_tail:
        log.info(f"fragmen ekor pendek digabung: {n_before_tail} → {len(rows)} baris")

    step_ref[0] = "translate"
    prog.stage("translating", "Menerjemahkan ke Inggris (gpt-oss via Groq)")
    t = time.monotonic()
    items = [(r["idx"], r["text_id"], srt.budget_of(r["start"], r["end"])) for r in rows]
    en_map = groq_llm.translate_segments(
        log,
        cfg.groq_api_key,
        items,
        on_chunk=lambda i, n: prog.send(
            "translating", f"Menerjemahkan bagian {i}/{n}", percent=30 + int(15 * i / max(n, 1))
        ),
    )
    for r in rows:
        r["text_en"] = en_map.get(r["idx"], "")
    timings["translate_s"] = round(time.monotonic() - t, 2)
    empty = sum(1 for r in rows if not r["text_en"])
    report.stage_done(
        "translate",
        timings["translate_s"],
        segments=len(rows),
        via="groq-llm",
        empty=empty,
    )
    if empty == len(rows):
        raise RuntimeError("Terjemahan LLM kosong — tidak ada teks Inggris yang dihasilkan")

    # manifest = sumber kebenaran timing, dikunci mulai sekarang
    manifest = {
        "jobId": cfg.job_id,
        "durationSeconds": round(total_duration, 3),
        "sourceAspect": aspect,
        "segments": [{"idx": r["idx"], "start": r["start"], "end": r["end"]} for r in rows],
    }
    cfg.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    segs = [
        {
            "idx": r["idx"],
            "start": r["start"],
            "end": r["end"],
            "text_id": r["text_id"],
            "text_en": r["text_en"],
            "fit": "ok",
            "tempo": 1.0,
            "note": "",
        }
        for r in rows
    ]
    _tts_to_render(
        cfg, log, prog, report, timings, segs, total_duration, aspect, step_ref, target_wh
    )
    res = _upload(cfg, log, prog, report, timings)

    res.update(
        {
            "duration_seconds": round(total_duration, 3),
            "subtitles": _subtitle_dicts(segs),
            "subtitles_total": len(segs),
            "terms": [],  # NER istilah = produk era lama; §36 terjemahan 1:1 gpt-oss
            "manifest": manifest,
            "truncated": False,
            "timings": {**timings, "total_s": round(sum(timings.values()), 2)},
            "usage": {
                # SATU pass whisper (§36) atas audio. Bila guard degenerasi
                # retry auto-detect, tr diganti — yang terhitung pass terakhir.
                "audio_seconds": round(getattr(tr, "audio_seconds_billed", 0.0) or 0.0, 1),
                "llm_tokens_prompt": 0,  # usage gpt-oss belum diagregasi (§36)
                "llm_tokens_completion": 0,
                "tts_lines": len(segs),
            },
        }
    )
    return res


# ---------------------------------------------------------------------------
# mode: revision — SRT pass-through, TANPA Groq & OpenRouter
# ---------------------------------------------------------------------------


def _fetch_revision(cfg: Config, log: Logger) -> dict[str, Any]:
    log.info(f"mengambil SRT revisi #{cfg.revision_number} dari app")
    try:
        r = requests.get(
            cfg.revisions_url,
            headers={"user-agent": "inggrisin-engine", "accept": "application/json"},
            timeout=(15, 120),
        )
    except requests.RequestException as e:
        raise RuntimeError(f"gagal menghubungi app untuk SRT revisi: {type(e).__name__}") from e
    if r.status_code != 200:
        raise RuntimeError(f"app menolak permintaan SRT revisi: HTTP {r.status_code}")
    data = r.json()
    for field in ("srt", "manifest", "job_id", "revision_number"):
        if field not in data:
            raise RuntimeError(f"payload revisi tidak lengkap: {field} hilang")
    if str(data.get("job_id")) != cfg.job_id:
        raise RuntimeError("job_id payload revisi tidak cocok dengan dispatch")
    if int(data.get("revision_number") or 0) != cfg.revision_number:
        raise RuntimeError("revision_number payload tidak cocok dengan dispatch")
    return data


def run_revision(
    cfg: Config,
    log: Logger,
    prog: Progress,
    report: Report,
    timings: dict[str, float],
    step_ref: list[str],
) -> dict[str, Any]:
    step_ref[0] = "revision_fetch"
    prog.stage("downloading", "Mengambil subtitle revisi")
    t = time.monotonic()
    payload = _fetch_revision(cfg, log)
    srt_text = str(payload["srt"])
    srt.validate_srt_size(srt_text)
    cues = srt.parse_srt(srt_text)
    if not cues:
        raise RuntimeError("SRT revisi kosong/tidak terparse")
    manifest = (
        payload["manifest"]
        if isinstance(payload["manifest"], dict)
        else json.loads(payload["manifest"])
    )
    problems = srt.cues_match_manifest(cues, manifest.get("segments") or [])
    if problems:
        # kontrak dilanggar di sisi app — jangan proses, laporkan apa adanya
        raise RuntimeError("SRT tidak cocok dengan manifest: " + "; ".join(problems[:3]))
    cfg.srt_path.write_text(srt_text, encoding="utf-8")
    timings["revision_fetch_s"] = round(time.monotonic() - t, 2)
    report.stage_done("revision_fetch", timings["revision_fetch_s"], cues=len(cues))

    step_ref[0] = "download"
    prog.stage("downloading", f"Mengambil video asli ({cfg.source_name})")
    t = time.monotonic()
    dl = download.download(
        log, cfg.source_url, cfg.source_path, cfg.source_bytes, cfg.source_sha256
    )
    timings["download_s"] = round(time.monotonic() - t, 2)
    report.stage_done("download", timings["download_s"], bytes=dl.bytes)

    step_ref[0] = "probe"
    vp = vertical.probe_video(log, cfg.source_path)
    aspect = vertical.classify_aspect(vp.width, vp.height)
    total_duration = max(vp.duration_s or 0.0, float(manifest.get("durationSeconds") or 0.0))
    target_wh = vertical.pick_target_size(vp.width, vp.height)
    log.info(f"kanvas render: {target_wh[0]}x{target_wh[1]} (§35)")

    # tempo dihitung ulang deterministik dari teks (TANPA LLM) — sama seperti
    # aturan Worker: est = len / (slot × CPS), clamp 1,25
    segs: list[dict[str, Any]] = []
    for c in cues:
        slot = max(0.1, c.end - c.start)
        est = len(c.text) / (slot * srt.CPS)
        fit = "ok" if est <= 1.0 else "tempo"
        segs.append(
            {
                "idx": c.idx,
                "start": c.start,
                "end": c.end,
                "text_id": "",
                "text_en": c.text,
                "fit": fit,
                "tempo": round(min(max(est, 1.0), srt.TEMPO_MAX), 2),
                "note": "revisi dari subtitle yang disunting",
            }
        )

    _tts_to_render(
        cfg, log, prog, report, timings, segs, total_duration, aspect, step_ref, target_wh
    )
    res = _upload(
        cfg,
        log,
        prog,
        report,
        timings,
        overwrite_file_id=cfg.result_drive_file_id
        or str(payload.get("result_drive_file_id") or ""),
    )

    res.update(
        {
            "duration_seconds": round(total_duration, 3),
            "subtitles": _subtitle_dicts(segs),
            "subtitles_total": len(segs),
            "truncated": False,
            "timings": {**timings, "total_s": round(sum(timings.values()), 2)},
            "usage": {"tts_lines": len(segs)},
        }
    )
    return res


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    log = Logger(args.workspace / "engine.log" if not args.selfcheck else None)

    import os

    log.mask(
        [
            os.environ.get("GROQ_API_KEY"),
            os.environ.get("OPENROUTER_API_KEY"),
            os.environ.get("INGGRISIN_CALLBACK_SECRET"),
        ]
    )

    if args.selfcheck:
        info = tts.selfcheck(log, args.selfcheck_voice)
        print(json.dumps(info, indent=2))
        return 0 if info.get("ok") else 1

    cfg: Config | None = None
    try:
        cfg = Config.load(args.workspace, args.out, args.report)
    except ConfigError as e:
        log.error(f"konfigurasi tidak lengkap: {e.message}")
        _write_minimal_report(args.report, e.code, e.message, "config")
        return 2

    register_secrets(cfg.source_url, cfg.callback_url)
    if cfg.revisions_url:
        register_secrets(cfg.revisions_url)

    report = Report(cfg.report_path)
    report.set(
        job_id=cfg.job_id,
        mode=cfg.mode,
        revision_number=cfg.revision_number if cfg.mode == "revision" else 0,
        contract_version=cfg.contract_version,
        engine_version=cfg.engine_version,
        voice=cfg.voice,
        whisper_model=cfg.whisper_model if cfg.mode == "first" else "",
        llm_model=cfg.llm_model if cfg.mode == "first" else "",
        runner={
            "os": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
            "ffmpeg": _ffmpeg_version(),
        },
    )

    prog = Progress(
        log=log,
        callback_url=cfg.callback_url,
        secret=cfg.callback_secret,
        job_id=cfg.job_id,
        contract_version=cfg.contract_version,
        run_id=cfg.github_run_id,
        run_url=cfg.github_run_url,
    )

    timings: dict[str, float] = {}
    step_ref = ["init"]
    t_all = time.monotonic()
    try:
        prog.stage(
            "starting",
            "Mesin dubbing mulai"
            + (f" (revisi #{cfg.revision_number})" if cfg.mode == "revision" else ""),
        )
        result = (
            run_first(cfg, log, prog, report, timings, step_ref)
            if cfg.mode == "first"
            else run_revision(cfg, log, prog, report, timings, step_ref)
        )
        result.setdefault("timings", {})["total_s"] = round(time.monotonic() - t_all, 2)

        step_ref[0] = "callback"
        payload = callback.shrink_to_fit(_done_payload(cfg, result), log)
        ok = callback.send(
            log,
            url=cfg.callback_url,
            secret=cfg.callback_secret,
            payload=payload,
            event="result",
            run_id=cfg.github_run_id,
            run_url=cfg.github_run_url,
        )
        report.callback_delivered(ok)
        report.succeed()
        if not ok:
            log.error("callback hasil tidak terkirim — reconciler app yang akan menjemput status")
        print(json.dumps({"ok": True, "mode": cfg.mode, "file": str(cfg.final_video_path)}))
        return 0

    except Exception as e:  # noqa: BLE001
        code = getattr(e, "code", None) or "unknown"
        message = getattr(e, "message", None) or str(e)
        return _fail(cfg, log, prog, report, code, message, step_ref[0])


def _fail(
    cfg: Config | None,
    log: Logger,
    prog: Progress | None,
    report: Report | None,
    code: str,
    message: str,
    step: str,
) -> int:
    log.error(f"GAGAL di {step}: [{code}] {message}")
    if report is not None:
        report.fail(code, message, step)
    if cfg is not None:
        payload = _fail_payload(cfg, code, message, step, cfg.contract_version, cfg.job_id)
        try:
            callback.send(
                log,
                url=cfg.callback_url,
                secret=cfg.callback_secret,
                payload=payload,
                event="result",
                run_id=cfg.github_run_id,
                run_url=cfg.github_run_url,
            )
        except Exception as ce:  # noqa: BLE001
            log.warn(f"callback error tidak terkirim: {type(ce).__name__}")
        if prog is not None:
            try:
                prog.send("failed", message[:400])
            except Exception as pe:  # noqa: BLE001
                log.warn(f"progress failed tidak terkirim: {type(pe).__name__}")
    print(json.dumps({"ok": False, "code": code, "message": message[:500], "step": step}))
    return 1


def _write_minimal_report(path: Path, code: str, message: str, step: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"ok": False, "error": {"code": code, "message": message, "error_step": step}},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError:
        pass


if __name__ == "__main__":
    sys.exit(main())
