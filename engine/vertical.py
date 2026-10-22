"""
vertical.py — SATU pass ffmpeg: susun 9:16 + burn subtitle + mux audio.

Aturan keras (docs/00 §7, docs/01 §15):
  * scale/blur DAN burn subtitle WAJIB dalam satu command — dua pass berarti
    dua kali re-encode (kualitas turun dua kali, waktu dobel).
  * sumber horizontal → latar blur (tidak ada bagian gambar terbuang).
  * sumber vertikal → scale/pad secukupnya (sering no-op), tetap satu encode
    karena subtitle harus menempel.
  * audio = voice track (TTS) sepenuhnya; audio asli TIDAK dipertahankan
    (musik tidak dipisahkan — docs/01 §17).

Fungsi pembangun command dipisah dari eksekusi supaya bisa diuji tanpa ffmpeg.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .log import Logger
from .srt import Cue, sanitize_for_burn

TARGET_W = 1080
TARGET_H = 1920
TARGET_FPS = 30
SUBTITLE_BOTTOM_PCT = 19  # di atas safe zone platform (docs/01 §16)
FONT_SIZE = 58


class VerticalError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(slots=True)
class VideoProbe:
    width: int
    height: int
    duration_s: float


def probe_video(log: Logger, path: Path) -> VideoProbe:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise VerticalError("ffmpeg_failed", f"ffprobe gagal: {proc.stderr.strip()[-300:]}")
    data = json.loads(proc.stdout or "{}")
    w = h = 0
    for st in data.get("streams") or []:
        if st.get("codec_type") == "video":
            w = int(st.get("width") or 0)
            h = int(st.get("height") or 0)
            break
    dur = float((data.get("format") or {}).get("duration") or 0.0)
    if w <= 0 or h <= 0:
        raise VerticalError("ffmpeg_failed", "tidak ditemukan stream video di sumber")
    return VideoProbe(width=w, height=h, duration_s=dur)


def classify_aspect(width: int, height: int) -> str:
    if width <= 0 or height <= 0:
        return "unknown"
    r = width / height
    if r > 1.05:
        return "horizontal"
    if r < 0.95:
        return "vertical"
    return "square"


# ---------------------------------------------------------------------------
# ASS (posisi subtitle presisi: MarginV = 19% tinggi — di atas safe zone)
# ---------------------------------------------------------------------------


def ass_time(seconds: float) -> str:
    """Format waktu ASS: H:MM:SS.cc (centisecond)."""
    cs = max(0, round(seconds * 100))
    h, rem = divmod(cs, 360_000)
    m, rem = divmod(rem, 6_000)
    s, cc = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cc:02d}"


_ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,{fs},&H00FFFFFF,&H00FFFFFF,&H00101010,&H96000000,-1,0,0,0,100,100,0,0,1,3.2,2,2,64,64,{mv},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def build_ass(
    cues: list[Cue],
    out: Path,
    *,
    w: int = TARGET_W,
    h: int = TARGET_H,
    bottom_pct: int = SUBTITLE_BOTTOM_PCT,
    font_size: int = FONT_SIZE,
) -> Path:
    margin_v = round(h * bottom_pct / 100)
    parts = [_ASS_HEADER.format(w=w, h=h, fs=font_size, mv=margin_v)]
    for c in cues:
        text = sanitize_for_burn(c.text).replace("\n", "\\N")
        parts.append(f"Dialogue: 0,{ass_time(c.start)},{ass_time(c.end)},Default,,0,0,0,,{text}\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(parts), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Render satu pass
# ---------------------------------------------------------------------------


def _escape_filter_path(p: Path) -> str:
    """Escape path untuk dipakai di filter subtitles='...' ffmpeg (Linux)."""
    s = str(p)
    s = s.replace("\\", "\\\\").replace("'", "\u2019").replace(":", "\\:")
    return s


def build_render_cmd(
    src: Path,
    voice_wav: Path,
    ass: Path,
    out: Path,
    aspect: str,
    *,
    duration_s: float,
    w: int = TARGET_W,
    h: int = TARGET_H,
    fps: int = TARGET_FPS,
    crf: int = 20,
) -> list[str]:
    # Tier bitrate menurut durasi (docs/01 §34): CRF 20 polos membuat file
    # 10+ menit membengkak (126 MB utk sumber 35 MB — upscaling sumber 360p
    # ke 1080x1920 tidak butuh bitrate besar). Untuk video panjang: CRF lebih
    # tinggi + maxrate/bufsize sebagai plafon, audio 96k cukup utk suara orang.
    if crf == 20 and duration_s > 300:
        crf = 23
        maxrate, bufsize, audio_k = "800k", "1600k", "96k"
    elif crf == 20 and duration_s > 120:
        crf = 21
        maxrate, bufsize, audio_k = "1500k", "3000k", "128k"
    else:
        maxrate, bufsize, audio_k = "2500k", "5000k", "192k"
    sub = f"subtitles='{_escape_filter_path(ass)}'"
    if aspect in ("horizontal", "square", "unknown"):
        # latar blur: split → latar diperbesar+d crop+blur, subjek di tengah
        fc = (
            f"[0:v]split=2[bg][fg];"
            f"[bg]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},gblur=sigma=20[bgout];"
            f"[fg]scale={w}:{h}:force_original_aspect_ratio=decrease[fgout];"
            f"[bgout][fgout]overlay=(W-w)/2:(H-h)/2,{sub}[vout]"
        )
    else:
        # sumber vertikal: scale-down bila perlu + pad (sering no-op) — TANPA blur
        fc = (
            f"[0:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,{sub}[vout]"
        )
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-i",
        str(voice_wav),
        "-filter_complex",
        fc,
        "-map",
        "[vout]",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(crf),
        "-maxrate",
        maxrate,
        "-bufsize",
        bufsize,
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(fps),
        "-c:a",
        "aac",
        "-b:a",
        audio_k,
        "-ar",
        "44100",
        "-t",
        f"{max(0.1, duration_s):.3f}",
        "-movflags",
        "+faststart",
        str(out),
    ]


def render(log: Logger, cmd: list[str]) -> Path:
    out = Path(cmd[-1])
    log.info(
        f"render satu pass → {out.name} (filter_complex {len(cmd[cmd.index('-filter_complex') + 1])} chars)"
    )
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-12:])
        raise VerticalError(
            "vertical_render_failed", f"ffmpeg render gagal (exit {proc.returncode}):\n{tail}"
        )
    if not out.exists() or out.stat().st_size == 0:
        raise VerticalError("vertical_render_failed", "hasil render kosong")
    return out
