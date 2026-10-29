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
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .log import Logger
from .srt import Cue, sanitize_for_burn

TARGET_W = 1080
TARGET_H = 1920
TARGET_W_SMALL = 720
TARGET_H_SMALL = 1280
TARGET_FPS = 30
FPS_MAX = 60  # sumber 60 fps dipertahankan; di atas itu diturunkan (hemat bit)
SUBTITLE_BOTTOM_PCT = 19  # di atas safe zone platform (docs/01 §16)
FONT_SIZE = 58
OUTLINE = 3.2  # tebal garis luar ASS pada font referensi (58)
SHADOW = 2.0
# Referensi gaya subtitle = kanvas vertical 1080x1920 ("versi vertical sudah
# bagus"). Font diskalakan menurut DIAGONAL kanvas sehingga kanvas 9:16 dan
# 16:9 di kelas resolusi yang sama mendapat gaya absolut identik, dan outline/
# shadow diskalakan menurut font sehingga RASIO tebal-garis-vs-glyph konstan
# (dulu Outline konstan 3.2 unit PlayRes: di kanvas pendek fs mengecil tapi
# outline tidak → subtitle desktop terlihat bergaris tebal & font kecil).
REF_DIAG = (TARGET_W**2 + TARGET_H**2) ** 0.5


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
    fps: float = 0.0


def pick_fps(vp: VideoProbe, fallback: int = TARGET_FPS) -> int:
    """fps keluaran = fps sumber (dipertahankan supaya gerak tidak patah/
    diduplikasi), dibulatkan & dijepit [15, FPS_MAX]; probe gagal → fallback.

    Dulu `-r 30` dipaksa untuk semua sumber: video 60 fps kehilangan setengah
    temporal resolution-nya (= merusak quality, req task kualitas).
    """
    if vp.fps <= 0:
        return fallback
    return int(min(FPS_MAX, max(15, round(vp.fps))))


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
        raise VerticalError("ffmpeg_failed", f"ffprobe failed: {proc.stderr.strip()[-300:]}")
    data = json.loads(proc.stdout or "{}")
    w = h = 0
    fps = 0.0
    for st in data.get("streams") or []:
        if st.get("codec_type") == "video":
            w = int(st.get("width") or 0)
            h = int(st.get("height") or 0)
            num, _, den = str(st.get("avg_frame_rate") or "0/1").partition("/")
            try:
                fps = float(num) / float(den or 1)
            except ValueError:
                fps = 0.0
            break
    dur = float((data.get("format") or {}).get("duration") or 0.0)
    if w <= 0 or h <= 0:
        raise VerticalError("ffmpeg_failed", "no video stream found in the source")
    return VideoProbe(width=w, height=h, duration_s=dur, fps=fps)


def classify_aspect(width: int, height: int) -> str:
    if width <= 0 or height <= 0:
        return "unknown"
    r = width / height
    if r > 1.05:
        return "horizontal"
    if r < 0.95:
        return "vertical"
    return "square"


def pick_target_size(src_w: int, src_h: int) -> tuple[int, int]:
    """Kanvas 9:16 ber-tier resolusi sumber (docs/01 §35).

    Upscaling sumber rendah-res ke 1080x1920 hanya menambah volume file tanpa
    menambah detail (kasus job_cOBvywQXvtdQCcK1: sumber 360p 37,5 MB → hasil
    60,7 MB di kanvas 1080). Sisi pendek sumber < 1080 → kanvas 720x1280;
    subtitle burn-in TETAP tajam karena ASS dirender pada resolusi keluaran
    (font size diskalakan proporsional PlayResY di build_ass).
    Probe tak diketahui (≤ 0) → kanvas penuh seperti sebelumnya.
    """
    if src_w <= 0 or src_h <= 0:
        return TARGET_W, TARGET_H
    if min(src_w, src_h) >= TARGET_W:
        return TARGET_W, TARGET_H
    return TARGET_W_SMALL, TARGET_H_SMALL


def ratio_label(width: int, height: int) -> str:
    """Label rasio kanvas keluaran — untuk field INFORMASIONAL `result.aspect`.

    Bukan input keputusan render apa pun; hanya laporan ke app supaya log/DB
    menyebut bentuk video yang benar-benar dihasilkan (output "Asli" bisa
    16:9, 1:1, 4:5, dst). Rasio direduksi dengan gcd; kombinasi ganjil
    (mis. 854x480 → 427:240) dilaporkan 'custom' daripada pecahan panjang
    yang tidak bisa dibaca orang.
    """
    if width <= 0 or height <= 0:
        return "unknown"
    g = math.gcd(width, height) or 1
    w, h = width // g, height // g
    if w <= 40 and h <= 40:
        return f"{w}:{h}"
    return "custom"


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
Style: Default,DejaVu Sans,{fs},&H00FFFFFF,&H00FFFFFF,&H00101010,&H96000000,-1,0,0,0,100,100,0,0,1,{ol:g},{sh:g},2,64,64,{mv},1

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
    font_size: int | None = None,
) -> Path:
    margin_v = round(h * bottom_pct / 100)
    # Font mengikuti DIAGONAL kanvas relatif ke referensi vertical (REF_DIAG):
    # kanvas 9:16 mendapat angka yang SAMA persis seperti sebelumnya
    # (58 @ 1080x1920, 39 @ 720x1280); kanvas landscape/square mendapat font
    # sekelas resolusi yang sama alih-alih mengecil mengikuti tinggi
    # (1920x1080 dulu fs 33 → subtitle desktop terlihat kecil).
    # Outline & shadow proporsional ke font supaya rasio tebal-garis identik di
    # semua kanvas — keluhan "outline desktop terlalu tebal" datang dari Outline
    # konstan 3.2 unit PlayRes sementara fs menyusut.
    fs = round(FONT_SIZE * (w**2 + h**2) ** 0.5 / REF_DIAG) if font_size is None else font_size
    scale = fs / FONT_SIZE
    ol = round(OUTLINE * scale, 1)
    sh = round(SHADOW * scale, 1)
    parts = [_ASS_HEADER.format(w=w, h=h, fs=fs, ol=ol, sh=sh, mv=margin_v)]
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
    crf: int = 18,
) -> list[str]:
    # Tier kualitas menurut DURASI × JUMLAH PIKSEL kanvas (bukan tinggi kanvas:
    # desktop 1920x1080 punya piksel sama dengan vertical 1080x1920, dulu malah
    # kena plafon kelas kecil 1500k → hasil desktop lembek). CRF 18 = nyaris
    # transparan untuk source feed; tier panjang melonggar bertahap supaya file
    # 10+ menit tidak meledak (kasus lama: 126 MB) tapi TIDAK lagi serendah
    # 800k yang visibly merusak (req: "enhance tanpa merusak quality").
    # Preset medium (≤5 mnt) / fast (>5 mnt): efisiensi kompresi lebih baik pada
    # CRF sama = kualitas naik tanpa bitrate naik; encoder tetap satu pass.
    # Plafon bitrate PROPORSIONAL JUMLAH PIKSEL kanvas: plafonflat kelas
    # besar/kecil dulu membuat kanvas lebar ber-res tinggi (mis. sumber 4K
    # pada output "Asli", 8,3 Mpx) dipaksa 12M — CRF 18 minta jauh lebih
    # banyak sehingga adegan gerak pecah blok, sementara versi vertical
    # (downscale ke 2 Mpx) aman. Sekarang maxrate = kbps-per-Mpx menurut
    # tier durasi (lantai 4M untuk kanvas mungil, atap 48M hanya tersentuh
    # ≥4K); CRF tetap pengendali mutu utama, maxrate hanya penjaga puncak.
    mpx = w * h / 1_000_000
    if crf == 18 and duration_s > 300:
        crf, preset, k = 21, "fast", 3500
    elif crf == 18 and duration_s > 120:
        crf, preset, k = 19, "medium", 5500
    else:
        preset, k = "medium", 9000
    rate_k = int(min(48_000, max(4000, round(mpx * k / 100) * 100)))
    maxrate, bufsize = f"{rate_k}k", f"{rate_k * 2}k"
    audio_k = "192k"  # suara manusia + BGM: 192k AAC ≈ transparan, semua tier
    sub = f"subtitles='{_escape_filter_path(ass)}'"
    if aspect == "source":
        # aspek asli: kanvas == dimensi sumber → scale no-op, subtitle burn-in
        fc = f"[0:v]scale={w}:{h}:force_original_aspect_ratio=decrease,{sub}[vout]"
    elif aspect in ("horizontal", "square", "unknown"):
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
        preset,
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
            "vertical_render_failed", f"ffmpeg render failed (exit {proc.returncode}):\n{tail}"
        )
    if not out.exists() or out.stat().st_size == 0:
        raise VerticalError("vertical_render_failed", "render output is empty")
    return out
