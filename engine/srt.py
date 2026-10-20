"""
SRT: parse, serialize, sanitize, budget.

Padanan Python dari `packages/shared/src/subtitles.ts` — kedua sisi HARUS
menghasilkan nilai identik (budget, timestamp, normalisasi). Test kedua sisi
memakai vektor yang sama (lihat tests/test_engine.py::test_parity_*).

Kontrak (docs/00 §6):
  - timing berasal dari manifest, tidak pernah dari user
  - budget karakter = slot × CPS × TEMPO_MAX (lantai 12)
  - teks disanitasi sebelum burn-in: tag ASS {..} dan karakter kontrol dibuang
    (Worker sudah menolak; engine tetap defensif — defense in depth)
"""
from __future__ import annotations

import re
from dataclasses import dataclass

CPS = 15
TEMPO_MAX = 1.25
MAX_SRT_BYTES = 512 * 1024

_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ASS_TAG_RE = re.compile(r"\{[^}]*\}")


@dataclass(slots=True)
class Cue:
    idx: int
    start: float
    end: float
    text: str


def budget_of(start: float, end: float) -> int:
    """Identik dengan budgetOf() TS: floor(slot × CPS × 1.25), lantai 12."""
    slot = max(0.0, end - start)
    return max(12, int(slot * CPS * TEMPO_MAX))


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def srt_time(seconds: float) -> str:
    ms = max(0, round(seconds * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, mm = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{mm:03d}"


_TIME_RE = re.compile(r"^(\d{2,}):(\d{2}):(\d{2})[,.](\d{3})$")


def parse_srt_time(v: str) -> float | None:
    m = _TIME_RE.match(v.strip())
    if not m:
        return None
    h, mi, s, ms = (int(x) for x in m.groups())
    return h * 3600 + mi * 60 + s + ms / 1000


def serialize_srt(cues: list[Cue]) -> str:
    out: list[str] = []
    for i, c in enumerate(cues, start=1):
        text = normalize_text(c.text).replace("\n", " ")
        out.append(f"{i}\n{srt_time(c.start)} --> {srt_time(c.end)}\n{text}\n")
    return "\n".join(out) + "\n" if out else ""


def parse_srt(srt: str) -> list[Cue]:
    cues: list[Cue] = []
    for block in re.split(r"\n\s*\n", (srt or "").replace("\r", "")):
        lines = [ln for ln in block.split("\n") if ln.strip() != ""]
        if len(lines) < 2:
            continue
        t = 1 if re.fullmatch(r"\d+", lines[0].strip()) else 0
        parts = lines[t].split("-->") if t < len(lines) else []
        if len(parts) != 2:
            continue
        start, end = parse_srt_time(parts[0]), parse_srt_time(parts[1])
        if start is None or end is None:
            continue
        cues.append(
            Cue(
                idx=len(cues),
                start=start,
                end=end,
                text=normalize_text(" ".join(lines[t + 1 :])),
            )
        )
    return cues


def sanitize_for_burn(text: str) -> str:
    """Pertahanan terakhir sebelum teks masuk ASS/libass: buang tag {..},
    karakter kontrol, dan sisa kurung kurawal. Worker sudah menolak teks
    seperti ini; kalau sampai sini, sesuatu melewati validasi — jangan
    biarkan menjadi directive libass."""
    t = _ASS_TAG_RE.sub("", text or "")
    t = _CTRL_RE.sub("", t)
    return t.replace("{", "").replace("}", "").strip()


class SrtError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def cues_match_manifest(cues: list[Cue], manifest_segments: list[dict], tolerance_s: float = 0.002) -> list[str]:
    """
    Aturan §6-1 dari sisi runner: jumlah + timestamp SRT harus identik dengan
    manifest (yang dikirim app bersama SRT). Mengembalikan daftar masalah
    (kosong = cocok). Runner menolak job bila tidak cocok — app dan runner
    harus melihat kenyataan yang sama.
    """
    problems: list[str] = []
    if len(cues) != len(manifest_segments):
        problems.append(f"jumlah segmen berbeda: srt={len(cues)} manifest={len(manifest_segments)}")
        return problems
    for cue, seg in zip(cues, manifest_segments, strict=True):
        if int(seg.get("idx", -1)) != cue.idx:
            problems.append(f"urutan idx berbeda pada posisi {cue.idx}")
        if abs(float(seg.get("start", -1)) - cue.start) > tolerance_s:
            problems.append(f"start berbeda pada cue {cue.idx}: {cue.start} vs {seg.get('start')}")
        if abs(float(seg.get("end", -1)) - cue.end) > tolerance_s:
            problems.append(f"end berbeda pada cue {cue.idx}: {cue.end} vs {seg.get('end')}")
    return problems


def validate_srt_size(srt: str) -> None:
    if len(srt.encode("utf-8")) > MAX_SRT_BYTES:
        raise SrtError("srt_too_large", f"SRT melebihi {MAX_SRT_BYTES} byte")
