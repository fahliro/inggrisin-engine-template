"""
Konfigurasi engine v2 (DUBBING), seluruhnya dari environment variable.

Kenapa env, bukan argumen CLI?
  - Secret (GROQ_API_KEY, INGGRISIN_CALLBACK_SECRET) tidak boleh muncul di
    `ps`/command line. Env var di GitHub Actions otomatis di-mask bila
    nilainya berasal dari `secrets`.
  - Workflow memetakan `inputs` → `env` di step "Run engine".

Perubahan v2 (docs/00 §5–§6):
  - mode: 'first' | 'revision'
  - mode revision TIDAK membutuhkan GROQ_API_KEY maupun OpenRouter key —
    pipeline revisi tidak memanggil keduanya (docs/01 §20). Kalau tetap
    disetel, tidak apa-apa; yang penting tidak wajib.
  - `options` divalidasi terhadap konstanta kontrak yang TERKUNCI — engine
    menolak job bila app mengirim nilai berbeda (proteksi drift dua arah).
  - drive_folder_id + dubbed_file_name wajib (folder per-video dibuat app
    sebelum dispatch).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .tts import VOICES

# Konstanta kontrak — HARUS identik dengan LOCKED di packages/shared/contracts.ts.
# CATATAN: `music` TIDAK lagi dikunci (docs/01 §27): mempertahankan BGM/SFX adalah
# kemampuan opsional — engine menerima true/false dari app (default true) dan
# selalu fallback mulus ke audio bersih bila pemisahan vokal gagal.
LOCKED_OPTIONS = {
    "fit_mode": "blur",
    "timing": "strict",
    "tempo_max": 1.25,
    "subtitles": "burn",
    "subtitle_bottom_pct": 19,
}
LOCKED_VOICE = "en_US-lessac-medium"  # default; pilihan lain = katalog tts.VOICES


class ConfigError(Exception):
    """Kesalahan konfigurasi yang bisa dilaporkan balik ke app."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(slots=True)
class Config:
    # identitas
    job_id: str
    idempotency_key: str
    contract_version: str
    engine_version: str
    mode: str  # 'first' | 'revision'

    # revisi (mode revision)
    revision_number: int
    revisions_url: str
    result_drive_file_id: str

    # sumber (kedua mode: proxy sekali-pakai; revision → video asli di Drive)
    source_url: str
    source_name: str
    source_bytes: int
    source_sha256: str

    # callback
    callback_url: str
    callback_secret: str

    # model (keputusan sistem)
    whisper_model: str
    llm_model: str
    voice: str

    # keluaran Drive
    drive_folder_id: str
    dubbed_file_name: str
    result_upload_url: str

    # BYOK (wajib hanya untuk mode first)
    groq_api_key: str
    openrouter_api_key: str

    # runtime
    workspace: Path
    out_dir: Path
    report_path: Path
    github_run_id: str
    github_run_url: str

    options: dict = field(default_factory=dict)
    output_aspect: str = "9:16"

    @staticmethod
    def load(workspace: Path, out_dir: Path, report_path: Path) -> Config:
        def req(name: str) -> str:
            v = os.environ.get(name, "").strip()
            if not v:
                raise ConfigError("missing_env", f"Environment variable {name} is not set")
            return v

        def opt(name: str, default: str = "") -> str:
            return os.environ.get(name, default).strip()

        mode = (opt("INGGRISIN_MODE", "first") or "first").lower()
        if mode not in ("first", "revision"):
            raise ConfigError("contract_mismatch", f"Unknown mode: {mode!r} (first|revision)")

        # options = konstanta kontrak terkunci; tolak bila app mengirim lain
        try:
            raw_opts = json.loads(opt("INGGRISIN_OPTIONS", "{}") or "{}")
            if not isinstance(raw_opts, dict):
                raw_opts = {}
        except json.JSONDecodeError:
            raw_opts = {}
        for key, expect in LOCKED_OPTIONS.items():
            got = raw_opts.get(key, expect)
            if isinstance(expect, float):
                same = abs(float(got) - expect) < 1e-9
            else:
                same = got == expect
            if not same:
                raise ConfigError(
                    "contract_mismatch",
                    f"options.{key}={got!r} violates the locked contract ({expect!r}) — "
                    "dubbing parameters are not user input (docs/01 §18)",
                )

        voice = opt("INGGRISIN_VOICE", LOCKED_VOICE)
        # suara = pilihan user dari katalog Piper (docs/01 §27, menggantikan
        # kunci satu-suara §21). Di luar katalog → tolak (proteksi dua arah:
        # app juga memvalidasi terhadap VOICES di @inggrisin/shared).
        if voice not in VOICES:
            raise ConfigError(
                "contract_mismatch",
                f"unknown voice={voice!r}; choices: {', '.join(sorted(VOICES))}",
            )

        output_aspect = (opt("INGGRISIN_OUTPUT_ASPECT", "9:16") or "9:16").strip()
        if output_aspect not in ("9:16", "source"):
            raise ConfigError(
                "contract_mismatch",
                f"unknown output_aspect={output_aspect!r}; choices: 9:16, source",
            )

        cfg = Config(
            job_id=req("INGGRISIN_JOB_ID"),
            idempotency_key=opt("INGGRISIN_IDEMPOTENCY_KEY") or req("INGGRISIN_JOB_ID"),
            contract_version=opt("INGGRISIN_CONTRACT_VERSION", "unknown"),
            engine_version=opt("INGGRISIN_ENGINE_VERSION", "unknown"),
            mode=mode,
            revision_number=int(opt("INGGRISIN_REVISION_NUMBER", "0") or 0),
            revisions_url=opt("INGGRISIN_REVISIONS_URL"),
            result_drive_file_id=opt("INGGRISIN_RESULT_DRIVE_FILE_ID"),
            source_url=req("INGGRISIN_SOURCE_URL"),
            source_name=opt("INGGRISIN_SOURCE_NAME", "source.bin"),
            source_bytes=int(opt("INGGRISIN_SOURCE_BYTES", "0") or 0),
            source_sha256=opt("INGGRISIN_SOURCE_SHA256"),
            callback_url=req("INGGRISIN_CALLBACK_URL"),
            callback_secret=req("INGGRISIN_CALLBACK_SECRET"),
            # §29: v3 penuh (BUKAN turbo) — hanya v3 yang mendukung endpoint
            # /audio/translations di Groq; biaya tetap receh ($0.111/jam).
            whisper_model=opt("INGGRISIN_WHISPER_MODEL", "whisper-large-v3"),
            llm_model=opt("INGGRISIN_LLM_MODEL", "openrouter/free"),
            voice=voice,
            output_aspect=output_aspect,
            drive_folder_id=req("INGGRISIN_DRIVE_FOLDER_ID"),
            dubbed_file_name=req("INGGRISIN_DUBBED_FILE_NAME"),
            result_upload_url=opt("INGGRISIN_RESULT_UPLOAD_URL"),
            groq_api_key=opt("GROQ_API_KEY"),
            openrouter_api_key=opt("INGGRISIN_OPENROUTER_API_KEY") or opt("OPENROUTER_API_KEY"),
            workspace=workspace,
            out_dir=out_dir,
            report_path=report_path,
            github_run_id=opt("GITHUB_RUN_ID_OUT") or opt("GITHUB_RUN_ID", ""),
            github_run_url=opt("GITHUB_RUN_URL_OUT", ""),
            # music = kemampuan (bukan kunci kontrak): app boleh mengirim
            # false; default true = BGM/SFX asli dipertahankan (docs/01 §27).
            options=dict(LOCKED_OPTIONS, music=bool(raw_opts.get("music", True))),
        )

        # validasi per mode
        if cfg.mode == "first":
            if not cfg.groq_api_key:
                raise ConfigError("missing_env", "GROQ_API_KEY is required for first mode (Whisper)")
            # §29: OpenRouter TIDAK lagi dipakai build pertama (whisper translate
            # langsung). Key tetap dibaca bila ada — tidak wajib.
        else:
            if cfg.revision_number <= 0:
                raise ConfigError(
                    "missing_env", "INGGRISIN_REVISION_NUMBER must be > 0 for revision mode"
                )
            if not cfg.revisions_url:
                raise ConfigError(
                    "missing_env", "INGGRISIN_REVISIONS_URL is required for revision mode"
                )
            if not cfg.result_drive_file_id:
                raise ConfigError(
                    "missing_env",
                    "INGGRISIN_RESULT_DRIVE_FILE_ID is required for revision mode (the file being overwritten)",
                )
            # revision TANPA Groq/LLM — pastikan tidak ada yang mengira perlu
        workspace.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        return cfg

    # ------------------------------------------------------------------ paths
    @property
    def source_path(self) -> Path:
        return self.workspace / "source.bin"

    @property
    def audio_path(self) -> Path:
        return self.workspace / "audio.opus"

    @property
    def chunks_dir(self) -> Path:
        return self.workspace / "chunks"

    @property
    def lines_dir(self) -> Path:
        return self.workspace / "lines"

    @property
    def tempo_dir(self) -> Path:
        return self.workspace / "tempo"

    @property
    def voice_track_path(self) -> Path:
        return self.workspace / "voice.wav"

    @property
    def music_dir(self) -> Path:
        return self.workspace / "music"

    @property
    def accompaniment_path(self) -> Path:
        return self.music_dir / "accompaniment.wav"

    @property
    def final_audio_path(self) -> Path:
        return self.workspace / "final_audio.wav"

    @property
    def ass_path(self) -> Path:
        return self.out_dir / "subtitles.ass"

    @property
    def srt_path(self) -> Path:
        return self.out_dir / "revision.srt"

    @property
    def final_video_path(self) -> Path:
        return self.out_dir / self.dubbed_file_name.replace("/", "_")

    @property
    def transcript_path(self) -> Path:
        return self.out_dir / "transcript.json"

    @property
    def manifest_path(self) -> Path:
        return self.out_dir / "manifest.json"

    @property
    def log_path(self) -> Path:
        return self.workspace / "engine.log"
