"""
Test engine v2 (dubbing) — semuanya offline: tanpa ffmpeg, tanpa jaringan.
Yang diuji adalah fungsi murni (pembangun command, parser, validator) dan
logika orkestrasi dengan HTTP di-mock.

Paritas dengan packages/shared: nilai budget & vektor SRT HARUS identik
dengan contracts.test.ts (CPS=15, TEMPO_MAX=1.25, lantai 12).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from pathlib import Path

import pytest

from engine import audio, callback, srt, translate, vertical
from engine import config as config_mod
from engine import main as main_mod
from engine import transcribe as transcribe_mod
from engine.log import Logger

LOG = Logger(None)


# ---------------------------------------------------------------------------
# srt.py — paritas dengan subtitles.ts
# ---------------------------------------------------------------------------


def test_parity_budget_dengan_ts():
    # vektor identik dengan contracts.test.ts
    assert srt.budget_of(0, 6) == 112
    assert srt.budget_of(15, 24) == 168
    assert srt.budget_of(0, 0.2) == 12


def test_srt_roundtrip():
    cues = [
        srt.Cue(0, 0.0, 6.0, "Hi everyone, welcome back to my kitchen."),
        srt.Cue(
            1, 6.0, 15.0, "Today we're cooking authentic Padang rendang — a recipe passed down."
        ),
        srt.Cue(2, 15.0, 22.0, "Just two main ingredients: beef and coconut milk. 🎉"),
    ]
    text = srt.serialize_srt(cues)
    parsed = srt.parse_srt(text)
    assert len(parsed) == 3
    for a, b in zip(cues, parsed, strict=True):
        assert a.text == b.text
        assert abs(a.start - b.start) < 0.0011
        assert abs(a.end - b.end) < 0.0011


def test_parse_srt_tanpa_nomor():
    text = "00:00:01,000 --> 00:00:02,000\nHalo\n\n00:00:03,000 --> 00:00:04,000\nDunia\n"
    parsed = srt.parse_srt(text)
    assert len(parsed) == 2
    assert parsed[1].text == "Dunia"


def test_sanitize_buang_tag_ass_dan_kontrol():
    assert srt.sanitize_for_burn("Hi {\\an8}everyone") == "Hi everyone"
    assert srt.sanitize_for_burn("a\x07b\x00c") == "abc"
    assert srt.sanitize_for_burn("sisa { kurung }") == "sisa"  # seluruh blok {..} dibuang


def test_cues_match_manifest():
    cues = [srt.Cue(0, 0.0, 6.0, "a"), srt.Cue(1, 6.0, 15.0, "b")]
    man = [{"idx": 0, "start": 0.0, "end": 6.0}, {"idx": 1, "start": 6.0, "end": 15.0}]
    assert srt.cues_match_manifest(cues, man) == []
    # jumlah beda
    assert srt.cues_match_manifest(cues, man[:1])
    # start bergeser
    bad = [{"idx": 0, "start": 0.5, "end": 6.0}, {"idx": 1, "start": 6.0, "end": 15.0}]
    problems = srt.cues_match_manifest(cues, bad)
    assert problems and "start" in problems[0]


def test_validate_srt_size():
    with pytest.raises(srt.SrtError):
        srt.validate_srt_size("x" * (srt.MAX_SRT_BYTES + 1))


# ---------------------------------------------------------------------------
# vertical.py — SATU pass, tidak pernah -c:v copy
# ---------------------------------------------------------------------------


def test_classify_aspect():
    assert vertical.classify_aspect(1920, 1080) == "horizontal"
    assert vertical.classify_aspect(1080, 1920) == "vertical"
    assert vertical.classify_aspect(1000, 1000) == "square"
    assert vertical.classify_aspect(1020, 1000) == "square"  # rasio 1.02


def test_ass_margin_di_atas_safe_zone(tmp_path):
    cues = [srt.Cue(0, 0.0, 6.0, "Hi {\\an8}there")]
    out = vertical.build_ass(cues, tmp_path / "s.ass")
    text = out.read_text(encoding="utf-8")
    assert "PlayResY: 1920" in text
    # MarginV = 19% × 1920 = 365 (kolom ke-22 pada Style)
    style = next(ln for ln in text.splitlines() if ln.startswith("Style:"))
    fields = [f.strip() for f in style.split(",")]
    assert fields[-2] == "365", fields
    # teks dialogue sudah disanitasi dari tag ASS
    assert "{\\an8}" not in text
    assert "Hi there" in text


def test_ass_time_format():
    assert vertical.ass_time(0) == "0:00:00.00"
    assert vertical.ass_time(61.55) == "0:01:01.55"
    assert vertical.ass_time(3600 + 2 * 60 + 3.5) == "1:02:03.50"


def _render_cmd(aspect):
    return vertical.build_render_cmd(
        Path("src.mp4"),
        Path("voice.wav"),
        Path("sub.ass"),
        Path("out.mp4"),
        aspect,
        duration_s=100.0,
    )


def test_render_horizontal_satu_pass_blur():
    cmd = _render_cmd("horizontal")
    joined = " ".join(cmd)
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "gblur" in fc and "overlay" in fc and "subtitles=" in fc
    assert joined.count("libx264") == 1  # SATU encode video
    assert "-c:v copy" not in joined  # tidak pernah copy
    assert joined.count("subtitles=") == 1  # burn sekali, di pass yang sama
    assert "-t" in cmd and "100.000" in cmd  # durasi dibatasi
    assert "1:a:0" in joined  # audio = voice track


def test_render_vertical_tanpa_blur():
    cmd = _render_cmd("vertical")
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "gblur" not in fc
    assert "pad=1080:1920" in fc
    assert "subtitles=" in fc


def test_escape_filter_path():
    assert "\\:" in vertical._escape_filter_path(Path("/tmp/a:b/x.ass"))


# ---------------------------------------------------------------------------
# audio.py — rencana voice track (fungsi murni)
# ---------------------------------------------------------------------------


def test_plan_voice_track_gap_dan_tail():
    plan = audio.plan_voice_track([(2.0, 3.0), (10.0, 2.0)], total_s=15.0)
    kinds = [k for k, _ in plan]
    assert kinds == ["gap", "seg", "gap", "seg", "gap"]
    assert plan[0] == ("gap", 2.0)
    assert plan[2] == ("gap", 5.0)
    assert plan[4] == ("gap", 3.0)


def test_plan_voice_track_overlap_trim():
    plan = audio.plan_voice_track([(0.0, 8.0), (5.0, 2.0)], total_s=12.0)
    assert ("trim", 3.0) in plan


# ---------------------------------------------------------------------------
# translate.py — LLM di-mock
# ---------------------------------------------------------------------------


def _segs():
    return [
        translate.InSegment(0, 0.0, 6.0, "Halo semuanya"),
        translate.InSegment(1, 6.0, 15.0, "Hari ini kita memasak rendang"),
    ]


def _fake_chat(responses):
    calls = []

    def fake(log, api_key, model, system, user, json_mode, tag):
        calls.append(tag)
        return responses[len(calls) - 1], {"prompt_tokens": 10, "completion_tokens": 5}

    return fake, calls


def test_translate_happy_path(monkeypatch):
    resp = json.dumps(
        {
            "segments": [
                {"idx": 0, "text_en": "Hi everyone"},
                {"idx": 1, "text_en": "Today we cook rendang"},
            ],
            "terms": [{"from": "rendang", "to": "rendang"}],
        }
    )
    fake, calls = _fake_chat([resp])
    monkeypatch.setattr(translate, "_chat", fake)
    res = translate.translate(LOG, "k", _segs(), "m")
    assert [s.text_en for s in res.segments] == ["Hi everyone", "Today we cook rendang"]
    assert all(s.fit == "ok" and s.tempo == 1.0 for s in res.segments)
    assert res.terms == [{"from": "rendang", "to": "rendang"}]
    assert res.tokens_prompt == 10 and res.tokens_completion == 5
    assert calls == ["translate"]


def test_translate_missing_idx_gagal(monkeypatch):
    resp = json.dumps({"segments": [{"idx": 0, "text_en": "Hi"}]})
    fake, _ = _fake_chat([resp])
    monkeypatch.setattr(translate, "_chat", fake)
    with pytest.raises(translate.TranslateError) as e:
        translate.translate(LOG, "k", _segs(), "m")
    assert e.value.code == "llm_bad_json"


def test_translate_over_budget_memicu_repair(monkeypatch):
    panjang = "word " * 60  # 300 karakter > budget slot 6 dtk (112)
    resp1 = json.dumps(
        {
            "segments": [
                {"idx": 0, "text_en": panjang.strip()},
                {"idx": 1, "text_en": "Today we cook rendang"},
            ]
        }
    )
    resp2 = json.dumps({"segments": [{"idx": 0, "text_en": "Hi everyone, welcome back"}]})
    fake, calls = _fake_chat([resp1, resp2])
    monkeypatch.setattr(translate, "_chat", fake)
    res = translate.translate(LOG, "k", _segs(), "m")
    assert calls == ["translate", "translate-condense"]
    assert res.segments[0].text_en == "Hi everyone, welcome back"


def test_translate_tempo_dihitung_deterministik(monkeypatch):
    # slot 6 dtk: natural ≤ 90 char; budget 112. 100 char → est 1.11 → fit tempo.
    teks = "x" * 100
    resp1 = json.dumps(
        {"segments": [{"idx": 0, "text_en": teks}, {"idx": 1, "text_en": "ok text"}]}
    )
    resp2 = json.dumps({"segments": [{"idx": 0, "text_en": teks}]})  # repair gagal merigkas
    fake, _ = _fake_chat([resp1, resp2])
    monkeypatch.setattr(translate, "_chat", fake)
    res = translate.translate(LOG, "k", _segs(), "m")
    s0 = res.segments[0]
    assert s0.fit == "tempo" and 1.0 < s0.tempo <= 1.25


def test_translate_lewat_budget_total_clip_125(monkeypatch):
    teks = "y" * 150  # > budget 112, repair tetap panjang
    resp = json.dumps({"segments": [{"idx": 0, "text_en": teks}, {"idx": 1, "text_en": "fine"}]})
    fake, _ = _fake_chat([resp, resp])
    monkeypatch.setattr(translate, "_chat", fake)
    res = translate.translate(LOG, "k", _segs(), "m")
    s0 = res.segments[0]
    assert s0.tempo == 1.25 and "dipercepat maksimal" in s0.note


# ---------------------------------------------------------------------------
# translate.py — _chat: cabang error HTTP OpenRouter (body + petunjuk)
# ---------------------------------------------------------------------------


class _HttpResp:
    """Response palsu minimal untuk menguji cabang error HTTP _chat."""

    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text else (json.dumps(payload) if payload is not None else "")

    def json(self):
        if self._payload is None:
            raise ValueError("bukan JSON")
        return self._payload


def test_chat_404_data_policy_menyertakan_body_dan_petunjuk(monkeypatch):
    resp = _HttpResp(404, {"error": {"message": "No endpoints found matching your data policy"}})
    monkeypatch.setattr(translate.requests, "post", lambda *a, **k: resp)
    with pytest.raises(translate.TranslateError) as e:
        translate._chat(LOG, "k", "openrouter/free", "s", "u", False, "t")
    assert e.value.code == "openrouter_failed"
    assert "No endpoints found matching your data policy" in e.value.message
    assert "openrouter.ai/settings/privacy" in e.value.message


def test_chat_404_model_paid_tanpa_petunjuk_privasi(monkeypatch):
    resp = _HttpResp(404, {"error": {"message": "Model not found"}})
    monkeypatch.setattr(translate.requests, "post", lambda *a, **k: resp)
    with pytest.raises(translate.TranslateError) as e:
        translate._chat(LOG, "k", "openai/gpt-4o-mini", "s", "u", False, "t")
    assert e.value.code == "openrouter_failed"
    assert "Model not found" in e.value.message
    assert "settings/privacy" not in e.value.message


def test_chat_402_menyertakan_body(monkeypatch):
    resp = _HttpResp(
        402, {"error": {"message": "Insufficient credits — buy more at openrouter.ai"}}
    )
    monkeypatch.setattr(translate.requests, "post", lambda *a, **k: resp)
    with pytest.raises(translate.TranslateError) as e:
        translate._chat(LOG, "k", "openai/gpt-4o-mini", "s", "u", False, "t")
    assert e.value.code == "openrouter_no_credit"
    assert "Insufficient credits" in e.value.message


def test_chat_body_bukan_json_tetap_aman(monkeypatch):
    resp = _HttpResp(400, None, text="<html>Bad Request</html>")
    monkeypatch.setattr(translate.requests, "post", lambda *a, **k: resp)
    with pytest.raises(translate.TranslateError) as e:
        translate._chat(LOG, "k", "m", "s", "u", False, "t")
    assert e.value.code == "openrouter_failed"
    assert "Bad Request" in e.value.message


def test_chat_matikan_reasoning_dan_json_mode(monkeypatch):
    captured = {}

    class R:
        status_code = 200

        def json(self):
            return {
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {},
            }

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(json)
        return R()

    monkeypatch.setattr(translate.requests, "post", fake_post)
    translate._chat(LOG, "k", "openrouter/free", "s", "u", True, "t")
    assert captured["reasoning"] == {"enabled": False}
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["max_tokens"] == 16000


def test_chat_reasoning_wajib_diulang_tanpa_param(monkeypatch):
    """Endpoint yang MEWAJIBKAN reasoning (400) → engine buang param & ulang."""
    calls = []
    msg_400 = "Reasoning is mandatory for this endpoint and cannot be disabled."

    class R400:
        status_code = 400
        text = json.dumps({"error": {"message": msg_400}})

        def json(self):
            return {"error": {"message": msg_400}}

    class R200:
        status_code = 200

        def json(self):
            return {
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {},
            }

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(dict(json))
        return R400() if "reasoning" in json else R200()

    monkeypatch.setattr(translate.requests, "post", fake_post)
    content, _ = translate._chat(LOG, "k", "openrouter/free", "s", "u", True, "t")
    assert content == "{}"
    assert len(calls) == 2
    assert "reasoning" not in calls[1]


def test_chat_finish_reason_length_diwarnakan(monkeypatch, capsys):
    class R:
        status_code = 200

        def json(self):
            return {
                "choices": [{"message": {"content": "{}"}, "finish_reason": "length"}],
                "usage": {},
            }

    monkeypatch.setattr(translate.requests, "post", lambda *a, **k: R())
    translate._chat(LOG, "k", "m", "s", "u", True, "t")
    assert "finish_reason=length" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# config.py — mode & konstanta terkunci
# ---------------------------------------------------------------------------


def _env_base(monkeypatch, tmp_path, **over):
    base = {
        "INGGRISIN_JOB_ID": "job_12345678",
        "INGGRISIN_SOURCE_URL": "https://app.example.com/dl/x",
        "INGGRISIN_SOURCE_NAME": "v.mp4",
        "INGGRISIN_CALLBACK_URL": "https://app.example.com/cb",
        "INGGRISIN_CALLBACK_SECRET": "s" * 32,
        "INGGRISIN_DRIVE_FOLDER_ID": "folder1",
        "INGGRISIN_DUBBED_FILE_NAME": "v (EN).mp4",
    }
    base.update({k: str(v) for k, v in over.items()})
    for k, v in base.items():
        monkeypatch.setenv(k, v)
    for k in (
        "INGGRISIN_MODE",
        "INGGRISIN_REVISION_NUMBER",
        "INGGRISIN_REVISIONS_URL",
        "INGGRISIN_RESULT_DRIVE_FILE_ID",
        "INGGRISIN_OPTIONS",
        "INGGRISIN_VOICE",
        "GROQ_API_KEY",
        "OPENROUTER_API_KEY",
    ):
        if k not in base:
            monkeypatch.delenv(k, raising=False)
    return tmp_path


def test_config_first_butuh_groq_saja(monkeypatch, tmp_path):
    # §29: OpenRouter TIDAK lagi wajib di mode first (whisper translate langsung)
    _env_base(monkeypatch, tmp_path)
    with pytest.raises(config_mod.ConfigError):
        config_mod.Config.load(tmp_path / "w", tmp_path / "o", tmp_path / "r.json")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_x")
    cfg = config_mod.Config.load(tmp_path / "w", tmp_path / "o", tmp_path / "r.json")
    assert cfg.mode == "first"
    assert cfg.whisper_model == "whisper-large-v3"  # v3 penuh: dukung /audio/translations


def test_config_revision_tanpa_groq_llm(monkeypatch, tmp_path):
    _env_base(
        monkeypatch,
        tmp_path,
        INGGRISIN_MODE="revision",
        INGGRISIN_REVISION_NUMBER="2",
        INGGRISIN_REVISIONS_URL="https://app.example.com/rev?token=t",
        INGGRISIN_RESULT_DRIVE_FILE_ID="oldfile",
    )
    cfg = config_mod.Config.load(tmp_path / "w", tmp_path / "o", tmp_path / "r.json")
    assert cfg.mode == "revision" and cfg.revision_number == 2
    assert cfg.groq_api_key == ""  # memang tidak diisi, dan tidak ditolak


def test_config_revision_butuh_field_revisi(monkeypatch, tmp_path):
    _env_base(monkeypatch, tmp_path, INGGRISIN_MODE="revision", INGGRISIN_REVISION_NUMBER="1")
    with pytest.raises(config_mod.ConfigError):
        config_mod.Config.load(tmp_path / "w", tmp_path / "o", tmp_path / "r.json")


def test_config_options_terkunci(monkeypatch, tmp_path):
    _env_base(
        monkeypatch,
        tmp_path,
        GROQ_API_KEY="g",
        OPENROUTER_API_KEY="o",
        INGGRISIN_OPTIONS=json.dumps({"fit_mode": "crop"}),
    )
    with pytest.raises(config_mod.ConfigError) as e:
        config_mod.Config.load(tmp_path / "w", tmp_path / "o", tmp_path / "r.json")
    assert e.value.code == "contract_mismatch"


def test_config_voice_katalog_diterima(monkeypatch, tmp_path):
    # docs/01 §27: suara = pilihan user dari katalog Piper (bukan kunci mati)
    _env_base(
        monkeypatch,
        tmp_path,
        GROQ_API_KEY="g",
        OPENROUTER_API_KEY="o",
        INGGRISIN_VOICE="en_US-ryan-high",
    )
    cfg = config_mod.Config.load(tmp_path / "w", tmp_path / "o", tmp_path / "r.json")
    assert cfg.voice == "en_US-ryan-high"


def test_config_voice_diluar_katalog_ditolak(monkeypatch, tmp_path):
    _env_base(
        monkeypatch,
        tmp_path,
        GROQ_API_KEY="g",
        OPENROUTER_API_KEY="o",
        INGGRISIN_VOICE="id_ID-penipu-medium",
    )
    with pytest.raises(config_mod.ConfigError) as e:
        config_mod.Config.load(tmp_path / "w", tmp_path / "o", tmp_path / "r.json")
    assert "tidak dikenal" in e.value.message


def test_config_music_default_on_dan_bisa_dimatikan(monkeypatch, tmp_path):
    _env_base(monkeypatch, tmp_path, GROQ_API_KEY="g", OPENROUTER_API_KEY="o")
    cfg = config_mod.Config.load(tmp_path / "w", tmp_path / "o", tmp_path / "r.json")
    assert cfg.options["music"] is True
    _env_base(
        monkeypatch,
        tmp_path,
        GROQ_API_KEY="g",
        OPENROUTER_API_KEY="o",
        INGGRISIN_OPTIONS='{"music": false}',
    )
    cfg2 = config_mod.Config.load(tmp_path / "w2", tmp_path / "o2", tmp_path / "r2.json")
    assert cfg2.options["music"] is False


# ---------------------------------------------------------------------------
# audio.py — pemisahan vokal (demucs) + mix BGM (docs/01 §27)
# ---------------------------------------------------------------------------


def test_separate_accompaniment_tanpa_demucs(monkeypatch, tmp_path):
    monkeypatch.setattr(audio, "demucs_available", lambda: False)
    res = audio.separate_accompaniment(
        LOG, tmp_path / "src.mp4", tmp_path / "work", tmp_path / "a.wav"
    )
    assert res is None


def test_separate_accompaniment_sukses(monkeypatch, tmp_path):
    monkeypatch.setattr(audio, "demucs_available", lambda: True)
    calls = []

    def fake_run(log, cmd, step):
        calls.append(cmd)
        if "demucs" in cmd:
            stem = tmp_path / "work" / audio.DEMUCS_MODEL / "src_audio" / "no_vocals.wav"
            stem.parent.mkdir(parents=True, exist_ok=True)
            stem.write_bytes(b"x" * 4096)
        return None

    monkeypatch.setattr(audio, "_run", fake_run)
    out = tmp_path / "acc.wav"
    res = audio.separate_accompaniment(LOG, tmp_path / "src.mp4", tmp_path / "work", out)
    assert res == out and out.exists()
    assert any("demucs" in c for c in calls)


def test_separate_accompaniment_gagal_graceful(monkeypatch, tmp_path):
    monkeypatch.setattr(audio, "demucs_available", lambda: True)

    def boom(log, cmd, step):
        raise audio.AudioError("ffmpeg_failed", "pura-pura rusak")

    monkeypatch.setattr(audio, "_run", boom)
    res = audio.separate_accompaniment(LOG, tmp_path / "s.mp4", tmp_path / "w", tmp_path / "o.wav")
    assert res is None  # TIDAK melempar — pemanggil fallback ke dub bersih


def test_mix_voice_with_music_perintah(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(audio, "_run", lambda log, cmd, step: calls.append(cmd))
    monkeypatch.setattr(audio, "loudnorm", lambda log, src, dst, target_i=-16.0: dst)
    v = tmp_path / "voice.wav"
    v.write_bytes(b"1")
    m = tmp_path / "music.wav"
    m.write_bytes(b"1")
    audio.mix_voice_with_music(LOG, v, m, 12.0, tmp_path / "final.wav")
    joined = " ".join(calls[0])
    assert "amix=inputs=2:duration=first:normalize=0" in joined
    assert "volume=0.600" in joined
    assert "atrim=0:12.000" in joined


# ---------------------------------------------------------------------------
# callback + payload
# ---------------------------------------------------------------------------


def test_callback_sign_vektor():
    sig = callback.sign("secret", "job1", b'{"a":1}')
    expect = "sha256=" + hmac.new(b"secret", b"job1." + b'{"a":1}', hashlib.sha256).hexdigest()
    assert sig == expect


def test_subtitle_dicts_camelcase():
    rows = [
        {
            "idx": 0,
            "start": 0.0,
            "end": 6.0,
            "text_id": "Halo",
            "text_en": "Hi",
            "fit": "ok",
            "tempo": 1.0,
            "note": "",
        }
    ]
    out = main_mod._subtitle_dicts(rows)
    assert out[0]["textEn"] == "Hi" and out[0]["textId"] == "Halo"
    assert "note" not in out[0]  # note kosong dihilangkan


def test_done_payload_revision_number(monkeypatch, tmp_path):
    _env_base(
        monkeypatch,
        tmp_path,
        INGGRISIN_MODE="revision",
        INGGRISIN_REVISION_NUMBER="3",
        INGGRISIN_REVISIONS_URL="https://x.co/r",
        INGGRISIN_RESULT_DRIVE_FILE_ID="f",
    )
    cfg = config_mod.Config.load(tmp_path / "w", tmp_path / "o", tmp_path / "r.json")
    p = main_mod._done_payload(cfg, {"drive_file_id": "f"})
    assert p["status"] == "done" and p["revision_number"] == 3


def test_fetch_revision_validasi(monkeypatch, tmp_path):
    _env_base(
        monkeypatch,
        tmp_path,
        INGGRISIN_MODE="revision",
        INGGRISIN_REVISION_NUMBER="1",
        INGGRISIN_REVISIONS_URL="https://x.co/r",
        INGGRISIN_RESULT_DRIVE_FILE_ID="f",
    )
    cfg = config_mod.Config.load(tmp_path / "w", tmp_path / "o", tmp_path / "r.json")

    class Resp:
        status_code = 200

        def __init__(self, data):
            self._d = data

        def json(self):
            return self._d

    good = {
        "job_id": cfg.job_id,
        "revision_number": 1,
        "srt": "1\n00:00:00,000 --> 00:00:02,000\nHi\n",
        "manifest": {"segments": [{"idx": 0, "start": 0.0, "end": 2.0}]},
    }
    monkeypatch.setattr(main_mod.requests, "get", lambda *a, **k: Resp(good))
    assert main_mod._fetch_revision(cfg, LOG)["srt"].startswith("1\n")

    bad_job = dict(good, job_id="job_lain")
    monkeypatch.setattr(main_mod.requests, "get", lambda *a, **k: Resp(bad_job))
    with pytest.raises(RuntimeError, match="job_id"):
        main_mod._fetch_revision(cfg, LOG)

    bad_rev = dict(good, revision_number=9)
    monkeypatch.setattr(main_mod.requests, "get", lambda *a, **k: Resp(bad_rev))
    with pytest.raises(RuntimeError, match="revision_number"):
        main_mod._fetch_revision(cfg, LOG)


# ---------------------------------------------------------------------------
# tts.py — _synth_one kompatibel piper-tts 1.2 (synthesize) & 1.3+ (synthesize_wav)
# ---------------------------------------------------------------------------


def test_synth_one_piper_12_pakai_synthesize(tmp_path):
    """piper-tts 1.2.0 (versi di-pin) cuma punya synthesize(text, wav_file)."""
    from engine import tts

    written = {}

    class FakeWav:
        def setnchannels(self, n):
            written["ch"] = n

        def setsampwidth(self, n):
            written["sw"] = n

        def setframerate(self, n):
            written["fr"] = n

        def writeframes(self, b):
            written["frames"] = len(b)

    class Voice12:
        def synthesize(self, text, wav_file, syn_config=None):
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)
            wav_file.writeframes(b"\x00\x00" * 22050)

    dst = tmp_path / "line_00000.wav"
    # wave.open asli dipakai; pastikan file WAV valid dihasilkan
    tts._synth_one(Voice12(), "hello world", dst)
    assert dst.exists() and dst.stat().st_size > 1000
    import wave

    with wave.open(str(dst), "rb") as w:
        assert w.getnchannels() == 1 and w.getframerate() == 22050


def test_synth_one_piper_13_pakai_synthesize_wav(tmp_path):
    """Bila piper punya synthesize_wav (1.3+), jalur itu yang dipakai."""
    from engine import tts

    called = {}

    class Voice13:
        def synthesize_wav(self, text, fh):
            called["text"] = text
            fh.write(b"RIFF" + b"\x00" * 4000)

    dst = tmp_path / "line_00001.wav"
    tts._synth_one(Voice13(), "hello", dst)
    assert called.get("text") == "hello"
    assert dst.exists() and dst.stat().st_size > 1000


# ---------------------------------------------------------------------------
# upload.py — proxy upload menandatangani `${job_id}.${sha256}` (skema callback)
# ---------------------------------------------------------------------------


def test_upload_proxy_protokol2_chunked_dan_signature(tmp_path, monkeypatch):
    import hashlib
    import hmac as hmac_mod

    from engine import upload as upload_mod

    p = tmp_path / "hasil.mp4"
    p.write_bytes(b"x" * (8 * 1024 * 1024 + 100))  # 2 chunk
    captured: dict = {"puts": []}

    class RInit:
        status_code = 200
        text = ""

        def json(self):
            return {"ok": True, "session": "SESS.BLOB", "chunk_bytes": 8 * 1024 * 1024}

    class R204:
        status_code = 204
        text = ""

    class RFinal:
        status_code = 200
        text = ""

        def json(self):
            return {
                "drive_file_id": "fid123",
                "drive_web_url": "https://drive.google.com/file/d/fid123/view",
                "bytes": 8 * 1024 * 1024 + 100,
            }

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["init"] = {"url": url, "json": json, "headers": headers}
        return RInit()

    def fake_put(url, data=None, headers=None, timeout=None):
        captured["puts"].append((headers["content-range"], len(data)))
        if len(captured["puts"]) == 1:
            return R204()
        return RFinal()

    monkeypatch.setattr(upload_mod.requests, "post", fake_post)
    monkeypatch.setattr(upload_mod.requests, "put", fake_put)
    res = upload_mod.upload_result(
        LOG,
        p,
        proxy_url="https://app.example.com/api/hooks/v1/jobs/j1/result",
        filename="hasil.mp4",
        content_type="video/mp4",
        job_id="j1",
        auth_secret="sekret",  # noqa: S106 — secret dummy utk test
    )
    total = 8 * 1024 * 1024 + 100
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    expect = "sha256=" + hmac_mod.new(b"sekret", f"j1.{sha}".encode(), hashlib.sha256).hexdigest()
    assert captured["init"]["headers"]["x-inggrisin-signature"] == expect
    assert captured["init"]["headers"]["x-inggrisin-protocol"] == "2"
    assert captured["init"]["json"]["total_bytes"] == total
    assert captured["puts"] == [
        (f"bytes 0-{8 * 1024 * 1024 - 1}/{total}", 8 * 1024 * 1024),
        (f"bytes {8 * 1024 * 1024}-{total - 1}/{total}", 100),
    ]
    assert res.drive_file_id == "fid123" and res.mode == "proxy-chunked"


def test_upload_proxy_chunk_resume_setelah_gagal(tmp_path, monkeypatch):
    from engine import upload as upload_mod

    p = tmp_path / "hasil.mp4"
    p.write_bytes(b"y" * (8 * 1024 * 1024 + 100))
    total = 8 * 1024 * 1024 + 100
    state = {"puts": 0}

    class RInit:
        status_code = 200
        text = ""

        def json(self):
            return {"ok": True, "session": "S.B", "chunk_bytes": 8 * 1024 * 1024}

    class R204:
        status_code = 204
        text = ""

    class R500:
        status_code = 500
        text = "boom"

    class RStatus:
        status_code = 200
        text = ""

        def json(self):
            return {"ok": True, "received": 8 * 1024 * 1024}  # chunk-1 sudah masuk

    class RFinal:
        status_code = 200
        text = ""

        def json(self):
            return {"drive_file_id": "f9", "drive_web_url": "u", "bytes": total}

    def fake_put(url, data=None, headers=None, timeout=None):
        rng = headers["content-range"]
        if rng == f"bytes */{total}":
            return RStatus()
        state["puts"] += 1
        if state["puts"] == 2:  # chunk kedua gagal sekali
            return R500()
        return R204() if state["puts"] == 1 else RFinal()

    monkeypatch.setattr(
        upload_mod.requests, "post", lambda url, json=None, headers=None, timeout=None: RInit()
    )
    monkeypatch.setattr(upload_mod.requests, "put", fake_put)
    monkeypatch.setattr(upload_mod.time, "sleep", lambda s: None)
    res = upload_mod.upload_result(
        LOG, p, proxy_url="u", filename="h.mp4", content_type="video/mp4", job_id="j1"
    )
    assert res.drive_file_id == "f9"
    assert res.mode == "proxy-chunked"


# ---------------------------------------------------------------------------
# §29 — plumbing endpoint translations (DORMAN sejak §36: build pertama 1 pass)
# ---------------------------------------------------------------------------


def test_transcribe_file_task_translate_pakai_endpoint_translations(monkeypatch, tmp_path):
    captured = {}

    class R:
        status_code = 200

        def json(self):
            return {
                "text": "Hello there",
                "language": "english",
                "duration": 2.0,
                "segments": [{"start": 0.0, "end": 2.0, "text": "Hello there"}],
            }

    def fake_post(url, headers=None, files=None, data=None, timeout=None):
        captured["url"] = url
        captured["data"] = data
        return R()

    monkeypatch.setattr(transcribe_mod.requests, "post", fake_post)
    p = tmp_path / "audio.wav"
    p.write_bytes(b"x" * 64)
    tr = transcribe_mod.transcribe_file(
        LOG, "gsk_k", p, model="whisper-large-v3", language="id", task="translate"
    )
    assert captured["url"].endswith("/audio/translations")
    assert "language" not in captured["data"]  # endpoint translate tak menerima language
    assert (
        "timestamp_granularities[]" not in captured["data"]
    )  # transcriptions-only (400 bila dikirim)
    assert tr.segments[0].text == "Hello there"


def test_chat_404_provider_error_transient_retry(monkeypatch):
    # openrouter/free: 404 "Provider returned error" = transient → backoff+retry
    calls = []
    resps = [
        _HttpResp(404, {"error": {"message": "Provider returned error"}}),
        _HttpResp(
            200, {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}], "usage": {}}
        ),
    ]

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json.get("model"))
        return resps[len(calls) - 1]

    monkeypatch.setattr(translate.requests, "post", fake_post)
    monkeypatch.setattr(translate.time, "sleep", lambda s: None)
    content, _ = translate._chat(LOG, "k", "openrouter/free", "s", "u", False, "t")
    assert content == "OK"
    assert len(calls) == 2


def test_chat_404_data_policy_tidak_diretry(monkeypatch):
    calls = []
    resp = _HttpResp(404, {"error": {"message": "No endpoints found matching your data policy"}})

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(1)
        return resp

    monkeypatch.setattr(translate.requests, "post", fake_post)
    monkeypatch.setattr(translate.time, "sleep", lambda s: None)
    with pytest.raises(translate.TranslateError) as e:
        translate._chat(LOG, "k", "openrouter/free", "s", "u", False, "t")
    assert len(calls) == 1  # permanen → tanpa retry
    assert "settings/privacy" in e.value.message


def test_translate_json_retry_dengan_backoff(monkeypatch):
    good = json.dumps(
        {
            "segments": [
                {"idx": 0, "text_en": "Hi everyone"},
                {"idx": 1, "text_en": "Today we cook rendang"},
            ],
            "terms": [],
        }
    )
    seq = ["bukan json", "masih bukan json", good]
    calls = []

    def fake_chat(log, api_key, model, system, user, json_mode, tag):
        calls.append(tag)
        return seq[len(calls) - 1], {"prompt_tokens": 1, "completion_tokens": 1}

    monkeypatch.setattr(translate, "_chat", fake_chat)
    monkeypatch.setattr(translate.time, "sleep", lambda s: None)
    res = translate.translate(LOG, "k", _segs(), "m")
    assert calls == ["translate", "translate", "translate"]
    assert res.segments[0].text_en == "Hi everyone"


# ---------------------------------------------------------------------------
# §32 — pemecahan baris panjang (helper _tr dipakai juga oleh guard §36)
# ---------------------------------------------------------------------------
def _tr(segs: list[tuple[float, float, str]]) -> transcribe_mod.Transcript:
    return transcribe_mod.Transcript(
        text=" ".join(t for _, _, t in segs),
        language="id",
        duration_s=segs[-1][1] if segs else 0.0,
        segments=[transcribe_mod.Segment(start=a, end=b, text=t) for a, b, t in segs],
    )


def test_split_long_rows_memecah_dinding_teks():
    wall = (
        "Why are there so many parents here? The children are struggling. "
        "Wow, this is so hard. It turns out that what they need, kindness, is "
        "spreading. Just start and continue."
    )
    rows = [{"idx": 0, "start": 0.0, "end": 10.0, "text_id": "Terima kasih.", "text_en": wall}]
    out = srt.split_long_rows(rows)
    assert len(out) >= 2
    # timing kontigu: start pertama & end terakhir utuh, tanpa celah/overlap
    assert out[0]["start"] == 0.0
    assert out[-1]["end"] == 10.0
    for a, b in zip(out, out[1:], strict=False):
        assert a["end"] == b["start"]
    # idx di-reindex urut
    assert [r["idx"] for r in out] == list(range(len(out)))
    # seluruh kalimat sumber tetap ada (tidak ada yang terbuang)
    joined = " ".join(r["text_en"] for r in out)
    for sentence in ("The children are struggling.", "Just start and continue."):
        assert sentence in joined
    # tiap baris hasil ≤ batas keterbacaan
    for r in out:
        assert len(r["text_en"]) <= srt.MAX_SUB_CHARS


def test_split_long_rows_baris_pendek_tidak_disentuh():
    rows = [
        {
            "idx": 0,
            "start": 0.0,
            "end": 2.0,
            "text_id": "Akhirnya sampai juga.",
            "text_en": "Finally made it.",
        },
        {"idx": 1, "start": 2.0, "end": 4.0, "text_id": "Kasian juga.", "text_en": "Poor thing."},
    ]
    out = srt.split_long_rows(rows)
    assert out == rows


def test_split_long_rows_teks_sumber_yang_panjang_juga_dipecah():
    panjang_id = "Narasi panjang tanpa titik yang harus dipecah " * 4
    rows = [
        {
            "idx": 0,
            "start": 0.0,
            "end": 20.0,
            "text_id": panjang_id.strip(),
            "text_en": "Short line.",
        }
    ]
    out = srt.split_long_rows(rows)
    assert len(out) >= 2
    for r in out:
        assert len(r["text_id"]) <= srt.MAX_SUB_CHARS


def test_groq_llm_translate_segments_sukses(monkeypatch):
    from engine import groq_llm

    class R:
        status_code = 200

        def json(self):
            return {
                "choices": [
                    {"message": {"content": '{"segments":[{"idx":0,"text_en":"Thank you."}]}'}}
                ]
            }

    seen = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        seen["body"] = json
        return R()

    monkeypatch.setattr(groq_llm.requests, "post", fake_post)
    out = groq_llm.translate_segments(LOG, "k", [(0, "Terima kasih.", 187)])
    assert out == {0: "Thank you."}
    assert seen["body"]["model"] == "openai/gpt-oss-120b"
    assert "idx=0 budget=187" in seen["body"]["messages"][1]["content"]


def test_groq_llm_retry_429_lalu_sukses(monkeypatch):
    from engine import groq_llm

    calls = {"n": 0}

    class R429:
        status_code = 429
        text = "rate limit"

    class R200:
        status_code = 200

        def json(self):
            return {
                "choices": [{"message": {"content": '{"segments":[{"idx":1,"text_en":"Hi."}]}'}}]
            }

    def fake_post(url, **kw):
        calls["n"] += 1
        return R429() if calls["n"] == 1 else R200()

    monkeypatch.setattr(groq_llm.requests, "post", fake_post)
    monkeypatch.setattr(groq_llm.time, "sleep", lambda s: None)
    out = groq_llm.translate_segments(LOG, "k", [(1, "Hai.", 50)])
    assert out == {1: "Hi."}
    assert calls["n"] == 2


def test_groq_llm_idx_hilang_gagal(monkeypatch):
    from engine import groq_llm

    class R:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": '{"segments":[{"idx":0,"text_en":"x"}]}'}}]}

    monkeypatch.setattr(groq_llm.requests, "post", lambda url, **kw: R())
    monkeypatch.setattr(groq_llm.time, "sleep", lambda s: None)
    with pytest.raises(groq_llm.GroqLlmError):
        groq_llm.translate_segments(LOG, "k", [(0, "a", 10), (1, "b", 10)])


def test_speech_confidence_rata_rata_dan_none():
    tr = _tr([(0.0, 5.0, "a"), (5.0, 10.0, "b")])
    tr.segments[0].no_speech_prob = 0.2
    tr.segments[1].no_speech_prob = 0.4
    assert transcribe_mod.speech_confidence(tr) == pytest.approx(0.3)
    assert transcribe_mod.speech_confidence(_tr([(0.0, 2.0, "x")])) is None


def test_groq_llm_target_id_back_translate(monkeypatch):
    from engine import groq_llm

    seen = {}

    class R:
        status_code = 200

        def json(self):
            return {
                "choices": [
                    {"message": {"content": '{"segments":[{"idx":0,"text_id":"Terima kasih."}]}'}}
                ]
            }

    def fake_post(url, headers=None, json=None, timeout=None):
        seen["body"] = json
        return R()

    monkeypatch.setattr(groq_llm.requests, "post", fake_post)
    out = groq_llm.translate_segments(LOG, "k", [(0, "Thank you.", 50)], target="id")
    assert out == {0: "Terima kasih."}
    assert "English → Indonesian" in seen["body"]["messages"][0]["content"]
    assert 'idx=0 en="Thank you."' in seen["body"]["messages"][1]["content"]


def test_progress_heartbeat_mengirim_berkala():
    from engine.progress import Progress

    sent: list[str] = []

    class P(Progress):
        def send(self, status, message="", percent=None):  # type: ignore[override]
            sent.append(status)

    p = P(LOG, "", "s", "j1", "c")
    import time as _t

    with p.heartbeat("synthesizing", "msg", every=0.05):
        _t.sleep(0.18)
    assert len(sent) >= 2


def test_vertical_bitrate_tier_video_panjang(tmp_path):
    from engine import vertical

    cmd_long = vertical.build_render_cmd(
        tmp_path / "s.mp4",
        tmp_path / "v.wav",
        tmp_path / "x.ass",
        tmp_path / "o.mp4",
        "horizontal",
        duration_s=678.0,
    )
    assert "800k" in cmd_long and "96k" in cmd_long
    cmd_short = vertical.build_render_cmd(
        tmp_path / "s.mp4",
        tmp_path / "v.wav",
        tmp_path / "x.ass",
        tmp_path / "o.mp4",
        "horizontal",
        duration_s=30.0,
    )
    assert "2500k" in cmd_short and "192k" in cmd_short


def test_vertical_cmd_tier_bitrate_durasi(tmp_path):
    from engine import vertical

    args = dict(
        src=tmp_path / "s.mp4",
        voice_wav=tmp_path / "v.wav",
        ass=tmp_path / "x.ass",
        out=tmp_path / "o.mp4",
        aspect="horizontal",
    )
    short = vertical.build_render_cmd(**args, duration_s=60)
    long_ = vertical.build_render_cmd(**args, duration_s=700)
    assert "-maxrate" in short and "-maxrate" in long_
    i = long_.index("-maxrate")
    assert long_[i + 1] == "800k"
    j = short.index("-maxrate")
    assert short[j + 1] == "2500k"
    # audio: panjang 96k, pendek 192k
    assert long_[long_.index("-b:a") + 1] == "96k"
    assert short[short.index("-b:a") + 1] == "192k"


def test_pick_target_size_tier_resolusi():
    # sumber full-HD+ (sisi pendek >= 1080) → kanvas penuh
    assert vertical.pick_target_size(1920, 1080) == (1080, 1920)
    assert vertical.pick_target_size(1080, 1920) == (1080, 1920)
    assert vertical.pick_target_size(3840, 2160) == (1080, 1920)
    # sumber rendah-res → 720x1280 (docs/01 §35)
    assert vertical.pick_target_size(1280, 720) == (720, 1280)
    assert vertical.pick_target_size(854, 480) == (720, 1280)
    assert vertical.pick_target_size(720, 1280) == (720, 1280)
    # probe tak diketahui → kanvas penuh (defensif, perilaku lama)
    assert vertical.pick_target_size(0, 0) == (1080, 1920)


def test_vertical_cmd_tier_bitrate_kanvas_720(tmp_path):
    args = dict(
        src=tmp_path / "s.mp4",
        voice_wav=tmp_path / "v.wav",
        ass=tmp_path / "x.ass",
        out=tmp_path / "o.mp4",
        aspect="horizontal",
    )
    long_ = vertical.build_render_cmd(**args, duration_s=700, w=720, h=1280)
    assert long_[long_.index("-crf") + 1] == "23"
    assert long_[long_.index("-maxrate") + 1] == "500k"
    assert long_[long_.index("-bufsize") + 1] == "1000k"
    assert long_[long_.index("-b:a") + 1] == "96k"
    short = vertical.build_render_cmd(**args, duration_s=60, w=720, h=1280)
    assert short[short.index("-crf") + 1] == "20"
    assert short[short.index("-maxrate") + 1] == "1500k"
    assert short[short.index("-b:a") + 1] == "192k"
    # kanvas 720x1280 benar-benar dipakai filter scale/crop
    fc = long_[long_.index("-filter_complex") + 1]
    assert "scale=720:1280" in fc and "crop=720:1280" in fc


def test_build_ass_fontsize_proporsional_kanvas(tmp_path):
    cues = [srt.Cue(idx=0, start=0.0, end=2.0, text="Hello world")]
    t1080 = vertical.build_ass(cues, tmp_path / "a.ass").read_text()
    t720 = vertical.build_ass(cues, tmp_path / "b.ass", w=720, h=1280).read_text()
    assert "PlayResY: 1920" in t1080 and ",58," in t1080
    assert "PlayResY: 1280" in t720 and ",39," in t720
    # MarginV 19% kanvas: 365 vs 243 — posisi relatif identik
    assert ",64,64,365,1" in t1080
    assert ",64,64,243,1" in t720


# ---------------------------------------------------------------------------
# §36 — single-pass: guard degenerasi transkrip + groq_llm chunked/fallback
# ---------------------------------------------------------------------------


def test_transcript_degenerate_kunci_decoder():
    # kasus nyata §32: 10s ucapan riil (conf 0.24) tapi decoder terkunci
    # mengembalikan satu frasa pendek
    tr = _tr([(0.0, 10.0, "Terima kasih.")])
    tr.segments[0].no_speech_prob = 0.24
    assert transcribe_mod.transcript_degenerate(tr, 10.0) is True


def test_transcript_degenerate_sehat_dan_pengecualian():
    sehat = _tr(
        [
            (0.0, 5.0, "Kok banyak orang tua disini?"),
            (5.0, 10.0, "Anak-anaknya merantau dan bekerja di kota besar."),
        ]
    )
    for seg in sehat.segments:
        seg.no_speech_prob = 0.2
    assert transcribe_mod.transcript_degenerate(sehat, 10.0) is False
    # audio pendek (< 8s) dikecualikan — ucapan singkat sah bisa sangat pendek
    pendek = _tr([(0.0, 6.0, "Halo.")])
    pendek.segments[0].no_speech_prob = 0.1
    assert transcribe_mod.transcript_degenerate(pendek, 6.0) is False
    # tidak yakin ucapan (conf >= 0.5) → bukan urusan guard ini (jalur musik/
    # keheningan gagal lewat pemeriksaan teks kosong / no_speech)
    musik = _tr([(0.0, 30.0, "Thank you.")])
    musik.segments[0].no_speech_prob = 0.9
    assert transcribe_mod.transcript_degenerate(musik, 30.0) is False
    # ucapan sah yang jarang (30 char / 60s) TIDAK boleh kena
    jarang = _tr([(0.0, 60.0, "Selamat datang. Hari ini kita mulai.")])
    jarang.segments[0].no_speech_prob = 0.3
    assert transcribe_mod.transcript_degenerate(jarang, 60.0) is False


def test_groq_llm_chunking_dan_konteks_antar_chunk(monkeypatch):
    from engine import groq_llm

    items = [(i, f"kalimat nomor {i} di video ini", 500) for i in range(130)]
    dumps = json.dumps
    calls = []

    class R:
        status_code = 200

        def __init__(self, content):
            self._c = content

        def json(self):
            return {"choices": [{"message": {"content": self._c}}]}

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json)
        idxs = [int(m) for m in re.findall(r"idx=(\d+)", json["messages"][1]["content"])]
        return R(dumps({"segments": [{"idx": i, "text_en": f"line {i}"} for i in idxs]}))

    monkeypatch.setattr(groq_llm.requests, "post", fake_post)
    progress = []
    out = groq_llm.translate_segments(
        LOG, "k", items, on_chunk=lambda i, n: progress.append((i, n))
    )
    # 130 segmen > 60/chunk → minimal 2 panggilan; semua idx pulang 1:1
    assert len(calls) >= 2
    assert len(out) == 130 and out[129] == "line 129"
    assert progress == [(i, len(calls)) for i in range(1, len(calls) + 1)]
    # chunk ke-2 membawa konteks terjemahan chunk sebelumnya
    assert "context only" in calls[1]["messages"][1]["content"]
    assert 'en="line' in calls[1]["messages"][1]["content"]


def test_groq_llm_fallback_model_saat_404(monkeypatch):
    from engine import groq_llm

    seen_models = []

    class R404:
        status_code = 404
        text = "The model openai/gpt-oss-120b does not exist or you do not have access"

    class R200:
        status_code = 200

        def json(self):
            return {
                "choices": [{"message": {"content": '{"segments":[{"idx":0,"text_en":"Hi."}]}'}}]
            }

    def fake_post(url, headers=None, json=None, timeout=None):
        seen_models.append(json["model"])
        return R404() if json["model"] == groq_llm.MODEL_PRIMARY else R200()

    monkeypatch.setattr(groq_llm.requests, "post", fake_post)
    monkeypatch.setattr(groq_llm.time, "sleep", lambda s: None)
    out = groq_llm.translate_segments(LOG, "k", [(0, "Hai.", 50)])
    assert out == {0: "Hi."}
    assert seen_models == [groq_llm.MODEL_PRIMARY, groq_llm.MODEL_FALLBACK]


def test_groq_llm_kuota_harian_fail_fast(monkeypatch):
    from engine import groq_llm

    calls = {"n": 0}
    slept = []

    class R429Daily:
        status_code = 429
        text = '{"error":{"message":"You have reached your daily quota for this model."}}'

    def fake_post(url, **kw):
        calls["n"] += 1
        return R429Daily()

    monkeypatch.setattr(groq_llm.requests, "post", fake_post)
    monkeypatch.setattr(groq_llm.time, "sleep", lambda s: slept.append(s))
    with pytest.raises(groq_llm.GroqLlmError) as e:
        groq_llm.translate_segments(LOG, "k", [(0, "Hai.", 50)])
    assert e.value.code == "groq_daily_quota"
    assert calls["n"] == 1  # nol retry sia-sia (§31)
    assert not slept


def test_groq_llm_429_tpm_tunggu_jendela_penuh(monkeypatch):
    # insiden §36 nyata: TPM 8000 gpt-oss-120b — backoff buta 2/4/8s mati
    # sebelum jendela 60s berganti; retry TPM wajib menunggu ~60s
    from engine import groq_llm

    class R429TPM:
        status_code = 429
        text = (
            '{"error":{"message":"Rate limit reached for model `openai/gpt-oss-120b` '
            "in organization `org_x` service tier `on_demand` on tokens per minute "
            '(TPM): Limit 8000, Used 5540, Requested 3175."}}'
        )

    class R200:
        status_code = 200

        def json(self):
            return {
                "choices": [{"message": {"content": '{"segments":[{"idx":0,"text_en":"Hi."}]}'}}]
            }

    calls = {"n": 0}

    def fake_post(url, **kw):
        calls["n"] += 1
        return R429TPM() if calls["n"] == 1 else R200()

    slept = []
    monkeypatch.setattr(groq_llm.requests, "post", fake_post)
    monkeypatch.setattr(groq_llm.time, "sleep", lambda s: slept.append(s))
    out = groq_llm.translate_segments(LOG, "k", [(0, "Hai.", 50)])
    assert out == {0: "Hi."}
    assert calls["n"] == 2
    assert len(slept) == 1 and 60.0 <= slept[0] <= 63.0  # jendela TPM penuh


def test_groq_llm_retry_after_429_dihormati(monkeypatch):
    from engine import groq_llm

    class R429:
        status_code = 429
        text = "rate limit per minute"
        headers = {"retry-after": "7"}

    class R200:
        status_code = 200

        def json(self):
            return {
                "choices": [{"message": {"content": '{"segments":[{"idx":0,"text_en":"Hi."}]}'}}]
            }

    calls = {"n": 0}

    def fake_post(url, **kw):
        calls["n"] += 1
        return R429() if calls["n"] == 1 else R200()

    slept = []
    monkeypatch.setattr(groq_llm.requests, "post", fake_post)
    monkeypatch.setattr(groq_llm.time, "sleep", lambda s: slept.append(s))
    out = groq_llm.translate_segments(LOG, "k", [(0, "Hai.", 50)])
    assert out == {0: "Hi."}
    # header retry-after dihormati (+ jitter anti-stampede 0–1.5s)
    assert len(slept) == 1 and 7.0 <= slept[0] <= 8.5


def test_split_long_rows_en_kosong_alur_praterjemah():
    # §37: build pertama memecah baris SAAT text_en masih kosong — driver
    # otomatis teks sumber, grup EN tetap kosong utk diisi terjemahan 1:1
    asli = (
        "Di dalam pembuluh darah, aliran sirkulasi melambat, pemecahan gula dan "
        "lemak pun terhambat, risiko hipertensi, diabetes, penyakit jantung, dan "
        "kematian dini meningkat 20 sampai 30 persen."
    )
    rows = [{"idx": 0, "start": 34.0, "end": 50.44, "text_id": asli, "text_en": ""}]
    out = srt.split_long_rows(rows)
    assert len(out) >= 2
    # semua slot EN kosong siap diterjemahkan per baris
    assert all(r["text_en"] == "" for r in out)
    # gabungan fragmen ID = teks asli utuh & urut
    assert " ".join(r["text_id"] for r in out) == asli
    # timing kontigu menempel batas baris sumber
    assert out[0]["start"] == 34.0 and out[-1]["end"] == 50.44
    for a, b in zip(out, out[1:], strict=False):
        assert a["end"] == b["start"]
    assert [r["idx"] for r in out] == list(range(len(out)))


def test_merge_short_tails_gabung_fragmen_ekor():
    rows = [
        {
            "idx": 0,
            "start": 50.44,
            "end": 57.63,
            "text_id": "Kabar baiknya, rantai resiko ini bisa diputus hari ini juga lewat gerakan aktif yang",
            "text_en": "",
        },
        {"idx": 1, "start": 57.63, "end": 58.32, "text_id": "terukur.", "text_en": ""},
        {"idx": 2, "start": 60.0, "end": 62.0, "text_id": "Salam sehat!", "text_en": ""},
    ]
    out = srt.merge_short_tails(rows)
    assert len(out) == 2
    assert out[0]["text_id"].endswith("gerakan aktif yang terukur.")
    assert out[0]["end"] == 58.32  # timing ekor diserap
    # kalimat utuh berawalan kapital TIDAK digabung
    assert out[1]["text_id"] == "Salam sehat!"
    assert [r["idx"] for r in out] == [0, 1]


def test_merge_short_tails_gap_dan_batas_char():
    # celah > 0.6s → jangan gabung (jeda ucapan nyata)
    rows = [
        {
            "idx": 0,
            "start": 0.0,
            "end": 2.0,
            "text_id": "Kalimat pertama yang isi baris",
            "text_en": "",
        },
        {"idx": 1, "start": 5.0, "end": 5.5, "text_id": "sambungan.", "text_en": ""},
    ]
    assert len(srt.merge_short_tails(rows)) == 2
    # hasil gabung melebihi max_chars → jangan gabung
    panjang = "x " * 60
    rows = [
        {"idx": 0, "start": 0.0, "end": 2.0, "text_id": panjang.strip(), "text_en": ""},
        {"idx": 1, "start": 2.0, "end": 2.5, "text_id": "ekor.", "text_en": ""},
    ]
    assert len(srt.merge_short_tails(rows)) == 2
