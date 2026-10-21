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


def test_upload_proxy_mengirim_signature_hmac(tmp_path, monkeypatch):
    import hashlib
    import hmac as hmac_mod

    from engine import upload as upload_mod

    p = tmp_path / "hasil.mp4"
    p.write_bytes(b"x" * 2048)
    captured = {}

    class R:
        status_code = 200
        text = ""

        def json(self):
            return {
                "drive_file_id": "fid123",
                "drive_web_url": "https://drive.google.com/file/d/fid123/view",
                "bytes": 2048,
            }

    def fake_post(url, data=None, files=None, timeout=None, headers=None):
        captured["url"] = url
        captured["data"] = data
        captured["headers"] = headers
        return R()

    monkeypatch.setattr(upload_mod.requests, "post", fake_post)
    res = upload_mod.upload_result(
        LOG,
        p,
        proxy_url="https://app.example.com/api/hooks/v1/jobs/j1/result",
        filename="hasil.mp4",
        content_type="video/mp4",
        job_id="j1",
        auth_secret="sekret",  # noqa: S106 — secret dummy utk test
    )
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    expect = "sha256=" + hmac_mod.new(b"sekret", f"j1.{sha}".encode(), hashlib.sha256).hexdigest()
    assert captured["headers"]["x-inggrisin-signature"] == expect
    assert captured["data"]["sha256"] == sha
    assert res.drive_file_id == "fid123" and res.mode == "proxy"


# ---------------------------------------------------------------------------
# §29 — whisper translate langsung: aligner + endpoint translations
# ---------------------------------------------------------------------------


def test_align_translation_overlap_dan_fallback_pusat():
    id_segs = [
        transcribe_mod.Segment(0.0, 4.0, "Halo"),
        transcribe_mod.Segment(4.0, 8.0, "Dunia"),
        transcribe_mod.Segment(8.0, 12.0, "Apa kabar"),
    ]
    en_segs = [
        transcribe_mod.Segment(0.5, 3.5, "Hello"),  # overlap penuh ke slot 0
        transcribe_mod.Segment(3.6, 7.9, "World"),  # overlap terbesar slot 1
        transcribe_mod.Segment(13.0, 15.0, "How are you"),  # NOL overlap → pusat terdekat (slot 2)
    ]
    out = transcribe_mod.align_translation(id_segs, en_segs)
    assert out == ["Hello", "World", "How are you"]


def test_align_translation_gabung_urut_dan_kosong():
    id_segs = [transcribe_mod.Segment(0.0, 10.0, "Panjang")]
    en_segs = [
        transcribe_mod.Segment(0.0, 4.0, "Part one"),
        transcribe_mod.Segment(4.0, 8.0, "part two"),
    ]
    assert transcribe_mod.align_translation(id_segs, en_segs) == ["Part one part two"]
    assert transcribe_mod.align_translation(id_segs, []) == [""]


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
    assert "timestamp_granularities[]" not in captured["data"]  # transcriptions-only (400 bila dikirim)
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
