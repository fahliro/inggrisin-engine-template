# inggrisin engine

> ⚠️ **Status: SPEC TARGET.** Kode Python di folder ini masih generasi pertama
> (pipeline era konsep lama). Spec yang mengikat adalah dokumen ini +
> `docs/00-ARCHITECTURE.md` §5–§7. Penyelarasan kode = milestone M1.

Mesin komputasi **inggrisin**: repo GitHub yang dibuat otomatis di akun kamu
dari template, tempat GitHub Actions melakukan dubbing — memakai **kuota
Actions kamu sendiri**, **Groq API key kamu sendiri**, dan **OpenRouter key
kamu sendiri**. Tidak ada server inggrisin yang menyentuh video kamu.

## Dua mode

```
--mode first (build pertama — pipeline penuh)
  video (Drive kamu, lewat proxy sekali-pakai)
   ├─ ffmpeg      → extract audio 16kHz mono opus ≤24MB (chunk 10 menit bila perlu)
   ├─ Groq        → Whisper: transkrip + timestamp per kalimat
   ├─ OpenRouter  → terjemahkan + pasakan ke durasi (ketat, maks 1,25×, ringkas bila perlu)
   │                + pertahankan nama diri/istilah otomatis (tanpa kamus user)
   ├─ Piper TTS   → SATU suara Inggris untuk semua kalimat (CPU runner, gratis)
   ├─ ffmpeg      → SATU pass: susun 9:16 (sumber horizontal → latar blur)
   │                + burn subtitle di atas safe zone + mux + loudnorm
   └─Drive+HMAC   → (EN).mp4 ke folder video kamu; subtitle+status ke dashboard

--mode revision (rebuild sesudah edit subtitle — TANPA Groq, TANPA LLM)
  SRT tervalidasi (diambil dari app, persis seperti yang kamu kirim)
   + video asli (dari folder Drive yang sama)
   ├─ Piper TTS   → rekam ulang dari teks SRT
   ├─ ffmpeg      → satu pass seperti di atas
   └─ Drive+HMAC  → TIMPA file (EN).mp4 yang lama; laporkan revision_number
```

Biaya mode revision: **API Rp 0** — hanya menit GitHub Actions.

## Isi repo (target)

| Path | Fungsi |
|---|---|
| `.github/workflows/inggrisin.yml` | workflow utama (`workflow_dispatch`), timeout 300 menit |
| `.github/workflows/selfcheck.yml` | validasi lingkungan (Piper, ffmpeg, kunci) saat setup |
| `engine/main.py` | orkestrator `--mode first\|revision` (idempotent, step-wise) |
| `engine/download.py` | sumber via proxy app (first) / via Drive (revision) |
| `engine/audio.py` | ffmpeg: probe, extract, chunk, loudnorm |
| `engine/transcribe.py` | klien Groq Whisper + retry + stitch chunk **[first]** |
| `engine/translate.py` | klien OpenRouter: JSON strict, time-fit, NER istilah **[first]** |
| `engine/tts.py` | Piper TTS — satu suara, per kalimat |
| `engine/vertical.py` | **satu pass**: scale/blur 9:16 + burn subtitle + mux |
| `engine/upload.py` | Drive: tulis hasil; revision = timpa file id lama |
| `engine/callback.py` | hasil bertanda tangan HMAC (`X-Inggrisin-Signature`) |
| `engine/callback_guard.py` | jaring pengaman di step `if: always()` |
| `engine/config.py` | parsing & validasi environment (`INGGRISIN_*`) |
| `engine/log.py` | logging dengan sensor secret |
| `engine/report.py` | report.json inkremental |

## Keamanan

1. Semua secret di-`::add-mask::` di step pertama workflow.
2. Access token Google **tidak pernah** masuk runner — sumber diunduh lewat
   proxy sekali-pakai (TTL ≤ 15 menit) yang men-stream dari Drive di sisi
   Worker.
3. SRT revisi diambil lewat endpoint app ber-auth HMAC/installation token —
   bukan lewat payload dispatch (batas 64 KB).
4. Hasil dikirim balik dengan HMAC-SHA256 atas `"{job_id}.{raw_body}"`,
   secret sekali-pakai yang dienkripsi di database app.
5. `permissions: contents: read` — workflow tidak menulis ke repo.
6. `log.py` menyensor URL bertoken, bearer, dan pola `sk-`/`gsk-`.

## Batas yang perlu kamu tahu

| Batas | Nilai | Penanganan |
|---|---|---|
| Groq audio | 25 MB (free) / 100 MB (dev) | bitrate dinamis dari durasi; chunk 10 menit bila masih kelebihan |
| GitHub Actions (free) | 2.000 menit/bln | video 9:16 ≈ 1–1,5× durasi; 16:9 ≈ 2–3× durasi; app menampilkan estimasi sebelum mulai |
| Payload dispatch | ±64 KB | SRT/manifest via pull dari app (§Keamanan-3) |
| Callback body | 1 MB | subtitle dipotong; JSON lengkap → Drive (overflow) |
| Voice model | ±60 MB | di-download sekali, di-cache `actions/cache` |

## Menjalankan lokal (pengembangan)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m engine.main --mode first \
  --source ./contoh.mp4 --out ./out \
  --groq-key "$GROQ_API_KEY" --openrouter-key "$OPENROUTER_KEY"
# mode revision:
python -m engine.main --mode revision \
  --source ./contoh.mp4 --srt ./revisi.srt --manifest ./manifest.json --out ./out
```

Semua variabel lingkungan `INGGRISIN_*` didokumentasikan di `engine/config.py`
(spec) dan disinkronkan app sebagai secret repo saat setup.
