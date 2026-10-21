# chordanalyzer

Upload an audio file, get back its tempo, key and beat-aligned chord progression as JSON.

The same engine is available three ways:

- **HTTP API** (`chordanalyzer-api`): any language uploads a file with a multipart POST.
- **CLI** (`chordanalyzer song.mp3`).
- **Python package** (`from chordanalyzer import Analyzer`).

| Job | Model | License |
|---|---|---|
| Chords | [BTC](https://github.com/jayg996/BTC-ISMIR19) (ISMIR 2019), 25-label and 170-label models | MIT |
| Beats, downbeats, tempo | [Beat This!](https://github.com/CPJKU/beat_this) (ISMIR 2024); librosa fallback if its weights are missing | MIT |
| Key | Krumhansl-Kessler profiles on CQT chroma, plus a vote from the detected chords | this repo |

See [MODELS.md](MODELS.md) for weight sources, hashes and licenses.

## Quick start

Requirements: Python 3.12 or 3.13, ffmpeg (with ffprobe) on the PATH.

```bash
uv venv && source .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio   # CPU-only torch
uv pip install -e ".[api,eval,dev]"
python scripts/fetch_models.py          # BTC + Beat This! weights into models/
```

CLI:

```bash
chordanalyzer song.mp3                  # summary
chordanalyzer song.mp3 --json out.json  # full result
chordanalyzer song.mp3 --vocabulary majmin --notation name -v
```

API:

```bash
API_KEYS=my-secret chordanalyzer-api    # http://localhost:8000, docs at /docs
```

Docker (CPU image, weights baked in):

```bash
API_KEYS=my-secret docker compose up --build
```

## Calling the API

`POST /v1/analyze` with a multipart form. Only `file` is required.

| Field | Values | Default |
|---|---|---|
| `file` | mp3, wav, flac, m4a/aac, ogg; up to 50 MB and 5 to 600 s | |
| `chord_vocabulary` | `majmin` (24 chords + N) or `full` (170 labels: 12 roots x 14 qualities, N, X) | `full` |
| `include` | comma list of `beats`, `bars`, `tempo_curve`, `chord_probabilities` | `beats,bars` |
| `notation` | `name`, `roman` or `both` | `both` |

Header `X-API-Key: <key>` when the server has `API_KEYS` set.

```bash
curl -H "X-API-Key: my-secret" -F file=@song.mp3 -F chord_vocabulary=majmin \
     http://localhost:8000/v1/analyze
```

```javascript
const form = new FormData();
form.append("file", fileInput.files[0]);
const res = await fetch("http://localhost:8000/v1/analyze", {
  method: "POST", headers: { "X-API-Key": "my-secret" }, body: form,
});
const analysis = await res.json();
```

```python
import httpx
with open("song.mp3", "rb") as f:
    r = httpx.post("http://localhost:8000/v1/analyze", files={"file": f},
                   headers={"X-API-Key": "my-secret"}, timeout=120)
print(r.json()["key"]["name"])
```

For other languages, generate a typed client from the committed [openapi.json](openapi.json)
(also served live at `/openapi.json`), for example with `openapi-generator-cli generate -i
openapi.json -g csharp` (or `java`, `kotlin`, `swift5`, `typescript-fetch`, `go`, `php`, ...).

### Other endpoints

| Endpoint | Purpose |
|---|---|
| `POST /v1/jobs` | Same form as `/v1/analyze`, returns `202 {"job_id": ...}` and a `Location` header immediately |
| `GET /v1/jobs/{job_id}` | `queued`, `running`, `succeeded` (with `result`) or `failed` (with `error`) |
| `GET /v1/health` | `200` once models are loaded (`503` while loading), with model names |
| `GET /docs`, `GET /openapi.json` | Interactive docs and the machine-readable spec |

### Response

```json
{
  "schema_version": "1.0",
  "engine": {"version": "0.1.0", "chord_model": "btc-full", "beat_model": "beat_this-final0", "key_model": "kk-profile+chord-vote"},
  "file": {"sha256": "7b9b...", "duration": 11.0, "format": "wav", "codec": "pcm_s16le"},
  "tempo": {"bpm": 120.2, "confidence": 0.977, "bpm_alternatives": [60.1, 240.4], "beats_per_bar": 4, "time_signature": "4/4"},
  "key": {"tonic": "C", "mode": "major", "name": "C major", "label": "C:maj", "confidence": 0.93,
          "runner_up": {"tonic": "A", "mode": "minor", "name": "A minor", "label": "A:min"}},
  "chord_vocabulary": "full",
  "chords": [
    {"start": 0.0,  "end": 1.02, "label": "N",     "name": "N.C.", "bar": 0, "beat": 0, "confidence": 0.935},
    {"start": 1.02, "end": 3.02, "label": "C:maj", "name": "C", "roman": "I", "bar": 1, "beat": 1, "confidence": 0.912},
    {"start": 3.02, "end": 5.02, "label": "G:maj", "name": "G", "roman": "V", "bar": 2, "beat": 1, "confidence": 0.946}
  ],
  "progression": [
    {"bar": 1, "bars": 1, "label": "C:maj", "name": "C", "roman": "I"},
    {"bar": 2, "bars": 1, "label": "G:maj", "name": "G", "roman": "V"}
  ],
  "beats": [{"time": 1.02, "bar": 1, "beat": 1}, {"time": 1.51, "bar": 1, "beat": 2}],
  "bars": [{"bar": 0, "start": 0.0, "end": 1.02}, {"bar": 1, "start": 1.02, "end": 3.02}],
  "warnings": []
}
```

Field rules:

- Times are seconds, rounded to 2 decimals.
- `label` is always Harte syntax (`root:quality`, `N` = no chord, `X` = unknown), with the root
  spelled for the key (`Bb:maj`, not `A#:maj`, in F major). Use it for machine processing;
  `name` and `roman` are for display.
- `roman` is relative to the detected key, using major-scale degrees for both modes
  (A minor reads `i`, `bIII`, `iv`, `V`, `bVI`, `bVII`).
- `bar` is 1-based from the first downbeat; `0` is a pickup before it. `beat` is 1-based within
  the bar; `0` means before the first detected beat.
- `progression` is one chord per bar (the one sounding longest), with consecutive repeats
  collapsed into `bars`.
- `warnings` codes: `low_tempo_confidence`, `ambiguous_key`, `mostly_no_chord`,
  `clipped_audio`, `beat_tracker_fallback` (Beat This! weights missing, weaker beats, 4/4 assumed).
- Adding fields is non-breaking. Renaming or removing one bumps `schema_version` and the URL
  prefix (`/v2`). `tests/test_api.py` fails if the OpenAPI spec drifts from `openapi.json`.
- Same file, same options and same engine version give the same JSON (results are cached by
  file hash). Processing time is in the `X-Processing-Time` header, cache status in `X-Cache`.

### Errors

`application/problem+json` (RFC 9457): `{"type": ".../unsupported-media", "title": "...", "status": 415, "detail": "..."}`

| Status | When |
|---|---|
| 400 | Missing file, bad option value |
| 401 | Missing or wrong API key |
| 404 | Unknown or expired job id |
| 413 | File over the upload limit |
| 415 | Not decodable audio (the format is sniffed with ffprobe; the extension is ignored) |
| 422 | Audio shorter than 5 s or longer than 600 s |
| 503 | Models still loading, or too many analyses queued (with `Retry-After`) |

## Configuration

Environment variables (see `src/chordanalyzer/settings.py`):

| Variable | Default | Meaning |
|---|---|---|
| `API_KEYS` | empty (auth off) | Comma list of accepted keys; stored hashed, compared in constant time |
| `MODELS_DIR` | `./models` | Where the weights live |
| `DEVICE` | `cpu` | `cpu` or `cuda` |
| `BEAT_TRACKER` | `auto` | `auto` (Beat This!, else librosa), `beat_this`, `librosa` |
| `BEAT_CHECKPOINT` | `final0` | `final0` (accurate) or `small0` (8 MB, faster) |
| `WORKERS` | `1` | Analysis processes; each loads the models (about 1.5 to 2 GB RAM) |
| `MAX_QUEUE` | `8` | Analyses allowed to wait before answering 503 |
| `MAX_UPLOAD_MB` | `50` | Upload limit |
| `MIN_DURATION_S` / `MAX_DURATION_S` | `5` / `600` | Duration limits |
| `TORCH_THREADS` | torch default | Threads per worker; set to vCPUs per worker |
| `CACHE_DIR` | memory only | Directory for cached results |
| `CORS_ORIGINS` | none | Comma list, for calling the API from a browser app |

## How it works

1. **Validate and decode**: ffprobe checks the real format and duration; ffmpeg decodes once to
   22,050 Hz mono float32, which both models share.
2. **Beats**: Beat This! returns beat and downbeat times.
3. **Tempo**: BPM = 60 / median inter-beat interval; confidence = 1 minus the coefficient of
   variation; meter = median beats between downbeats; a BPM-per-bar curve.
4. **Chord frames**: 144-bin log-CQT (24 bins/octave, hop 2048, about 93 ms) through BTC on
   108-frame windows with 50% overlap, softmax averaged where windows overlap.
5. **Key**: chroma (folded from the same CQT) and the chord sequence are correlated with the 24
   Krumhansl-Kessler key profiles, plus a bonus for time spent on each key's tonic chord.
6. **Smoothing**: chord probabilities averaged between beats, one chord per beat chosen by a
   Viterbi pass with a switch penalty, then merged into segments.
7. **Spelling**: key-aware enharmonics (F# in D major, Gb in Db major), display names, Roman
   numerals, per-bar progression.

```
src/chordanalyzer/
  audio.py        ffprobe validation, ffmpeg decode
  beats.py        BeatTracker protocol, Beat This! and librosa trackers
  tempo.py        BPM, tempo curve, meter
  chords/         ChordModel protocol, vendored BTC, CQT + overlapping-window inference, vocabularies
  key.py          KeyEstimator protocol, profile + chord-vote estimator
  smoothing.py    beat pooling, Viterbi, segments
  spelling.py     names, enharmonics, Roman numerals
  schema.py       Pydantic models = the JSON contract
  pipeline.py     Analyzer: runs the steps, bar/beat grid
  cli.py          command line
  api/            FastAPI app, process-pool engine, in-memory job store
scripts/          fetch_models.py, export_openapi.py
eval/             run_eval.py (mir_eval), make_synthetic_set.py
tests/            unit, API, golden and end-to-end tests
```

## Tests and evaluation

```bash
pytest -q                     # tests marked `models` are skipped if the weights are missing
UPDATE_GOLDEN=1 pytest tests/test_pipeline.py   # refresh golden snapshots after an intended change
python eval/run_eval.py DATASET_DIR --out results.csv
```

`eval/run_eval.py` expects audio files with `.lab` (chords), and optionally `.key`, `.bpm` and
`.beats` annotations next to them, and reports chord WCSR (root, majmin, mirex, sevenths), key
MIREX score, tempo Accuracy1/2 and beat F-measure. `eval/make_synthetic_set.py` writes a
synthetic set to smoke-test it; real accuracy needs real annotated songs (phase 3).
