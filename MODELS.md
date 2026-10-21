# Model weights: sources and licenses

Weights are not stored in git. `python scripts/fetch_models.py` downloads them
into `models/`, checks SHA-256 where a hash is pinned, and converts BTC to
safetensors.

| File | Source | License | SHA-256 of the download |
|---|---|---|---|
| `btc_majmin.safetensors` | [jayg996/BTC-ISMIR19](https://github.com/jayg996/BTC-ISMIR19) `test/btc_model.pt` @ `2682317` | MIT (code and weights) | `71c2c5db17e8c43b8a9a9da5db36ef2d667158c07a214eba16344c154c00bf54` |
| `btc_full.safetensors` | same repo, `test/btc_model_large_voca.pt` | MIT | `1673d23f8f9a55ae7f9e8b80a51da616debb22675b8d8b67ea6ce0ef37b0ab51` |
| `beat_this-final0.ckpt` | [CPJKU/beat_this](https://github.com/CPJKU/beat_this), JKU cloud share | MIT (code and weights) | not pinned yet (host unreachable when written) |
| `beat_this-small0.ckpt` | same | MIT | not pinned yet |

The BTC safetensors files hold the original parameters unchanged plus two
scalars, `feature_mean` and `feature_std`, used to normalise the CQT input.

Vendored code: `src/chordanalyzer/chords/btc_model.py` is adapted from
BTC-ISMIR19 (see `LICENSES/BTC-ISMIR19.txt`). Beat This! is used as a pip
dependency (`LICENSES/beat_this.txt` for reference).

Training data: both models were trained by their authors on copyrighted
recordings with published annotations (Isophonics, Billboard, etc. for BTC;
16 beat datasets for Beat This!). The weights themselves are MIT licensed;
get legal advice before a large commercial launch, as the plan notes.

Not used, for license reasons: madmom models (CC BY-NC-SA 4.0), Essentia (AGPL-3.0).
