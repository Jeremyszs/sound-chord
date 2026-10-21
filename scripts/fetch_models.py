"""Download the model weights, verify their SHA-256 and convert BTC to safetensors.

Usage: python scripts/fetch_models.py [--models-dir models] [--beat-checkpoints final0 small0]

BTC checkpoints are pickled PyTorch files containing NumPy scalars, which
torch>=2.6 refuses to load by default. They are loaded once here with
weights_only=False (from a pinned commit with a known hash) and re-saved as
safetensors, so the service never unpickles anything at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

BTC_COMMIT = "2682317be668032e6e4b269ded36adaa2ad57df0"
BTC_URL = f"https://raw.githubusercontent.com/jayg996/BTC-ISMIR19/{BTC_COMMIT}/test/{{name}}"
BTC_FILES = {
    "majmin": ("btc_model.pt",
               "71c2c5db17e8c43b8a9a9da5db36ef2d667158c07a214eba16344c154c00bf54"),
    "full": ("btc_model_large_voca.pt",
             "1673d23f8f9a55ae7f9e8b80a51da616debb22675b8d8b67ea6ce0ef37b0ab51"),
}

BEAT_THIS_URL = "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/{name}.ckpt"
# Not pinned yet: the hosting server was unreachable when this script was written.
# Fill in after the first successful download (the script prints the hash).
BEAT_THIS_SHA256: dict[str, str] = {}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        while block := r.read(1 << 20):
            f.write(block)
    tmp.replace(dest)


def convert_btc(pt_path: Path, out_path: Path) -> None:
    import torch
    from safetensors.torch import save_file

    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    tensors = {k: v.contiguous() for k, v in ckpt["model"].items()}
    tensors["feature_mean"] = torch.tensor(float(ckpt["mean"]))
    tensors["feature_std"] = torch.tensor(float(ckpt["std"]))
    save_file(tensors, str(out_path), metadata={
        "source": f"jayg996/BTC-ISMIR19@{BTC_COMMIT}/test/{pt_path.name}", "license": "MIT"})


def fetch_btc(models_dir: Path, source_dir: Path | None) -> None:
    for vocab, (name, expected) in BTC_FILES.items():
        out = models_dir / f"btc_{vocab}.safetensors"
        if out.exists():
            print(f"ok      {out.name} (exists)")
            continue
        pt = models_dir / name
        if source_dir is not None and (source_dir / name).exists():
            pt.write_bytes((source_dir / name).read_bytes())
        elif not pt.exists():
            print(f"fetch   {name}")
            download(BTC_URL.format(name=name), pt)
        digest = sha256(pt)
        if digest != expected:
            sys.exit(f"SHA-256 mismatch for {name}: {digest}")
        convert_btc(pt, out)
        pt.unlink()
        print(f"ok      {out.name}")


def fetch_beat_this(models_dir: Path, names: list[str]) -> bool:
    all_ok = True
    for name in names:
        out = models_dir / f"beat_this-{name}.ckpt"
        if not out.exists():
            print(f"fetch   beat_this {name}")
            try:
                download(BEAT_THIS_URL.format(name=name), out)
            except Exception as e:  # noqa: BLE001
                print(f"FAILED  beat_this {name}: {e}\n"
                      f"        Download {BEAT_THIS_URL.format(name=name)} by hand into {out}")
                all_ok = False
                continue
        digest = sha256(out)
        expected = BEAT_THIS_SHA256.get(name)
        if expected and digest != expected:
            sys.exit(f"SHA-256 mismatch for {out.name}: {digest}")
        print(f"ok      {out.name} sha256={digest}")
    return all_ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--models-dir", type=Path, default=Path(__file__).resolve().parents[1] / "models")
    p.add_argument("--btc-source-dir", type=Path, default=None,
                   help="Use BTC .pt files from a local clone instead of downloading")
    p.add_argument("--beat-checkpoints", nargs="*", default=["final0", "small0"])
    args = p.parse_args()
    args.models_dir.mkdir(parents=True, exist_ok=True)
    fetch_btc(args.models_dir, args.btc_source_dir)
    if not fetch_beat_this(args.models_dir, args.beat_checkpoints):
        print("Beat This! weights missing: the analyzer will fall back to librosa beat tracking.")


if __name__ == "__main__":
    main()
