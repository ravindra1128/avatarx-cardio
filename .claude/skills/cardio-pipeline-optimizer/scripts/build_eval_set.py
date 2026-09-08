"""Index data/eval_corpus/ into eval_manifest.json with a deterministic tune/holdout split.

The split is keyed on each file's sha1, not on order or name, so it is identical on
every machine and every run — which is what makes "never tune on holdout" enforceable.

Usage:
  python build_eval_set.py --holdout-frac 0.4 --seed 13
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess

from _common import corpus_dir, dump_json, manifest_path, repo

VIDEO_EXT = {".webm", ".avi", ".mp4", ".mkv"}


def _sha1(path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _probe(path) -> dict:
    """codec/width/height/frame count via ffprobe (present wherever ffmpeg is)."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-count_frames", "-show_entries",
             "stream=codec_name,width,height,nb_read_frames",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=300).stdout.strip().split(",")
        codec, w, h, n = (out + ["", "", "", ""])[:4]
        return {"codec": codec, "width": int(w or 0), "height": int(h or 0),
                "n_frames": int(n or 0)}
    except Exception as e:  # noqa: BLE001
        return {"codec": None, "width": 0, "height": 0, "n_frames": 0,
                "probe_error": f"{type(e).__name__}: {e}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-frac", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    corpus = corpus_dir()
    files = sorted(p for p in corpus.iterdir()
                   if p.suffix.lower() in VIDEO_EXT and p.is_file())
    if not files:
        raise SystemExit(f"no recordings in {corpus} — copy the retained clips there first")

    records = []
    seen: dict[str, str] = {}
    for p in files:
        sha = _sha1(p)
        # Exact-duplicate recordings (same bytes under two names) would
        # double-weight one clip and can land on both sides of the split.
        # Found on the first real run: two holdout records with identical
        # coverage to 16 digits were one recording retained twice.
        if sha in seen:
            print(f"[eval-set] {p.stem:<32} DUPLICATE of {seen[sha]} — skipped")
            continue
        seen[sha] = p.stem
        # deterministic split: hash the sha with the seed, threshold on the fraction
        r = int(hashlib.sha1(f"{args.seed}:{sha}".encode()).hexdigest(), 16) / 16 ** 40
        split = "holdout" if r < args.holdout_frac else "tune"
        rec = {"id": p.stem, "path": str(p.relative_to(repo())), "split": split,
               "sha1": sha[:12], "sidecar": (p.parent / f"{p.name}.timestamps.json").exists(),
               "source": "demo" if p.stem.startswith("demo_") else "webapp",
               **_probe(p)}
        records.append(rec)
        print(f"[eval-set] {rec['id']:<32} {split:<7} {rec['width']}x{rec['height']} "
              f"{rec['codec']} {rec['n_frames']}f sidecar={rec['sidecar']}")

    out = dump_json({"seed": args.seed, "holdout_frac": args.holdout_frac,
                     "records": records}, manifest_path())
    n_h = sum(r["split"] == "holdout" for r in records)
    print(f"[eval-set] {len(records)} records: {len(records) - n_h} tune / {n_h} holdout -> {out}")


if __name__ == "__main__":
    main()
