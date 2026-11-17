"""Build a filename-only manifest, without reading any audio content."""
import argparse
import csv
from pathlib import Path

def build(root: Path, stage: str, output: Path):
    noisy_folder = root / stage / "noisy"
    clean_folder = root / stage / "clean"
    if not noisy_folder.is_dir() or not clean_folder.is_dir():
        raise FileNotFoundError("Missing noisy/clean folders for " + stage)
    rows = []
    for noisy in sorted(noisy_folder.glob("*_noisy.wav")):
        name = noisy.name.replace("_noisy.wav", "")
        clean = clean_folder / (name + ".wav")
        if not clean.is_file():
            raise FileNotFoundError("Clean pair missing: " + str(clean))
        rows.append({
            "segment_id": name,
            "noisy_path": str(noisy.relative_to(root)),
            "clean_path": str(clean.relative_to(root)),
        })
    if not rows:
        raise ValueError("No WAV pairs discovered")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(stage, "paired files:", len(rows), "manifest:", output)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--stage", choices=("train", "validation"), required=True)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    build(a.root, a.stage, a.output)
