"""Apply the selected noisy-only algorithm to input/*.wav, writing PCM16 output/*.wav."""
from pathlib import Path
import argparse
import json
from dataclasses import replace
import numpy as np
import soundfile as sf
from hybrid_v3 import Config, enhance

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,default=Path("input"))
    parser.add_argument("--output",type=Path,default=Path("output"))
    args=parser.parse_args()
    if args.input.resolve()==args.output.resolve(): parser.error("Use a distinct output folder.")
    paths=sorted(args.input.glob("*.wav"))
    if not paths: parser.error("No input WAVs found.")
    settings=json.loads(Path(__file__).with_name("selected_config.json").read_text())
    cfg=replace(Config(),**settings["config"])
    args.output.mkdir(parents=True,exist_ok=True)
    for path in paths:
        x,sr=sf.read(path,dtype="float64")
        if sr!=16000 or x.ndim!=1: raise ValueError(f"16 kHz mono required: {path}")
        y=enhance(x,sr,cfg,estimator="quantile",blend=settings["blend"])
        if x.shape!=y.shape or not np.isfinite(y).all(): raise ValueError(f"Invalid output: {path}")
        sf.write(args.output/path.name,np.clip(y,-1,1),sr,subtype="PCM_16")
if __name__=="__main__":
    main()
