"""Run fixed candidates on a split and score decoded PCM16 using the TA notebook."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from hybrid_v3 import Config, enhance

HERE = Path(__file__).resolve().parent
def ta_functions():
    ns = {"Path": Path}
    cells = json.loads((HERE/"noise_reduction_baselines.ipynb").read_text())["cells"]
    for i in (6,8,32):
        exec("".join(cells[i]["source"]),ns)
    return ns
TA = ta_functions()
BASE = Config(**json.loads((HERE/"best_config.json").read_text())["config"])

def score(task):
    data, row, modes = task
    clean, noisy = TA["read_pair"](data,row)
    v=int(row.get("valid_samples",len(clean)))
    assert 0<v<=len(clean)
    c=clean[:v]
    p0,err=TA["pesq_score"](c,noisy[:v])
    if err: raise ValueError(err)
    s0=TA["snr_db"](c,noisy[:v])
    results=[]
    for mode in modes:
        y=enhance(noisy,16000,replace(BASE,structure=mode),estimator="quantile",blend=1.)
        assert y.shape==noisy.shape and np.isfinite(y).all()
        wav=io.BytesIO()
        sf.write(wav,y,16000,format="WAV",subtype="PCM_16")
        wav.seek(0)
        assert sf.info(wav).subtype=="PCM_16"
        wav.seek(0)
        decoded,sr=sf.read(wav)
        assert sr==16000 and decoded.shape==y.shape
        p,err=TA["pesq_score"](c,decoded[:v])
        s=TA["snr_db"](c,decoded[:v])
        results.append(dict(segment_id=row["segment_id"],speaker=row["segment_id"].split("_")[0],
            noise_category=row["noise_category"],target_snr_db=row["target_snr_db"],method=mode,
            pesq=p,snr_db=s,delta_pesq=p-p0,delta_snr_db=s-s0,pesq_error=err))
    return results

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage", choices=["dev","confirm","extra","validation"])
    parser.add_argument("modes",nargs="+")
    parser.add_argument("--data",type=Path,default=HERE.parent/"noise_reduction_challenge")
    parser.add_argument("--workers",type=int,default=6)
    args=parser.parse_args()
    if args.stage=="validation":
        d=TA["load_manifest"](args.data,"validation")
    else:
        d=pd.read_csv(HERE/("extra.csv" if args.stage=="extra" else "train_split.csv")).query("stage==@args.stage")
    with ProcessPoolExecutor(args.workers) as pool:
        results=list(pool.map(score,[(args.data,r,args.modes) for r in d.to_dict("records")]))
    df=pd.DataFrame(sum(results,[]))
    out=HERE/"results"/(args.stage+"_"+"_".join(args.modes))
    out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/"per_file.csv",index=False)
    summary=TA["summarize_scores"](df)
    summary.to_csv(out/"summary.csv",index=False)
    print(summary.to_string(index=False))
    assert (summary.n_total==len(d)).all() and summary.pesq_failed.sum()==0
