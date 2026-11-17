"""Collect all measured scores and create paired group comparisons against original."""
import pandas as pd
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent
out=root/"results"
frames=[]
for f in sorted(out.glob("*_*/per_file.csv")):
    stage=f.parent.name.split("_",1)[0]
    if stage in ("dev","confirm","extra","validation"):
        frame=pd.read_csv(f)
        frame["stage"]=stage
        frames.append(frame)
d=pd.concat(frames,ignore_index=True)
baseline=pd.read_csv(out/"baseline/final_wav_per_file.csv")
manifest=pd.read_csv(root/"validation_manifest.csv")
base=baseline.merge(manifest[["segment_id","target_snr_db"]],validate="one_to_one")
base["speaker"]=base.segment_id.str.split("_").str[0]
base["method"]="base"
base["stage"]="validation"
d=pd.concat([d,base],ignore_index=True)
assert not d.duplicated(["stage","method","segment_id"]).any()
paired=d.merge(d[d.method=="base"][["stage","segment_id","pesq","snr_db"]],on=["stage","segment_id"],suffixes=("","_base"),validate="many_to_one")
paired["pesq_vs_base"]=paired.pesq-paired.pesq_base
paired["snr_vs_base"]=paired.snr_db-paired.snr_db_base
paired["wins"]=paired.pesq_vs_base>0
paired.to_csv(out/"all_file_scores.csv",index=False)
def summarize(data,keys):
    return data.groupby(keys).agg(n=("pesq","size"),valid=("pesq","count"),pesq=("pesq","mean"),
        snr_db=("snr_db","mean"),pesq_vs_base=("pesq_vs_base","mean"),snr_vs_base=("snr_vs_base","mean"),
        wins=("wins","sum")).reset_index()
s=summarize(paired,["stage","method"])
s.sort_values(["stage","pesq"],ascending=[True,False]).to_csv(out/"all_summary.csv",index=False)
for group in ("noise_category","target_snr_db","speaker"):
    summarize(paired[paired.stage=="validation"],["method",group]).to_csv(out/("validation_by_"+group+".csv"),index=False)
