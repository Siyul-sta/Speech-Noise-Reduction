"""Post-selection ablation on the two largest PESQ drops; no model selection."""
from pathlib import Path
from unittest.mock import patch
import argparse
import pandas as pd
from bench import score,HERE
import adaptive
from contextlib import nullcontext

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,default=HERE.parent/"noise_reduction_challenge")
    args=p.parse_args()
    scores=pd.read_csv(HERE/"results/all_file_scores.csv").query("method=='cross_low' and stage=='validation'")
    ids=scores.nsmallest(2,"pesq_vs_base").segment_id
    rows=pd.read_csv(args.data/"validation/samples.csv").set_index("segment_id",drop=False)
    out=[]
    for name,override in [("selected",nullcontext()),("bypass_sinusoid_subtraction",patch.object(adaptive,"notch_lines",lambda x,sr:x)),("static_noise_psd",patch.object(adaptive,"relative_noise",lambda power,initial,full:initial[:,None]))]:
        with override:
            for sample in ids:
                record=score((args.data,rows.loc[sample].to_dict(),["cross_low"]))[0]
                record["diagnostic"]=name
                out.append(record)
    pd.DataFrame(out).to_csv(HERE/"results/outlier_ablation.csv",index=False)
