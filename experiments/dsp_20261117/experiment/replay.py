"""Re-run the measured candidate batches. Expensive; uses manifests and saved batch names."""
import subprocess
import sys
from pathlib import Path
import pandas as pd
import argparse
root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument("--data",type=Path,required=True)
a=parser.parse_args()
for stage in ("dev","confirm","extra","validation"):
    for f in sorted((root/"results").glob(stage+"_*/summary.csv")):
        modes=pd.read_csv(f).method.to_list()
        subprocess.run([sys.executable,str(root/"bench.py"),stage,*modes,"--data",str(a.data.resolve())],check=True)
