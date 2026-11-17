"""Evaluate standard STOI on decoded PCM16 outputs, separate from official PESQ."""
import pandas as pd,soundfile as sf
from pathlib import Path
from pystoi import stoi
import argparse
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('wav_dirs',nargs='+');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 rows=[]
 for r in pd.read_csv(a.data/'validation/samples.csv').to_dict('records'):
    c,sr=sf.read(a.data/r['clean_path'])
    for folder in a.wav_dirs:
        wav=Path(folder)/Path(r['noisy_path']).name;y,rate=sf.read(wav)
        assert rate==sr==16000 and y.shape==c.shape
        rows.append([r['segment_id'],Path(folder).name,stoi(c,y,sr,extended=False)])
 d=pd.DataFrame(rows,columns=['id','folder','stoi']);d.to_csv(a.out,index=False);print(d.groupby('folder').stoi.mean())
