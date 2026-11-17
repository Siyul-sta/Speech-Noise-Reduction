"""Use the TA's original notebook metric and summary functions on saved output WAVs."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,soundfile as sf
from scipy import signal

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--notebook',type=Path,required=True);p.add_argument('--wav-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--split',choices=['train','validation'],default='validation');a=p.parse_args()
 nb=json.loads(a.notebook.read_text(encoding='utf-8'));ns=globals()
 for i in (6,8,32):exec(compile(''.join(nb['cells'][i]['source']),f'TA cell {i}','exec'),ns)
 manifest=ns['load_manifest'](a.data,a.split);rows=[]
 for row in manifest.to_dict('records'):
  c,x=ns['read_pair'](a.data,row);v=int(row.get('valid_samples',len(c)));path=a.wav_dir/Path(row['noisy_path']).name;y,sr=sf.read(path,dtype='float64');assert sr==16000 and c.shape==y.shape and sf.info(path).subtype=='PCM_16';c=c[:v];x=x[:v];y=y[:v];s=ns['snr_db'](c,y);p,e=ns['pesq_score'](c,y);npesq,_=ns['pesq_score'](c,x);rows.append(dict(segment_id=row['segment_id'],method='optimized',noise_category=row['noise_category'],snr_db=s,delta_snr_db=s-ns['snr_db'](c,x),pesq=p,delta_pesq=p-npesq,pesq_error=e))
 d=pd.DataFrame(rows);a.output.mkdir(parents=True,exist_ok=True);d.to_csv(a.output/'final_wav_per_file.csv',index=False);s=ns['summarize_scores'](d);s.to_csv(a.output/'final_wav_summary.csv',index=False);print(s.to_string(index=False));assert len(d)==len(manifest) and s.pesq_failed.sum()==0
if __name__=='__main__':main()
