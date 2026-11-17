from pathlib import Path
import pandas as pd

root=Path(__file__).resolve().parent
d=pd.read_csv(root/'results/baseline/final_wav_per_file.csv')
manifest=pd.read_csv(root/'validation_manifest.csv')
d=d.merge(manifest[['segment_id','target_snr_db']],on='segment_id',validate='one_to_one')
d['speaker']=d.segment_id.str.split('_').str[0]
for keys in [['noise_category'],['target_snr_db'],['speaker'],['noise_category','target_snr_db','speaker']]:
    summary=d.groupby(keys).agg(n=('pesq','size'),pesq=('pesq','mean'),snr_db=('snr_db','mean'),
        delta_pesq=('delta_pesq','mean'),delta_snr_db=('delta_snr_db','mean'))
    summary.sort_values('pesq').to_csv(root/'results'/('baseline_by_'+'_'.join(keys)+'.csv'))
d.nsmallest(20,'pesq').to_csv(root/'results/baseline_lowest20.csv',index=False)
