from pathlib import Path
import pandas as pd
import numpy as np
root=Path(__file__).resolve().parent/'results'
data=pd.concat([pd.read_csv(f) for stage in ('dev','confirm','extra','validation') for f in root.glob(stage+'_*.csv')],ignore_index=True)
assert not data.duplicated(['stage','mode','id']).any() and np.isfinite(data[['pesq','snr']]).all().all()
base=data.query("mode=='v5res_500_0.32_20'")[['stage','id','pesq','snr']].rename(columns={'pesq':'ref_pesq','snr':'ref_snr'})
df=data.merge(base,on=['stage','id'],validate='many_to_one')
df['delta_pesq']=df.pesq-df.ref_pesq;df['delta_snr']=df.snr-df.ref_snr;df['wins']=df.delta_pesq>0;df['speaker']=df.id.str.split('_').str[0]
df['diagnostic']=df['mode'].str.startswith('oracle_')
df.to_csv(root/'all_scores.csv',index=False)
df.groupby(['stage','diagnostic','mode']).agg(n=('id','size'),pesq=('pesq','mean'),snr=('snr','mean'),wins=('wins','sum'),delta=('delta_pesq','mean')).to_csv(root/'summary.csv')
v=df[df.stage=='validation']
for col in ['noise','target','speaker']:
 v.groupby(['mode',col]).agg(n=('id','size'),pesq=('pesq','mean'),delta=('delta_pesq','mean'),snr=('snr','mean')).to_csv(root/('by_'+col+'.csv'))
print(v.groupby('mode').agg(pesq=('pesq','mean'),snr=('snr','mean'),wins=('wins','sum')))
print('dev methods',df.query("stage=='dev' and diagnostic==False")['mode'].nunique())
