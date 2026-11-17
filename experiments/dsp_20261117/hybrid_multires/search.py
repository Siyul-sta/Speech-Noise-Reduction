from tune import *
base=json.loads(Path(__file__).with_name('starting_config.json').read_text())['config']; C.clear()
def cand(n,**kw):add(n,**(base|kw))
cand('control');cand('low07',psd_low_scale=.7);cand('low15',psd_low_scale=1.5);cand('high07',psd_high_scale=.7);cand('high15',psd_high_scale=1.5);cand('protect05',speech_protect=.5);cand('protect1',speech_protect=1.);cand('power12',gain_power=1.2);cand('power085',gain_power=.85);cand('log_smooth',log_smooth=True);cand('fft2048',n_fft=2048,hop=256);cand('fft512',n_fft=512,hop=128,smooth_time=4.,smooth_freq=.5);cand('hrnr1',harmonic_weight=1.);cand('hrnr3',harmonic_weight=3.);cand('t3',smooth_time=3.);cand('t4',smooth_time=4.);cand('f05',smooth_freq=.5);cand('f2',smooth_freq=2.);cand('ddmix02',dd_mix=.2);cand('ddmix06',dd_mix=.6)
if __name__=='__main__':
 m=load_manifest(ROOT,'validation');ids=pd.read_csv(Path(__file__).with_name('screen_ids.csv')).segment_id;screen=m[m.segment_id.isin(ids)];s=run(screen,list(C),'screen');top=s[s.method!='control'].method.head(4).tolist()
 for i,n in enumerate(top):
  for q in top[i+1:]:
   a={k:v for k,v in C[n][1].items() if base.get(k)!=v};b={k:v for k,v in C[q][1].items() if base.get(k)!=v}
   if not set(a)&set(b):cand(n+'__'+q,**(a|b))
 combos=['control']+top+[n for n in C if '__' in n];s2=run(screen,combos,'combo');fin=list(dict.fromkeys(['control']+s2[s2.method!='control'].method.head(4).tolist()));(OUT/'configs.json').write_text(json.dumps({'all':C,'finalists':fin},indent=2));run(m,fin,'full');d=pd.read_csv(OUT/'full_per_file.csv');summarize_scores(d[~d.segment_id.isin(ids)]).sort_values('pesq_mean',ascending=False).to_csv(OUT/'remaining_summary.csv',index=False);print('DONE',flush=True)
