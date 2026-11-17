"""Count detections during isolated noise and speech rises in the development set."""
from experiment import ROOT, TA
from scipy import signal, ndimage
import numpy as np
import pandas as pd

def power(x):
    return np.abs(signal.stft(x, fs=16000, nperseg=1024, noverlap=768)[2])**2

def changes(p):
    recent = ndimage.uniform_filter1d(p, 5, axis=1)
    previous = np.roll(recent, 8, axis=1)
    return (recent + 1e-18) / (previous + 1e-18)

def main():
    records = []
    for row in pd.read_csv(ROOT/'r6/train_split.csv').query("stage=='dev'").to_dict('records'):
        clean, noisy = TA['read_pair'](ROOT/'noise_reduction_challenge', row)
        mixture = power(noisy)
        rising = changes(mixture) > 2
        neighbors = ndimage.uniform_filter1d(rising.astype(float), 9, axis=0)
        detected = ndimage.minimum_filter1d(neighbors, 3, axis=1) >= 0.67
        speech = power(clean)
        noise = power(noisy-clean)
        s, n = changes(speech), changes(noise)
        usable = mixture > mixture.max()*1e-6
        usable[:, :14] = False
        usable[:, -6:] = False
        for label, mask in [('noise',(n>2)&(s<1.25)&(noise>speech)), ('speech',(s>2)&(n<1.25)&(speech>noise))]:
            eligible = mask & usable
            records.append((row['segment_id'],label,eligible.sum(),(eligible & detected).sum()))
    data = pd.DataFrame(records,columns=['id','source','bins','detections'])
    data.to_csv(ROOT/'bandwidth_round/diagnostic.csv',index=False)
    print(data.groupby('source')[['bins','detections']].sum())
if __name__=='__main__':
    main()
