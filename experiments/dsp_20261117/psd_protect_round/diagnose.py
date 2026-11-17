"""Clean-referenced diagnostic at the raw 1024-point first-pass input only."""
from experiment import ROOT, TA, noisy_estimators
from bench import BASE
from enhance_gain import GainConfig, init_noise_psd
from scipy import signal, ndimage
import numpy as np
import pandas as pd


def main():
    records = []
    rows = pd.read_csv(ROOT / 'r6/train_split.csv').query("stage == 'dev'")
    for row in rows.to_dict('records'):
        clean, noisy = TA['read_pair'](ROOT / 'noise_reduction_challenge', row)
        options = dict(fs=16000, nperseg=1024, noverlap=768)
        _, times, spectrum = signal.stft(noisy, **options)
        power = abs(spectrum)**2
        noise_power = ndimage.gaussian_filter(abs(signal.stft(noisy-clean, **options)[2])**2, (1.5, 1))
        speech_power = ndimage.gaussian_filter(abs(signal.stft(clean, **options)[2])**2, (1.5, 1))
        centers = np.rint(times*16000).astype(int)
        full = np.flatnonzero((centers >= 512) & (centers+512 <= len(noisy)))
        config = GainConfig(n_fft=1024, hop=256, init_quantile=BASE.init_quantile, init_dof=BASE.init_dof, init_shrink_beta=BASE.init_shrink_beta, init_adaptive_dof=BASE.init_adaptive_dof)
        initial = init_noise_psd(power, config)
        frequency = np.fft.rfftfreq(1024, 1/16000)
        initial *= BASE.psd_low_scale**(1/(1+(frequency/700)**2))
        initial *= BASE.psd_high_scale**((frequency/2500)**2/(1+(frequency/2500)**2))
        estimated = noisy_estimators.estimate(power, initial, full, 'local_15_0.5') * BASE.noise_scale
        ratio_db = 10*np.log10(np.maximum(estimated, 1e-18)/np.maximum(noise_power, 1e-18))
        usable = (noise_power > noise_power.max()*1e-6)
        usable[:, np.setdiff1d(np.arange(power.shape[1]), full)] = False
        for group, mask in [('speech_dominant', speech_power > 2*noise_power), ('noise_dominant', speech_power < .1*noise_power)]:
            values = ratio_db[usable & mask]
            if values.size:
                records.append(dict(id=row['segment_id'], group=group, median_psd_error_db=np.median(values), overestimate_fraction=np.mean(values > 3), bins=len(values)))
    result = pd.DataFrame(records)
    result.to_csv(ROOT / 'psd_protect_round/diagnostic.csv', index=False)
    print(result.groupby('group')[['median_psd_error_db','overestimate_fraction']].mean().to_string())


if __name__ == '__main__':
    main()
