from scipy.signal import butter, lfilter
import scipy.io as scio

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def run():
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    n_stage='REM'
    raw_signal_path=os.path.join('./analysis',n_stage,'N1_GAN.mat')
    fitered_singal_path=os.path.join('./analysis',n_stage,'synthesis_filtered.mat')
    raw_signals=scio.loadmat(raw_signal_path)['x']
    fs = 128.0
    lowcut = 0.2
    highcut = 35.0
    filtered_signals=np.apply_along_axis(lambda x:butter_bandpass_filter(x,lowcut,highcut,fs,order=6),axis=1,arr=raw_signals)
    scio.savemat(fitered_singal_path,{
        'x':filtered_signals
    })

run()