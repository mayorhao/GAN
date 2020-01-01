import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
# Fixing random state for reproducibility
np.random.seed(19680801)


t = np.arange(0.0, 30, 30/3840)
fake=scio.loadmat("/home/fanjiahao/GAN/GAN/eeggan/examples/conv_lin/analysis_30K/N1/synthesis.mat")["x"][0]
real=scio.loadmat("/home/fanjiahao/GAN/GAN/data/stages-new/01-03-0001.mat/stages.mat")["N1"][0]
NFFT = 1024  # the length of the windowing segments
Fs = 128 # the sampling frequency

fig, (ax1, ax2) = plt.subplots(nrows=2)
ax1.plot(t, real)
Pxx, freqs, bins, im = ax2.specgram(real, NFFT=NFFT, Fs=Fs, noverlap=900)
# The `specgram` method returns 4 objects. They are:
# - Pxx: the periodogram
# - freqs: the frequency vector
# - bins: the centers of the time bins
# - im: the matplotlib.image.AxesImage instance representing the data in the plot
plt.show()