import scipy.io as scio
import matplotlib.pyplot as plt
import os
import numpy as np
import glob
# from torch.autograd import Variable
# import
n_stage="WAKE"
fig_path=os.path.join('../analysis_30K',n_stage)
# filtered_signal=scio.loadmat(os.path.join('./',n_stage,'synthesis_filtered.mat'))['x']
raw_signal=scio.loadmat(os.path.join('../analysis_30K',n_stage,'N1_GAN.mat'))['x']
plt.figure()
x = np.linspace(0, 30,3840)
y = raw_signal[0]
plt.plot(x, y, label="", color="red")
plt.xlabel('time')
plt.ylabel('amtipute')
plt.savefig(os.path.join(fig_path,'1_channel_time.png'))
plt.close()

# freqency domin compare
# fs=128
# # fft = np.fft.rfft(data.numpy(), axis=2)
# # ams = np.abs(fft).mean(axis=3).mean(axis=0).squeeze()
# # show_rates = np.fft.rfftfreq(data.numpy().shape[2], d=1 / sampleRate)
# fft_raw_signal=np.fft.rfft(z)
# ams_raw_sngal=np.abs(fft_raw_signal).squeeze()
# fft_filtered_signal=np.fft.rfft(y)
# ams_filtered_signal=np.abs(fft_filtered_signal).squeeze()
#
# fft_x=np.linspace(0,60,1921)
# plt.figure()
# plt.plot(fft_x,ams_filtered_signal,label="filtered_signal")
# # plt.plot(fft_x,ams_raw_sngal,label="raw signal")
# plt.legend()
# plt.show()
# plt.savefig(os.path.join('./',n_stage,'filtered_vs_unfilterd_fft.png'))
# freqency domin compare end
def readRealData(n_stage):
    # for estar
    # data_path='/home/STOREAGE/fanjiahao/GAN/data/stages-c3-128/*.mat'
    # for dell
    data_path = '/home/fanjiahao/GAN/extractSleepData/output/stages-c3-128/*.mat'
    data_list = glob.glob(os.path.join(data_path))
    data_list.sort()
    for i in range(len(data_list)):
        data_tmp = scio.loadmat(os.path.join(data_list[i], 'stages.mat'))
        if n_stage in data_tmp:
            eeg_data_tmp = data_tmp[n_stage]
            try:
                data_set = np.vstack((data_set, eeg_data_tmp))
            except NameError:
                data_set = eeg_data_tmp[:, :]

    train = np.expand_dims(data_set, axis=1)
    train = np.reshape(train, (train.shape[0], train.shape[1], train.shape[2], 1))
    # train = train[:,:,:,None]
    train = train - train.mean()
    train = train / train.std()
    train = train / np.abs(train).max()
    return train

def fftTransform(data,sampleRate=128.):
    fft = np.fft.rfft(data, axis=1)
    ams = np.abs(fft).mean(axis=0).squeeze()
    return ams

fig_path=os.path.join('../analysis_30K',n_stage)
readRealData(n_stage)
batch_real = np.squeeze(readRealData(n_stage))
batch_fake=raw_signal
fake_fft_ams  = fftTransform(batch_fake)
real_fft_ams = fftTransform(batch_real)
plt.figure()
freqs_tmp = np.fft.rfftfreq((batch_real.shape[1]), d=1 / 128.)
plt.plot(freqs_tmp, np.log(fake_fft_ams), label='Fake')
plt.plot(freqs_tmp, np.log(real_fft_ams), label='Real')
plt.title('Frequency Spektrum')
plt.xlabel('Hz')
plt.legend()
if not os.path.exists(fig_path):
    os.makedirs(fig_path)
plt.savefig(os.path.join(fig_path, 'fft.png'))
plt.show()
# plt.close()

for i in range(10):
    plt.subplot(10, 1, i + 1)
    plt.plot(batch_fake[i].squeeze())
    plt.xticks((), ())
    plt.yticks((), ())
plt.subplots_adjust(hspace=0)
plt.savefig(os.path.join(os.path.join(fig_path, 'timeseries')))
plt.close()


