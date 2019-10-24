import matplotlib
import numpy as np
from torch.autograd import Variable
import torch
import os
import glob
import scipy.io as scio
import matplotlib.pyplot as plt
import sys
# for estar
# sys.path.append("/home/STOREAGE/fanjiahao/GAN")
# for dell
torch.cuda.set_device(3)
sys.path.append("/home/fanjiahao/GAN/GAN")
from eeggan.examples.conv_lin.model import Generator,Discriminator
MODEL_NAME="conv_linear_2_REM"


model_path='./models/GAN_debug/'+MODEL_NAME+'/Progressive0.gen'
N_STAGE='REM'
def fftTransform(data,sampleRate=128.):
    fft = np.fft.rfft(data.numpy(), axis=2)
    ams = np.abs(fft).mean(axis=3).mean(axis=0).squeeze()
    show_rates = np.fft.rfftfreq(data.numpy().shape[2], d=1 / sampleRate)
    return ams,show_rates
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
def main():
    n_z = 200
    task_ind = 0
    rng = np.random.RandomState(task_ind)
    generator = Generator(1, n_z)
    # load model start
    generator.train_init(alpha=0.01, betas=(0., 0.99))
    generator.load_model(os.path.join(model_path))
    generator.model.cur_block=5
    # generator=generator.cuda()
    # generator = generator.apply(weight_filler)
    # load_model end
    z_vars_im = rng.normal(0, 1, size=(1000, n_z)).astype(np.float32)  # see 1000*200
    z_vars = Variable(torch.from_numpy(z_vars_im),requires_grad=False)
    batch_fake = generator(z_vars)
    batch_real=readRealData(N_STAGE)
    batch_real=Variable(torch.from_numpy(batch_real),requires_grad=False).cuda()
    fake_fft_ams,fake_show_rate=fftTransform(batch_fake.data.cpu())
    real_fft_ams,real_show_rate=fftTransform(batch_real.data.cpu())
    #start to draw fig
    fig_path=os.path.join('./analysis',N_STAGE)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    freqs_tmp = np.fft.rfftfreq((batch_real.data.cpu().numpy().shape[2]), d=1 /128.)
    plt.figure()
    plt.plot(freqs_tmp, np.log(fake_fft_ams), label='Fake')
    plt.plot(freqs_tmp, np.log(real_fft_ams), label='Real')
    plt.title('Frequency Spektrum')
    plt.xlabel('Hz')
    plt.legend()
    plt.savefig(os.path.join(fig_path,'fft.png'))
    plt.close()
    #end draw fig
    # draw time fig
    batch_fake = batch_fake.data.cpu().numpy()
    batch_real=batch_real.data.cpu().numpy()
    plt.figure(figsize=(10, 10))
    for i in range(10):
        plt.subplot(10, 1, i + 1)
        plt.plot(batch_fake[i].squeeze())
        plt.xticks((), ())
        plt.yticks((), ())
    plt.subplots_adjust(hspace=0)
    plt.savefig(os.path.join(os.path.join(fig_path,'timeseries')))
    plt.close()
    # draw time fig end
    # draw temporary fig real and fake
    plt.figure()
    plt.plot(np.mean(batch_fake,axis=3).mean(axis=0).squeeze(),label="Fake")
    plt.plot(np.mean(batch_real,axis=3).mean(axis=0).squeeze(),label="True")
    plt.xlabel('time')
    plt.legend()
    plt.savefig(os.path.join(fig_path,'mean_temporary.png'))
    plt.close()
    # draw temporary fig real and fake end
    # save data
    filename=os.path.join(fig_path,'synthesis.mat')
    if not os.path.exists(filename):
        scio.savemat(filename,{
            'x':np.squeeze(batch_fake)
        })
    # save data end
    #draw filtered rawsignal time serise
    filtered_sinal=scio.loadmat(os.path.join(fig_path,'synthesis_filtered.mat'))['x']
    for i in range(10):
        plt.subplot(10, 1, i + 1)
        plt.plot(filtered_sinal[i].squeeze(),label='after')
        plt.plot(batch_fake[i].squeeze(),label='before')
        plt.xticks((), ())
        plt.yticks((), ())
        plt.legend()
    plt.subplots_adjust(hspace=0)
    plt.savefig(os.path.join(os.path.join(fig_path,'filtered_before_after_timeseries')))
    plt.close()

    #draw filtered rawsignal time serise end
    #

if __name__ == '__main__':
    main()

