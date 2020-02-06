import matplotlib
import numpy as np
from torch.autograd import Variable
import torch
import os
import glob
import scipy.io as scio
import matplotlib.pyplot as plt
import sys
import argparse
# for estar
# sys.path.append("/home/STOREAGE/fanjiahao/GAN")
# for dell
parser=argparse.ArgumentParser()
parser.add_argument("--stage",type=str,default="WAKE",help="determin which stege to be trained")
parser.add_argument("--GPU",type=int,default=3,help="the GPU device id")
parser.add_argument("--fold_idx",type=int,default=1,help="folds number")
parser.add_argument("--seed",type=int,default=3999,help="random seed")
parser.add_argument("--best_index",type=int,default=474,help="random seed")


args=parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
torch.cuda.set_device(args.GPU)
sys.path.append("/home/fanjiahao/GAN/GAN")
from eeggan.examples.conv_lin.model import Generator,Discriminator
n_batch=64
SYNTHESIS_SUM=n_batch*400
fold_idx=args.fold_idx
stage=args.stage
BEST_INDEX=args.best_index
model_path='./evolution/models/MASS-5_fold/k_fold_{}/{}/best_model_{}.gen'.format(fold_idx,stage,BEST_INDEX)
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
    task_ind = args.seed
    rng = np.random.RandomState(task_ind)
    generator = Generator(1, n_z)
    # load model start
    generator.train_init(alpha=0.01, betas=(0., 0.99))
    generator.load_model(os.path.join(model_path))
    generator.model.cur_block=5
    # load_model end
    batch=64
    iter=int(SYNTHESIS_SUM/batch)
    for i in range(iter):
        z_rng=np.random.RandomState(i)
        z_vars_im = z_rng.normal(0, 1, size=(batch, n_z)).astype(np.float32)  # see 1000*200
        z_vars = Variable(torch.from_numpy(z_vars_im),requires_grad=False)
        batch_fake = generator(z_vars).data.cpu().numpy()
        print("{}/{} done".format(i+1,iter))
        try:
            out=np.vstack((out,batch_fake))
        except NameError:
            out=batch_fake[:,:]
    #store
    fig_path=os.path.join('./evolution/analysis/5-fold-MASS/fold-{}/'.format(fold_idx))
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    filename=os.path.join(fig_path,'{}_GAN.mat'.format(stage))
    if not os.path.exists(filename):
        scio.savemat(filename,{
            'x':np.squeeze(out)
        })
    #end store

if __name__ == '__main__':
    main()

