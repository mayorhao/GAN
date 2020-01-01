import scipy.signal as signal
# data_dir="/home/fanjiahao/sleep-dataset/edf-two-in-one"
# file_list=os.listdir(data_dir)
# for idx,file in enumerate(file_list):
#     data=scio.loadmat(os.path.join(file,"stages.mat"))
#     eeg_data=data["x"]
#     for i in range(len(eeg_data)):
#         resampled_signal=signal.resample_poly(eeg_data[i],up=128,down=125,axis=0)
#         if i==0:
#             resampled_signals=resampled_signal[:]
#             X=np.linspace(0,100,100)
#             plt.plot(X,eeg_data[i][0:100],'b.-',X,resampled_signal[0:100],'r.-')
#             plt.legend(['raw','resampled'])
#             plt.show()
#         else:
#             resampled_signals=np.vstack((resampled_signals,resampled_signal))
#         print("{} done".format(i))

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
import gc
# for estar
# sys.path.append("/home/STOREAGE/fanjiahao/code/GAN")
# torch.cuda.set_device(2)
# for dell
sys.path.append("/home/fanjiahao/GAN/GAN")
from eeggan.examples.shallow_conv_lin.model import Generator,Discriminator
# 读取参数
parser=argparse.ArgumentParser()
parser.add_argument("--stage",type=str,default="REM",help="determin which stege to be trained")
parser.add_argument("--GPU",type=int,default=0,help="the GPU device id")
parser.add_argument("--fold_idx",type=int,default=0,help="folds number")
parser.add_argument("--seed",type=int,default=54645,help="random seed")


args=parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
torch.cuda.set_device(args.GPU)
N_STAGE=args.stage
n_batch=64
SYNTHESIS_SUM=n_batch*800
fold_idx=args.fold_idx
GPU=args.GPU
seed=args.seed
np.random.seed(seed)
task_ind=seed
N_BLOCK=8
print(("begin to generate fake data of : {},GPU:{},fold_idx:{}").format(N_STAGE,GPU,fold_idx))
gen_model_path='./models/5-fold-8-layer/k_fold_{}/{}/Progressive0.gen'.format(fold_idx,N_STAGE)
disc_model_path='./models/5-fold-8-layer/k_fold_{}/{}/Progressive0.disc'.format(fold_idx,N_STAGE)

def readRealData(fold_idx,n_stage):
    data_path='../../../data/stages-new'
    data_list = np.load(os.path.join("./", "k-fold-plan", "plan.npz"), allow_pickle=True)["plan"]
    data_list=np.array(data_list)
    subject_list=np.delete(data_list,fold_idx,0)
    subject_list_temp=[]
    for row in subject_list:
        subject_list_temp.extend(row.tolist())
    ## 5-fold to divide dataset
    for i  in range(len(subject_list_temp)):
        print("load traning data from {}".format(os.path.join(subject_list_temp[i], 'stages.mat')))
        data_tmp = scio.loadmat(os.path.join(data_path,subject_list_temp[i], 'stages.mat'))
        if n_stage in data_tmp:
            eeg_data_tmp = data_tmp[n_stage]
            try:
                data_set = np.vstack((data_set, eeg_data_tmp))
            except NameError:
                data_set = eeg_data_tmp[:, :]
    train = np.expand_dims(data_set, axis=1)
    # shape is N*C*3840*1
    train = np.reshape(train, (train.shape[0], train.shape[1], train.shape[2], 1))
    ## 標準化
    train = train - train.mean()
    train = train / train.std()
    train = train / np.abs(train).max()
    return train
def load_disc():
    discriminator=Discriminator(1)
    discriminator.model.alpha = 1
    discriminator = discriminator.cuda()
    discriminator.train_init(alpha=0.001, betas=(0., 0.99), eps_center=0.001,
                             one_sided_penalty=True, distance_weighting=True)
    discriminator.load_model(os.path.join(disc_model_path))
    discriminator.model.cur_block=0
    return discriminator

def main():
    n_z = 200
    batch=64

    losses_fake=[]
    losses_real=[]
    train_data=readRealData(fold_idx,N_STAGE)
    # real_data=Variable(torch.from_numpy(real_data),requires_grad=False).cuda()
    rng = np.random.RandomState(task_ind)
    generator = Generator(1, n_z)

    generator.model.alpha=1
    generator = generator
    # load model start
    generator.train_init(alpha=0.001, betas=(0., 0.99))
    generator.load_model(os.path.join(gen_model_path))

    generator.model.cur_block=N_BLOCK-1
    # load_model end
    iter=int(SYNTHESIS_SUM/batch)
    discriminator=load_disc()
    # loss_real = discriminator(real_data).detach().cpu().numpy()
    # losses_real.append(loss_real)
    print("gc status:{}".format(gc.isenabled()))
    for i in range(iter):
        z_rng=np.random.RandomState(i)
        z_vars_im = z_rng.normal(0, 1, size=(n_batch, n_z)).astype(np.float32)
        # see 1000*200
        z_vars = Variable(torch.from_numpy(z_vars_im),requires_grad=False)
        batch_fake = Variable(generator(z_vars).data,requires_grad=True).cuda()
        # see out of memory ,what is going on
        loss_fake=discriminator(batch_fake)
        mean_loss=loss_fake.mean()
        mean_loss.backward()
        losses_fake.append(loss_fake.detach().cpu().numpy())
        # del loss_fake
        batch_fake=batch_fake.cpu().detach().numpy()
        real_indices=np.arange(len(train_data))
        np.random.shuffle(real_indices)
        batch_real=train_data[real_indices[0:64]]
        batch_real=batch_real.astype(np.float32)
        batch_real = Variable(torch.from_numpy(batch_real), requires_grad=False).cuda()
        loss_real=discriminator(batch_real).detach().cpu().numpy()
        losses_real.append(loss_real)
        batch_real=batch_real.detach().cpu().numpy()
        batch_fake=np.squeeze(batch_fake)
        batch_real=np.squeeze(batch_real)
        # batch_fake = signal.resample_poly(batch_fake, up=125, down=128, axis=1)
        # batch_real=signal.resample_poly(batch_real,up=125, down=128, axis=1)
        try:
            fake_data_list=np.vstack((fake_data_list, batch_fake))
            real_data_list=np.vstack((real_data_list,batch_real))
        except NameError:
            fake_data_list= batch_fake[:, :]
            real_data_list=batch_real[:,:]
        print("progress:{}/{}".format(i+1,iter))

    #store
    fig_path=os.path.join('./analysis/pick-8-layer-ss3/fold-{}'.format(fold_idx),N_STAGE)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    filename=os.path.join(fig_path,'N1_GAN.mat')
    scio.savemat(filename,{
        'x':fake_data_list,
        'real':real_data_list,
        'losses_real':np.array(losses_real),
        'losses_fake':np.array(losses_fake)
    })
    #end store

if __name__ == '__main__':
    main()

