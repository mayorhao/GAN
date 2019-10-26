#%load_ext autoreload
#%autoreload 2
import argparse
import os
import joblib
import sys
# for estar
# sys.path.append("/home/STOREAGE/fanjiahao/GAN")
# for dell
sys.path.append("/home/fanjiahao/GAN/GAN")
from braindecode.datautil.iterators import get_balanced_batches
from eeggan.examples.conv_lin.model import Generator,Discriminator
from eeggan.util import weight_filler
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.io as scio
import glob
import time
#设置图像渲染方式
plt.switch_backend('agg')
#设置GPU以同步方式运算,默认是异步
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.enabled=True
# 对于固定的CNN结构，先测试选出最优效率的优化算法
torch.backends.cudnn.benchmark=True
# see????
# 读取参数
parser=argparse.ArgumentParser()
parser.add_argument("--stage",type=str,default="N1",help="determin which stege to be trained")
parser.add_argument("--task_id",type=int,default=0,help="the number to generate random seed")
parser.add_argument("--GPU",type=int,default=0,help="the GPU device id")
parser.add_argument("--i_block_tmp",type=int,default=0,help="which block to start with?")
parser.add_argument("--i_epoch_tmp",type=int,default=0,help="which epoch to start with?")
parser.add_argument("--reuse",type=bool,default=False,help="Do you need to resuse the models")

args=parser.parse_args()
# 读取参数 end
## 固定参数
n_critic = 5
n_batch = 64
jobid = 0
n_z = 200
lr = 0.001
n_blocks = 6
rampup = 200 # 这里resamle 跟随 epoch 数目来增加和减少
block_epochs = [200,200,400,400,400,400]
##  固定参数 end

## 可配置参数
n_stage=args.stage
# FIXME original task_ind is 0
task_ind = args.task_id#subj_ind
# FIXME allocate specific GPu
torch.cuda.set_device(args.GPU)
print("Begin to train stage:{} ,GPU:{},task_id:{},block_tmp:{},epoch_tmp:{}".format(n_stage,args.GPU,task_ind,args.i_block_tmp,args.i_epoch_tmp))
## 可配置参数 end

## 设置随机种子
np.random.seed(task_ind)
torch.manual_seed(task_ind)
torch.cuda.manual_seed_all(task_ind)
random.seed(task_ind)
rng = np.random.RandomState(task_ind)
## 设置随机种子 end

## 设置数据路径
data = os.path.join('/home/fanjiahao/GAN/extractSleepData/output/stages-c3-128/01-03-0064.mat/stages.mat')
#for estar
# data_path='/home/STOREAGE/fanjiahao/GAN/data/stages-c3-128/*.mat'
#for dell
data_path='/home/fanjiahao/GAN/extractSleepData/output/stage-total/total_stage.mat'
raw_data=scio.loadmat(data_path)
data_set=raw_data[n_stage]
train=np.expand_dims(data_set,axis=1)
# shape is N*C*3840*1
train=np.reshape(train,(train.shape[0],train.shape[1],train.shape[2],1))


modelpath = './models/formal'+'_'+n_stage
modelname = 'Progressive%s'
if not os.path.exists(modelpath):
    os.makedirs(modelpath)


generator = Generator(1,n_z)
discriminator = Discriminator(1)

generator.train_init(alpha=lr,betas=(0.,0.99))
discriminator.train_init(alpha=lr,betas=(0.,0.99),eps_center=0.001,
                        one_sided_penalty=True,distance_weighting=True)
generator = generator.apply(weight_filler)
discriminator = discriminator.apply(weight_filler)
#FIXME i_block_tmp and i_epoch_tmp are original 0
i_block_tmp = args.i_block_tmp
i_epoch_tmp = args.i_epoch_tmp
generator.model.cur_block = i_block_tmp
discriminator.model.cur_block = n_blocks-1-i_block_tmp
fade_alpha = 1.
generator.model.alpha = fade_alpha
discriminator.model.alpha = fade_alpha

generator = generator.cuda()
discriminator = discriminator.cuda()
losses_d = []
losses_g = []
#fixme load models start
if args.reuse:
    print("start load generator model...,from {}".format(os.path.join(modelpath,modelname%jobid+'.gen')))
    generator.load_model(os.path.join(modelpath,modelname%jobid+'.gen'))
    print("start load criminator model...,from {}".format(os.path.join(modelpath,modelname%jobid+'.disc')))
    discriminator.load_model(os.path.join(modelpath,modelname%jobid+'.disc'))
    i_epoch,loss_d,loss_g=joblib.load(os.path.join(modelpath,modelname%jobid+'_.data'))
# #fixme load models end

# summary(generator,(1,1,200,1))

generator.train()
discriminator.train()


i_epoch = 0
z_vars_im = rng.normal(0,1,size=(1000,n_z)).astype(np.float32) #see 1000*200
start_time=time.time()
last_epoch_hundred_time=time.time()
for i_block in range(i_block_tmp,n_blocks):
    print("=========================================")
    print("start to train block : {}".format(i_block))
    c = 0
    # train_tmp down_sample to fit the current cric block
    train_tmp = discriminator.model.downsample_to_block(Variable(torch.from_numpy(train).cuda(),requires_grad=False),discriminator.model.cur_block).data.cpu()

    for i_epoch in range(i_epoch_tmp,block_epochs[i_block]):
        i_epoch_tmp = 0

        if fade_alpha<1:
            fade_alpha += 1./rampup
            generator.model.alpha = fade_alpha
            discriminator.model.alpha = fade_alpha

        batches = get_balanced_batches(train.shape[0], rng, True, batch_size=n_batch)
        iters = int(len(batches)/n_critic)

        for it in range(iters):
            for i_critic in range(n_critic):
                train_batches = train_tmp[batches[it*n_critic+i_critic]]
                batch_real = Variable(train_batches,requires_grad=True).cuda()
#see indicate the batchsize
                z_vars = rng.normal(0,1,size=(len(batches[it*n_critic+i_critic]),n_z)).astype(np.float32)
                z_vars = Variable(torch.from_numpy(z_vars),requires_grad=False).cuda()
                batch_fake = Variable(generator(z_vars).data,requires_grad=True).cuda()
            #see this line error occurs
                loss_d = discriminator.train_batch(batch_real,batch_fake)
                assert np.all(np.isfinite(loss_d))
            z_vars = rng.normal(0,1,size=(n_batch,n_z)).astype(np.float32)
            z_vars = Variable(torch.from_numpy(z_vars),requires_grad=True).cuda()
            loss_g = generator.train_batch(z_vars,discriminator)

        losses_d.append(loss_d)
        losses_g.append(loss_g)


        if (i_epoch+1)%100 == 0:

            generator.eval()
            discriminator.eval()
            epoch_hundred_time=time.time()
            print('Block: %d Epoch: %d   Loss_F: %.3f   Loss_R: %.3f   Penalty: %.4f   Loss_G: %.3f  Duration:%.2f min'%(i_block,i_epoch,loss_d[0],loss_d[1],loss_d[2],loss_g,(epoch_hundred_time-last_epoch_hundred_time)/60))
            last_epoch_hundred_time=epoch_hundred_time
            joblib.dump((i_epoch,losses_d,losses_g),os.path.join(modelpath,modelname%jobid+'_.data'),compress=True)
            joblib.dump((i_epoch,losses_d,losses_g),os.path.join(modelpath,modelname%jobid+'_%d.data'%i_epoch),compress=True)
            #joblib.dump((n_epochs,n_z,n_critic,batch_size,lr),os.path.join(modelpath,modelname%jobid+'_%d.params'%i_epoch),compress=True)

            freqs_tmp = np.fft.rfftfreq(train_tmp.numpy().shape[2],d=1/(128./np.power(2,n_blocks-1-i_block)))

            train_fft = np.fft.rfft(train_tmp.numpy(),axis=2)
            train_amps = np.abs(train_fft).mean(axis=3).mean(axis=0).squeeze()


            z_vars = Variable(torch.from_numpy(z_vars_im),requires_grad=False).cuda()
            batch_fake = generator(z_vars)
            fake_fft = np.fft.rfft(batch_fake.data.cpu().numpy(),axis=2)
            # here are the average of train batches
            fake_amps = np.abs(fake_fft).mean(axis=3).mean(axis=0).squeeze()

            plt.figure()
            plt.plot(freqs_tmp,np.log(fake_amps),label='Fake')
            plt.plot(freqs_tmp,np.log(train_amps),label='Real')
            plt.title('Frequency Spektrum')
            plt.xlabel('Hz')
            plt.legend()
            plt.savefig(os.path.join(modelpath,modelname%jobid+'_fft_%d_%d.png'%(i_block,i_epoch)))
            plt.close()

            batch_fake = batch_fake.data.cpu().numpy()
            plt.figure(figsize=(10,10))
            for i in range(10):
                plt.subplot(10,1,i+1)
                plt.plot(batch_fake[i].squeeze())
                plt.xticks((),())
                plt.yticks((),())
            plt.subplots_adjust(hspace=0)
            plt.savefig(os.path.join(modelpath,modelname%jobid+'_fakes_%d_%d.png'%(i_block,i_epoch)))
            plt.close()

            discriminator.save_model(os.path.join(modelpath,modelname%jobid+'.disc'))
            generator.save_model(os.path.join(modelpath,modelname%jobid+'.gen'))

            plt.figure(figsize=(10,15))
            plt.subplot(3,2,1)
            plt.plot(np.asarray(losses_d)[:,0],label='Loss Real')
            plt.plot(np.asarray(losses_d)[:,1],label='Loss Fake')
            plt.title('Losses Discriminator')
            plt.legend()
            plt.subplot(3,2,2)
            plt.plot(np.asarray(losses_d)[:,0]+np.asarray(losses_d)[:,1]+np.asarray(losses_d)[:,2],label='Loss')
            plt.title('Loss Discriminator')
            plt.legend()
            plt.subplot(3,2,3)
            plt.plot(np.asarray(losses_d)[:,2],label='Penalty Loss')
            plt.title('Penalty')
            plt.legend()
            plt.subplot(3,2,4)
            plt.plot(-np.asarray(losses_d)[:,0]-np.asarray(losses_d)[:,1],label='Wasserstein Distance')
            plt.title('Wasserstein Distance')
            plt.legend()
            plt.subplot(3,2,5)
            plt.plot(np.asarray(losses_g),label='Loss Generator')
            plt.title('Loss Generator')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(modelpath,modelname%jobid+'_losses.png'))
            plt.close()

            generator.train()
            discriminator.train()


    fade_alpha = 0.
    generator.model.cur_block += 1
    discriminator.model.cur_block -= 1
    print("=========================================")
    print("end to train block :{}".format(i_block))
end_time=time.time()
print("time in total: %.2f"%((end_time-start_time)/60),"min")
#TODO:fix all volatile=true issue