#%load_ext autoreload
#%autoreload 2
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
plt.switch_backend('agg')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.enabled=True
torch.backends.cudnn.benchmark=True

n_critic = 5
#fixme 64-->8
n_batch = 64
input_length = 768
jobid = 0
n_z = 200
lr = 0.001
n_blocks = 6
rampup = 2000.
block_epochs = [2000,4000,4000,4000,4000,4000]
n_stage='WAKE'
# FIXME original task_ind is 0
task_ind = 4#subj_ind
# FIXME allocate specific GPu
torch.cuda.set_device(0)
# #subj_ind = 9
np.random.seed(task_ind)
torch.manual_seed(task_ind)
torch.cuda.manual_seed_all(task_ind)
random.seed(task_ind)
rng = np.random.RandomState(task_ind)
# data = os.path.join('/home/fanjiahao/GAN/extractSleepData/output/stages-c3-128/01-03-0064.mat/stages.mat')
#for estar
# data_path='/home/STOREAGE/fanjiahao/GAN/data/stages-c3-128/*.mat'
#for dell
data_path='/home/fanjiahao/GAN/extractSleepData/output/stages-c3-128/*.mat'
data_list=glob.glob(os.path.join(data_path))
data_list.sort()
for i in range(len(data_list)):
    data_tmp = scio.loadmat(os.path.join(data_list[i], 'stages.mat'))
    if n_stage in data_tmp:
        eeg_data_tmp=data_tmp[n_stage]
        try:
            data_set=np.vstack((data_set, eeg_data_tmp))
        except NameError:
            data_set=eeg_data_tmp[:,:]



# data_mat=scio.loadmat('/home/STOREAGE/fanjiahao/GAN/eeggan/stages.mat')
# EEG_data=data_mat['N1']
# train_set = EEG_data['train_set']
# test_set = EEG_data['test_set']
# train = np.concatenate((train_set.X,test_set.X))
# target = np.concatenate((train_set.y,test_set.y))
train=np.expand_dims(data_set,axis=1)
train=np.reshape(train,(train.shape[0],train.shape[1],train.shape[2],1))
# train = train[:,:,:,None]
train = train-train.mean()
train = train/train.std()
train = train/np.abs(train).max()
# FIXME leave one-hot
# target_onehot = np.zeros((target.shape[0],2))
# target_onehot[:,target] = 1


modelpath = './models/GAN_debug/conv_cub_'+str(task_ind)+'_'+n_stage
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
i_block_tmp = 0
i_epoch_tmp = 0
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
# generator.load_model(os.path.join(modelpath,modelname%jobid+'.gen'))
# discriminator.load_model(os.path.join(modelpath,modelname%jobid+'.disc'))
# i_block_tmp=5
# i_epoch_tmp=2500
# generator.model.cur_block = i_block_tmp
# discriminator.model.cur_block = n_blocks-1-i_block_tmp
# i_epoch,loss_d,loss_g=joblib.load(os.path.join(modelpath,modelname%jobid+'_.data'))
# #fixme load models end

# summary(generator,(1,1,200,1))

generator.train()
discriminator.train()


i_epoch = 0
z_vars_im = rng.normal(0,1,size=(1000,n_z)).astype(np.float32) #see 1000*200
start_time=time.time()
for i_block in range(i_block_tmp,n_blocks):
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


        if i_epoch%100 == 0:
            generator.eval()
            discriminator.eval()

            print('Epoch: %d   Loss_F: %.3f   Loss_R: %.3f   Penalty: %.4f   Loss_G: %.3f'%(i_epoch,loss_d[0],loss_d[1],loss_d[2],loss_g))
            joblib.dump((i_epoch,losses_d,losses_g),os.path.join(modelpath,modelname%jobid+'_.data'),compress=True)
            joblib.dump((i_epoch,losses_d,losses_g),os.path.join(modelpath,modelname%jobid+'_%d.data'%i_epoch),compress=True)
            #joblib.dump((n_epochs,n_z,n_critic,batch_size,lr),os.path.join(modelpath,modelname%jobid+'_%d.params'%i_epoch),compress=True)

            freqs_tmp = np.fft.rfftfreq(train_tmp.numpy().shape[2],d=1/(250./np.power(2,n_blocks-1-i_block)))

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
end_time=time.time()
print("time in total",int((start_time-end_time)/1000*60*60),"min")
#TODO:fix all volatile=true issue