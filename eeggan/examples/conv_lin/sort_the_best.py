import numpy as np
import scipy.io as scio
import torch
import argparse
import os
from mmd_rbf import mmd_rbf
from geomloss import SamplesLoss
parser=argparse.ArgumentParser()
parser.add_argument("--stage",type=str,default="WAKE",help="determin which stege to be trained")
parser.add_argument("--GPU",type=int,default=2,help="the GPU device id")
parser.add_argument("--fold_idx",type=int,default=0,help="folds number")
args=parser.parse_args()
stage=args.stage
GPU=args.GPU
stage=args.stage
fold_idx=args.fold_idx
n_batch=64
torch.cuda.set_device(GPU)
loss = SamplesLoss(loss="gaussian", p=2, blur=.05)
## older code to sort the best based on both EMD and mmd
# def readGendata(fold_idx,stage):
    # data_path="./analysis/pick-with-random-real-data/fold-{}/{}/N1_GAN.mat".format(fold_idx,stage)
    # save_path = "./analysis/sorted-pick-with-mmd/fold-{}/{}".format(fold_idx, stage)
    # saved_dic=scio.loadmat(data_path)
    # fake=saved_dic["fake"]
    # real=saved_dic["real"]
    # # losses_real=saved_dic["losses_real"]
    # # losses_fake=saved_dic["losses_fake"]
    # # losses_fake_flat=np.squeeze(losses_fake)
    # # losses_real_flat=np.squeeze(losses_real)
    # # abs_EMD=np.abs(losses_fake_flat+losses_real_flat).mean(axis=1)
    # # indices=np.argsort(abs_EMD,axis=0)
    # # s=fake.shape
    # # fake_splited=fake.reshape(int(s[0]/n_batch),n_batch,-1)
    # # sorted_fake=fake_splited[indices]
    # # s_2=sorted_fake.shape
    # # sorted_fake=sorted_fake.reshape(-1,s_2[-1])
    # # mmd_rbf
    # s=fake.shape
    # fake_splited=fake.reshape(int(s[0]/n_batch),n_batch,-1)
    # real_splited=real.reshape(int(s[0]/n_batch),n_batch,-1)
    # mmds=[]
    # for idx , batch in enumerate(real_splited):
    #     target=batch
    #     source=fake_splited[idx,:,:]
    #     target=torch.Tensor(target)
    #     source=torch.Tens   or(source)
    #     mmd=mmd_rbf(target,source)
    #     mmds.append(mmd.detach().cpu().numpy())
    #     print("finish batch {}/{}".format(idx+1,800))
    # print("begin to save data")
    # indices=np.argsort(mmds)
    # sorted_fake=fake_splited[indices]
    # sorted_fake=sorted_fake[0:400]
    # s_2=sorted_fake.shape
    # sorted_fake=sorted_fake.reshape(-1,s_2[-1])
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # scio.savemat(os.path.join(save_path,"N1_GAN.mat"),{
    #     "x":sorted_fake
    # })
    # print("done")


    # i want to sort these x files dependingn on the absolute value of losses_real-losses_fake
    # abs_EMD=losses_fake-losses_real.mean()
def readDataSet(fold_idx,stage):
    data_path = '../../../data/stages-new'
    data_list = np.load(os.path.join("./", "k-fold-plan", "plan.npz"), allow_pickle=True)["plan"]
    data_list = np.array(data_list)
    subject_list = np.delete(data_list, fold_idx, 0)
    subject_list_temp = []
    for row in subject_list:
        subject_list_temp.extend(row.tolist())
    ## 5-fold to divide dataset
    for i in range(len(subject_list_temp)):
        print("load traning data from {}".format(os.path.join(subject_list_temp[i], 'stages.mat')))
        data_tmp = scio.loadmat(os.path.join(data_path, subject_list_temp[i], 'stages.mat'))
        if stage in data_tmp:
            eeg_data_tmp = data_tmp[stage]
            try:
                data_set = np.vstack((data_set, eeg_data_tmp))
            except NameError:
                data_set = eeg_data_tmp[:, :]
    train =data_set
    # shape is N*C*3840*1
    # train = np.reshape(train, (train.shape[0], train.shape[1], train.shape[2], 1))
    ## 標準化
    train = train - train.mean()
    train = train / train.std()
    train = train / np.abs(train).max()
    return train
def _randomSample(dataset,size):
    return np.random.permutation(dataset)[:size]

def sortTheBest(train,fold_idx):
    np.random.RandomState(213213)
    batch_size=100
    mmds=[]
    data_path="./analysis/5-fold/fold-{}/{}/synthesis.mat".format(fold_idx,stage)
    save_path = "./analysis/5-fold/fold-{}/sorted/{}".format(fold_idx, stage)
    saved_dic=scio.loadmat(data_path)
    fake=saved_dic["x"]
    epoch_number=len(train)
    sample_group_number=len(fake)//batch_size
    fake_reshaped=np.reshape(fake,(sample_group_number,batch_size,3840))
    for idx in range(len(fake_reshaped)):
        real_indices = np.arange(len(train))
        np.random.shuffle(real_indices)
        real_batch = _randomSample(train,batch_size)
        real_batch = torch.Tensor(real_batch)
        fake_batch=fake_reshaped[idx]
        # fake_batch=_randomSample(train,batch_size)
        fake_batch=torch.Tensor(fake_batch)
        mmd=mmd_rbf(fake_batch,real_batch)
        mmds.append(mmd)
        print("finish batch {}/{},mmd:{}".format(idx+1,sample_group_number,mmd))
    indices=np.argsort(mmds)
    sorted_fake=fake_reshaped[indices]
    sorted_fake=np.vstack(sorted_fake)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    scio.savemat(os.path.join(save_path,"{}.mat".format(stage)),{
        "x":sorted_fake,
        "mmd":mmds
    })
    print("done")

def main():

   train= readDataSet(fold_idx,stage)
   sortTheBest(train,fold_idx)



if __name__ == '__main__':
    main()