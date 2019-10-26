import scipy.io as scio
import numpy as np

n_stage="WAKE"
data_path='/home/fanjiahao/GAN/extractSleepData/output/stage-total/total_stage.mat'
raw_data=scio.loadmat(data_path)
data_set=raw_data[n_stage]
# data_mat=scio.loadmat('/home/STOREAGE/fanjiahao/GAN/eeggan/stages.mat')
# EEG_data=data_mat['N1']
# train_set = EEG_data['train_set']
# test_set = EEG_data['test_set']
# train = np.concatenate((train_set.X,test_set.X))
# target = np.concatenate((train_set.y,test_set.y))
train=np.expand_dims(data_set,axis=1) # N*1*3840
train=np.reshape(train,(train.shape[0],train.shape[1],train.shape[2],1)) #N*1*3540*1