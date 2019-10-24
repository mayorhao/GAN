import glob
import scipy.io as scio
import os
import numpy as np
stages=['N1','N2','N3','REM','WAKE']
stage_dic={}
# for stage in stages:
#     stage_dic[stage]=[]
data_path='/home/fanjiahao/GAN/extractSleepData/output/stages-c3-128/*.mat'
data_list=glob.glob(os.path.join(data_path))
data_list.sort()
for i in range(len(data_list)):
    data_tmp = scio.loadmat(os.path.join(data_list[i], 'stages.mat'))
    for key in data_tmp:
        if key in stages:
            try:
                stage_dic[key]=np.vstack((stage_dic[key],data_tmp[key]))
            except KeyError:
                stage_dic[key]=data_tmp[key][:,:]
sum=0
for key in stage_dic:
    print(key,':',stage_dic[key].shape[0],str(stage_dic[key].shape[0]/120)+'hour','\n')
    sum=stage_dic[key].shape[0]+sum
print('total: '+str(sum))
