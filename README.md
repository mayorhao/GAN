# GAN

Code for
Hartmann, K. G., Schirrmeister, R. T., & Ball, T. (2018).
EEG-GAN: Generative adversarial networks for electroencephalograhic (EEG) brain signals.
Retrieved from https://arxiv.org/abs/1806.01875


cd /home/STOREAGE/fanjiahao/code/GAN/eeggan/examples/conv_lin
conda activate eeggan
nohup python -u run-new.py --stage=N1 --task_id=0 --GPU=0 >evolution/train_logs/MASS

cd /home/fanjiahao/GAN/GAN/eeggan/examples/shallow_conv_lin
conda activate eeggan
python run-new.py --stage=N1 --task_id=0 --GPU=0

cd /home/STOREAGE/fanjiahao/code/[git]GAN/GAN/eeggan/examples/conv_lin
python run-new.py --stage=N1 --task_id=0 --GPU=0



task_id_map={
    "N1":0,
    "N3":3,
    "REM":2,
    "WAKE":4
    
cd /home/fanjiahao/GAN/GAN/eeggan/examples/conv_lin
conda activate eeggan 
touch ./evolution/train_logs/MASS/fold_0/WAKE.log   
 nohup python -u run-new.py --stage=WAKE  --GPU=3 --seed=3 --fold_idx=0  > ./evolution/train_logs/edf/fold_0/WAKE.log 
 
 
 
 Begin to train stage:N1 ,GPU:0,seed:4,block_tmp:0,epoch_tmp:0,folds:1
load traning data from 403/stages.mat
Traceback (most recent call last):
  File "/home/fanjiahao/anaconda3/envs/eeggan/lib/python3.6/site-packages/scipy/io/matlab/mio.py", line 39, in _open_file
    return open(file_like, mode), True
FileNotFoundError: [Errno 2] No such file or directory: '/home/fanjiahao/STORERAGE/dataset/evolution/sleep-edf-absmax-staged-interploation/403/stages.mat'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "run-new.py", line 102, in <module>
    data_tmp = scio.loadmat(os.path.join(data_path,subject_list_temp[i], 'stages.mat'))
  File "/home/fanjiahao/anaconda3/envs/eeggan/lib/python3.6/site-packages/scipy/io/matlab/mio.py", line 216, in loadmat
    with _open_file_context(file_name, appendmat) as f:
  File "/home/fanjiahao/anaconda3/envs/eeggan/lib/python3.6/contextlib.py", line 81, in __enter__
    return next(self.gen)
  File "/home/fanjiahao/anaconda3/envs/eeggan/lib/python3.6/site-packages/scipy/io/matlab/mio.py", line 19, in _open_file_context
    f, opened = _open_file(file_like, appendmat, mode)
  File "/home/fanjiahao/anaconda3/envs/eeggan/lib/python3.6/site-packages/scipy/io/matlab/mio.py", line 45, in _open_file
    return open(file_like, mode), True
FileNotFoundError: [Errno 2] No such file or directory: '/home/fanjiahao/STORERAGE/dataset/sleep-edf-absmax-staged-interploation/403/stages.mat'
