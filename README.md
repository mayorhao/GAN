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

cd /home/STOREAGE/fanjiahao/code/[git]GAN/GAN/eeggan/examples/shallow_conv_lin
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
 