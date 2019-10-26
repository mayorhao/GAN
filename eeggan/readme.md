nohup python -u run.py --stage=WAKE --task_id=0 --GPU=0 > ./train_logs/train.log 2>$1

nohup python -u run.py --stage=N1 --task_id=1 --GPU=1 > ./train_logs/train.N1.log 2>./train_logs/error.N1.log
nohup python -u run.py --stage=N3 --task_id=2 --GPU=2 > ./train_logs/train.N3.log 2>./train_logs/error.N3.log
nohup python -u run.py --stage=REM --task_id=3 --GPU=3 > ./train_logs/train.REM.log 2>./train_logs/error.REM.log


parser.add_argument("--stage",type=str,default="WAKE",help="determin which stege to be trained")
parser.add_argument("--task_id",type=int,default=0,help="the number to generate random seed")
parser.add_argument("--GPU",type=int,default=0,help="the GPU device id")
parser.add_argument("--i_block_tmp",type=int,default=0,help="which block to start with?")
parser.add_argument("--i_epoch_tmp",type=int,default=0,help="which epoch to start with?


conda activate eeggan
cd /home/fanjiahao/GAN/GAN/eeggan/examples/conv_cub
