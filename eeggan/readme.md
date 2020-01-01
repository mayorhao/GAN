

conda activate eeggan
cd /home/fanjiahao/GAN/GAN/eeggan/examples/shallow_conv_lin
nohup python -u run-new.py --stage=WAKE --task_id=1 --GPU=2 > ./logs/shallow-train/WAKE.log
python run-new.py --fold_idx=0 --stage=WAKE --GPU=3 > ./logs/shallow