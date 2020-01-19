

conda activate eeggan
cd /home/fanjiahao/GAN/GAN/eeggan/examples/shallow_conv_lin
nohup python -u run-new.py --stage=WAKE --task_id=1 --GPU=2 > ./logs/shallow-train/WAKE.log
python run-new.py --fold_idx=0 --stage=WAKE --GPU=3 > ./logs/shallow

cd /home/STOREAGE/fanjiahao/code/[git]GAN/GAN/eeggan/examples/shallow_conv_lin
conda activate eeggan
nohup python -u run-shallow-no-progressive.py --stage=N1  --GPU=0 --fold_idx=2 > ./logs/shallow_train/fold2/N1.log && nohup python -u run-shallow-no-progressive.py --stage=WAKE --task_id=41323 --GPU=0 --fold_idx=2 > ./logs/shallow_train/fold2/WAKE.log
nohup python -u run-shallow-no-progressive.py --stage=N3  --GPU=1 --fold_idx=2 > ./logs/shallow_train/fold2/N3.log
nohup python -u run-shallow-no-progressive.py --stage=REM  --GPU=2 --fold_idx=2 > ./logs/shallow_train/fold2/REM.log



cd /home/fanjiahao/GAN/GAN/eeggan/examples/shallow_conv_lin
conda activate eeggan
nohup python -u run-shallow-no-progressive.py --stage=N1  --GPU=0 --fold_idx=2 > ./logs/shallow_train/fold2/N1.log && nohup python -u run-shallow-no-progressive.py --stage=WAKE --task_id=41323 --GPU=0 --fold_idx=2 > ./logs/shallow_train/fold2/WAKE.log
nohup python -u run-shallow-no-progressive.py --stage=N3  --GPU=1 --fold_idx=2 > ./logs/shallow_train/fold2/N3.log
nohup python -u run-shallow-no-progressive.py --stage=REM  --GPU=2 --fold_idx=2 > ./logs/shallow_train/fold2/REM.log


