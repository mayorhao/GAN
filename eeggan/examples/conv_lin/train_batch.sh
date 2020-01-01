#!/bin/bash
start=$1
end=$2
stage=$3
gpu=$4
#python -u run-new.py --stage=$stage --fold_idx=$fold_idx_start --GPU=$gpu >./edf-log/fold_$fold_idx_start/$stage.log && python -u run-new.py --stage=$stage --fold_idx=$fold_idx_end --GPU=$gpu >./edf-log/fold_$fold_idx_end/$stage.log
#i_block_tmp=$5
#i_epoch_tmp=$6
if  [[ -n "$start" ]] && [[ -n "$end" ]] && [[ -n "$gpu" ]]; then
#    python -u run-new.py --stage=$stage --fold_idx=$start --GPU=$gpu --i_block_tmp=$i_block_tmp --i_epoch_tmp=$i_epoch_tmp --reuse=reuse>./edf-log/fold_$start/$stage.log
    for i in $(eval echo {$start..$end})
    do
        python -u run-new.py --stage=$stage --fold_idx=$i --GPU=$gpu >./edf-log/fold_$i/$stage.log
    done
else
    echo "argument error"
fi