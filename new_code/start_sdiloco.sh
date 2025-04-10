#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_LAUNCH_BLOCKING=1
export HF_HOME='/data/hfhub'
export HF_ENDPOINT='https://hf-mirror.com'
export TIMEOUT_NCCL_MINUTES=120

run_experiment() {
    local args="$@"
    local timestamp=$(date +"%Y%m%d-%H%M%S")
    torchrun --nnodes=1 --nproc_per_node=4 --master_port=29500 baseline_streaming_diloco.py \
        --log_dir "logs/s_diloco/$timestamp" --use_wandb \
        $args
}


### LLAMA150M C4EN
# DiLoCo H=50
run_experiment --dataset_name "c4en" --model_name "llama150m" \
    --sync_interval 50 --use_nesterov \
    --eval_interval 10 \
    --use_amp --amp_type 'bf16' --total_steps 44000 \
    --checkpoint_interval 50 --checkpoint_dir 'ckpts/streaming_diloco_150m' \
    --resume --max_checkpoints 5 \

### BERT SST2
# DP
# run_experiment --sync_interval 1 --outer_lr 1.0 --use_nesterov

# DiLoCo H=50
# run_experiment --sync_interval 50 --outer_lr 0.7 --use_nesterov