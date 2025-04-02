#!/bin/bash

run_experiment() {
    local args="$@"
    local timestamp=$(date +"%Y%m%d-%H%M%S")
    torchrun --nnodes=1 --nproc_per_node=4 --master_port=29500 baseline_diloco.py \
        --log_dir "logs/diloco/$timestamp" --use_wandb \
        $args
}

run_experiment --dataset_name "c4en" --model_name "llama150m" --use_nesterov
# run_experiment --sync_interval 1
# run_experiment --sync_interval 1 --outer_lr 1.0