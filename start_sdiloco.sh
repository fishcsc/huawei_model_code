#!/bin/bash

TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
LOG_DIR="logs/s_diloco/$TIMESTAMP"

NUM_GPUS=4
echo "Starting training with timestamp: $TIMESTAMP"
CMD="torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS --master_port=29500 baseline_streaming_diloco.py"

# $CMD --log_dir $LOG_DIR --use_wandb --use_nesterov
$CMD --log_dir $LOG_DIR --use_wandb --use_nesterov --delay_steps 5

