#!/bin/bash

# 定义默认训练参数
EPOCHS=1
SYNC_INTERVAL=50
LOG_INTERVAL=50
TOTAL_STEPS=100000
BATCH_SIZE=16
LEARNING_RATE=2e-5
OUTER_LR=0.4  # 外部学习率
USE_NESTEROV=1  # 是否使用Nesterov动量
MODEL_NAME="bert-base-uncased"  # 默认模型名称
DATASET_NAME="sst2"  # 默认数据集名称
BANDWIDTH=""  # 默认为空，表示不模拟带宽限制，Mbps
LOG_NAME=""  # 默认为空，表示不指定日志名称
METHOD="diloco"  # 默认方法名称
# streaming diloco params
DELAY_STEPS=5  # 延迟步数τ
NUM_SHARDS=5  # 分片数
SYNC_OFFSET=0  # 同步偏移量
# beta dc streaming diloco params
USE_FO=0  # 是否使用一阶延迟补偿
FO_BETA=0.5  # 一阶延迟补偿系数β
USE_SO=0  # 是否使用二阶延迟补偿
SO_BETA=0.5  # 二阶延迟补偿系数β
HESSIAN_PHO=0.01  # 海森矩阵估计 rho
# dc diloco params
# DC_LAMBDA=1  # 延迟补偿系数λ，当为streaming diloco时为混合系数 alpha
# DC_METHOD="vanilla"  # 延迟补偿方法，默认为vanilla
# DC_ADP_LAMBDA=0   # 是否自适应λ，默认为0
# 程序名
FILE_NAME=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --dataset-name)
      DATASET_NAME="$2"
      shift 2
      ;;
    --sync-interval)
      SYNC_INTERVAL="$2"
      shift 2
      ;;
    --log-interval)
      LOG_INTERVAL="$2"
      shift 2
      ;;
    --ts | --total-steps)
      TOTAL_STEPS="$2"
      shift 2
      ;;
    --bs | --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --lr | --learning-rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --olr | --outer-lr)
      OUTER_LR="$2"
      shift 2
      ;;
    --une | --use-nesterov)
      USE_NESTEROV="$2"
      shift 2
      ;;
    --bw | --bandwidth)
      BANDWIDTH="$2"
      shift 2
      ;;
    --method)
      METHOD="$2"
      shift 2
      ;;
    --logname)
      LOG_NAME="$2"
      shift 2
      ;;
    # Streaming DiLoCo 参数解析
    --ns | --num-shards)
      NUM_SHARDS="$2"
      shift 2
      ;;
    --sync-offset)
      SYNC_OFFSET="$2"
      shift 2
      ;;
    # beta dc streaming diloco 参数解析
    --use-fo)
      USE_FO="$2"
      shift 2
      ;;
    --fbeta)
      FO_BETA="$2"
      shift 2
      ;;
    --use-so)
      USE_SO="$2"
      shift 2
      ;;
    --sbeta)
      SO_BETA="$2"
      shift 2
      ;;
    --hessian-rho)
      HESSIAN_PHO="$2"
      shift 2
      ;;
    # 延迟补偿参数解析
    --delay-steps)
      DELAY_STEPS="$2"
      shift 2
      ;;
    --dc-method)
      DC_METHOD="$2"
      shift 2
      ;;
    --dc-lambda)
      DC_LAMBDA="$2"
      shift 2
      ;;
    --dc-adp-lambda)
      DC_ADP_LAMBDA="$2"
      shift 2
      ;;
    *)
      # 处理未知选项
      echo "未知选项: $1"
      shift
      ;;
  esac
done

# 获取当前时间作为时间戳
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
echo "Starting training with timestamp: $TIMESTAMP"

# 输出训练参数
echo "Training parameters:"
echo "  Method Name: $METHOD"
echo "  Model Name: $MODEL_NAME"
echo "  Dataset Name: $DATASET_NAME"
echo "  Log Name: $LOG_NAME"
echo "  Epochs: $EPOCHS"
echo "  Sync Interval: $SYNC_INTERVAL"
echo "  Log Interval: $LOG_INTERVAL"
echo "  Total Steps: $TOTAL_STEPS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Outer Learning Rate: $OUTER_LR"
echo "  Use Nesterov: $USE_NESTEROV"
if [ -n "$BANDWIDTH" ]; then
  echo "  Simulated Bandwidth: ${BANDWIDTH}Mbps"
else
  echo "  Bandwidth: 实际带宽 (不模拟)"
fi
# 如果使用 Streaming DiLoCo 方法，输出对应参数
if [ "$METHOD" = "s_diloco" ]; then
  echo "  Number of Shards: $NUM_SHARDS"
  echo "  Sync Offset: $SYNC_OFFSET"
  echo "  Delay Steps: $DELAY_STEPS"
  echo "  Blend Alpha: $DC_LAMBDA"
fi
# 如果使用 beta dc streaming diloco 方法，输出对应参数
if [ "$METHOD" = "beta_dc_diloco" ]; then
  echo "  Delay Steps: $DELAY_STEPS"
  echo "  Use First Order Delay Compensation: $USE_FO"
  echo "  First Order Delay Compensation Beta: $FO_BETA"
  echo "  Use Second Order Delay Compensation: $USE_SO"
  echo "  Second Order Delay Compensation Beta: $SO_BETA"
  echo "  Hessian Matrix Estimation Rho: $HESSIAN_PHO"
fi
# 如果method=dc_diloco，输出对应参数
if [ "$METHOD" = "dc_diloco" ]; then
  echo "  DC Delay Steps: $DELAY_STEPS"
  echo "  DC Compensate Method: $DC_METHOD"  
  echo "  DC Lambda: $DC_LAMBDA"
  echo "  DC Adaptive Lambda: $DC_ADP_LAMBDA"
fi

# 设置 NCCL 相关环境变量（可选，根据集群环境调整）
# export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME=eth0  # 根据实际网卡名称调整

# 获取可用的 GPU 数量
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Running with $NUM_GPUS GPUs"

# 如果LOG_NAME不为空，则将日志文件夹重命名为<LOG_NAME>
if [ -n "$LOG_NAME" ]; then
  LOG_DIR="/data/yzhu/distrain/logs/$METHOD/$TIMESTAMP-$LOG_NAME"
else
  LOG_DIR="/data/yzhu/distrain/logs/$METHOD/$TIMESTAMP"
fi
mkdir -p "$LOG_DIR"  # 创建日志目录

# 使用 torchrun 启动分布式训练，传递所有训练参数
CMD_ARGS=(
  "--model_name" "$MODEL_NAME"
  "--dataset_name" "$DATASET_NAME"
  "--log_dir" "$LOG_DIR"
  "--epochs" "$EPOCHS"
  "--sync_interval" "$SYNC_INTERVAL"
  "--log_interval" "$LOG_INTERVAL"
  "--total_steps" "$TOTAL_STEPS"
  "--batch_size" "$BATCH_SIZE"
  "--learning_rate" "$LEARNING_RATE"
  "--outer_lr" "$OUTER_LR"
  "--use_nesterov" "$USE_NESTEROV"
)

# 添加可选参数
if [ -n "$BANDWIDTH" ]; then
  CMD_ARGS+=("--bandwidth" "$BANDWIDTH")
fi

# 添加方法特定参数
if [ "$METHOD" = "dc_diloco" ]; then
  CMD_ARGS+=("--dc_method" "$DC_METHOD")
  CMD_ARGS+=("--dc_lambda" "$DC_LAMBDA")
  CMD_ARGS+=("--dc_adp_lambda" "$DC_ADP_LAMBDA")
  CMD_ARGS+=("--delay_steps" "$DELAY_STEPS")
fi
if [ "$METHOD" = "beta_dc_diloco" ]; then
  CMD_ARGS+=("--first_order" "$USE_FO")
  CMD_ARGS+=("--fbeta" "$FO_BETA")
  CMD_ARGS+=("--second_order" "$USE_SO")
  CMD_ARGS+=("--sbeta" "$SO_BETA")
  CMD_ARGS+=("--rho" "$HESSIAN_PHO")
  CMD_ARGS+=("--delay_steps" "$DELAY_STEPS")
fi
if [ "$METHOD" = "s_diloco" ]; then
  CMD_ARGS+=("--num_shards" "$NUM_SHARDS")
  CMD_ARGS+=("--offset" "$SYNC_OFFSET")
  CMD_ARGS+=("--delay_steps" "$DELAY_STEPS")
  CMD_ARGS+=("--alpha" "$DC_LAMBDA")
fi

# 设置对应的文件
if [ "$METHOD" = "diloco" ]; then
  FILE_NAME='/data/yzhu/distrain/baseline_diloco.py'
elif [ "$METHOD" = "dc_diloco" ]; then
  FILE_NAME='/data/yzhu/distrain/exp_dc_diloco.py'
elif [ "$METHOD" = "s_diloco" ]; then
  FILE_NAME='/data/yzhu/distrain/baseline_streaming_diloco.py'
elif [ "$METHOD" = "beta_dc_diloco" ]; then
  FILE_NAME='/data/yzhu/distrain/beta_dc_diloco.py'
else
  echo "Unknown method: $METHOD. Please use 'diloco', 's_diloco', 'beta_dc_diloco', or 'dc_diloco'."
  exit 1
fi

# 执行命令并捕获退出状态
set -e  # 启用错误检测
torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS --master_port=29500 \
  "$FILE_NAME" \
  "${CMD_ARGS[@]}" || {
    # 如果 torchrun 命令失败执行此块
    echo "Training failed with exit code $?"
    # 检查并删除日志目录
    if [ -d "$LOG_DIR" ]; then
      echo "Removing failed run logs directory: $LOG_DIR"
      rm -rf "$LOG_DIR"
    fi
    exit 1  # 以错误状态退出
  };

# 结束后等待 5 秒
sleep 5


# 打印训练日志位置
echo "Training complete. Logs saved to: $LOG_DIR"
