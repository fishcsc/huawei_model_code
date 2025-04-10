from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description='分布式训练 BERT 模型')
    parser.add_argument('--log_dir', type=str, default=None, 
                        help='日志目录')
    # 添加训练相关参数
    parser.add_argument('--model_name', type=str, default='llama150m',
                        help='预训练模型名称')
    parser.add_argument('--dataset_name', type=str, default='c4en',
                        help='数据集名称')
    parser.add_argument('--epochs', type=int, default=3,
                        help='训练的轮数')
    parser.add_argument('--sync_interval', type=int, default=50,
                        help='模型参数同步的间隔步数')
    parser.add_argument('--total_steps', type=int, default=1500,
                        help='总训练步数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='每个设备上的批处理大小')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                        help='梯度累积步数，用于模拟更大批量')
    parser.add_argument('--effective_batch_size', type=int, default=512,
                        help='有效批量大小，如果设置，将自动计算梯度累积步数 (= effective_batch_size / batch_size')
    parser.add_argument('--learning_rate', type=float, default=4e-4,
                        help='学习率')
    parser.add_argument('--outer_lr', type=float, default=0.4,
                        help='外部学习率')
    parser.add_argument('--use_nesterov', action='store_true',
                        help='是否对外层使用 Nesterov 动量')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='权重衰减')
    parser.add_argument('--bandwidth', type=float, default=None,
                    help='模拟的网络带宽 (Mbps)，不设置则使用实际带宽')
    parser.add_argument('--log_interval', type=int, default=50,
                    help='日志输出的间隔步数')
    # Warm Up
    parser.add_argument('--warmup_steps', type=int, default=1000,
                    help='热身学习率')
    # AMP
    parser.add_argument('--use_amp', action='store_true',
                        help='是否使用自动混合精度 (AMP)')
    parser.add_argument('--amp_type', type=str, default='bf16',
                        help='训练精度 (fp16 或 bf16)')
    # 添加评估相关参数
    parser.add_argument('--eval_interval', type=int, default=50,
                        help='验证集评估的间隔步数')
    parser.add_argument('--eval_batch_size', type=int, default=32,
                        help='验证时的批处理大小')
    # 添加 wandb 相关参数
    parser.add_argument('--wandb_project', type=str, default='distrain',
                    help='Weights & Biases 项目名称')
    # parser.add_argument('--wandb_fork_from_best', action='store_true',
                    # help='是否从最佳检查点恢复训练')
    parser.add_argument('--wandb_name', type=str, default=None,
                    help='Weights & Biases 实验名称')
    parser.add_argument('--use_wandb', action='store_true',
                    help='是否使用 Weights & Biases 进行日志记录')
    # 检查点
    parser.add_argument('--resume', action='store_true',
                        help='是否从检查点恢复训练')
    parser.add_argument('--checkpoint_dir', type=str, default='ckpts/',
                        help='保存检查点的目录')
    parser.add_argument('--checkpoint_interval', type=int, default=10,
                        help='保存检查点的间隔步数')
    parser.add_argument('--max_checkpoints', type=int, default=2,
                        help='最多保存的检查点数量')
    
    # Streaming DiLoCo 特有参数
    parser.add_argument('--num_shards', type=int, default=5, 
                        help='模型参数分块数量')
    parser.add_argument('--delay_steps', type=int, default=5, 
                        help='通信延迟的步数')
    parser.add_argument('--alpha', type=float, default=0.5, 
                        help='本地模型与全局更新模型的混合比例')
    parser.add_argument('--offset', type=int, default=0, 
                        help='分片发送时间点的偏移量')
    
    return parser.parse_args()