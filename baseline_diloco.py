import os
os.environ['HF_HOME']='/data/hfhub'
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
os.environ['CUDA_LAUNCH_BLOCKING']='1'
import time
import argparse
import torch
import torch.distributed as dist
from torch.optim import AdamW, SGD
from copy import deepcopy
import wandb  # Add wandb import

from utils.common import *    # 从 utils/util.py 中导入函数
from utils.load_data_model import *  # 从 utils/load_data_model.py 中导入函数


# 批量 all-reduce 操作以提高通信效率
def sync_model_optimized(model, original_model, outer_optimizer, world_size, logger, comm_delay=None):
    with torch.no_grad():
        if outer_optimizer:
            # 更直接地计算梯度
            grads = []
            for param, original_param in zip(model.parameters(), original_model.parameters()):
                # 计算梯度（更新方向）
                grad = original_param.data - param.data  # 注意符号取反
                original_param.grad = grad
                grads.append(grad)
            
            # 批量通信 - 将所有梯度拼接
            comm_start = time.time()
            flat_grads = torch.cat([g.flatten() for g in grads])
            dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
            flat_grads.div_(world_size)
            
            # 将梯度复制回参数的梯度
            offset = 0
            for grad in grads:
                numel = grad.numel()
                grad.copy_(flat_grads[offset:offset+numel].view_as(grad))
                offset += numel
            
            comm_time = time.time() - comm_start
            
            # 使用优化器进行更新
            outer_optimizer.step()
            outer_optimizer.zero_grad()
            
            # 将更新后的参数复制回工作模型
            for param, original_param in zip(model.parameters(), original_model.parameters()):
                param.data.copy_(original_param.data)
        else:
            # 直接平均的方法
            comm_start = time.time()
            for param in model.parameters():
                dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
                param.data.div_(world_size)
            comm_time = time.time() - comm_start
        
        # 更新模型参数
        for param, original_param in zip(model.parameters(), original_model.parameters()):
            param.data.copy_(param.data)
            
    # 模拟通信延迟
    if comm_delay:
        logger.info(f"模拟通信时间: {comm_delay:.4f} 秒")
        comm_time = comm_delay
    else:
        logger.info(f"通信时间 (all-reduce): {comm_time:.4f} 秒")
    return comm_time

def evaluate(model, eval_dataloader, device, world_size, task_type):
    """在验证集上评估模型性能"""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_tokens = 0  # 用于语言模型计算perplexity
    correct = 0  # 用于分类任务计算准确率
    
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            if task_type == "language_modeling":
                # 语言模型评估
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["input_ids"]  # 自回归预测下一个token
                )
            else:
                # 分类任务评估
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["label"]  
                )
            
            loss = outputs.loss
            batch_size = batch["input_ids"].size(0)
            
            # 根据任务类型累计不同的指标
            if task_type == "language_modeling":
                non_pad_mask = batch["attention_mask"].bool()
                num_tokens = non_pad_mask.sum().item()
                total_tokens += num_tokens
                total_loss += loss.item() * num_tokens
            else:
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # 对于分类任务，计算准确率
                if task_type == "classification":
                    predictions = outputs.logits.argmax(dim=-1)
                    correct += (predictions == batch["label"]).sum().item()
    # 汇总所有GPU上的统计数据
    if task_type == "classification":
        metrics = torch.tensor([total_loss, total_samples, correct], dtype=torch.float64, device=device)
    else:
        metrics = torch.tensor([total_loss, total_samples], dtype=torch.float64, device=device)
    
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    
    # 计算平均损失
    avg_loss = metrics[0].item() / metrics[1].item()
    
    results = {"loss": avg_loss}
    
    # 对于分类任务，计算准确率
    if task_type == "classification":
        accuracy = metrics[2].item() / metrics[1].item()
        results["accuracy"] = accuracy
    
    # 对于语言建模任务，计算困惑度
    if task_type == "language_modeling":
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        results["perplexity"] = perplexity
    
    model.train()  # 切回训练模式
    return results

def train(model, dataloader, optimizer,
          device, logger, world_size, rank, comm_delay=None, 
          task_type="classification", args=None):
    """
    训练过程：
    - 模型以 train 模式运行；
    - 每个节点独立进行前向、反向传播及优化更新；
    - 每训练 sync_interval 例如50 个 step 后，同步一次模型参数；
    - 分别统计计算（前向、反向、优化）和通信（参数同步）的时间。
    """
    model.train()
    global_step = 0
    comp_time_total = 0.0
    comm_time_total = 0.0
    
    log_comp_time = 0.0
    
    original_snapshot = deepcopy(model)  # 保存一份参数计算外梯度
    
    # 获取训练和验证数据加载器
    train_dataloader, eval_dataloader = dataloader
    
    if args.outer_lr != 1.0:
        outer_optimizer = SGD(
            original_snapshot.parameters(), lr=args.outer_lr, momentum=0.9, 
            nesterov=bool(args.use_nesterov)
        )
    else:
        outer_optimizer = None

    for epoch in range(args.epochs):  # 这里设定了最大 epochs，可以根据需要调整
        epoch_step = 0  # 记录每个epoch内的step数
        logger.info(f"开始 Epoch {epoch+1}/{args.epochs}")
        
        for batch in train_dataloader:
            
            if global_step >= args.total_steps:
                break
            
            # 将 batch 数据移动到 GPU 上
            batch = {k: v.to(device) for k, v in batch.items()}
            start_comp = time.time()
            
            
            # 前向计算、损失计算、反向传播与优化
            outputs = model(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["label"])
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            comp_time = time.time() - start_comp
            comp_time_total += comp_time
            log_comp_time += comp_time
            
            # 每隔 log_interval 步，输出一次日志
            if global_step % args.log_interval == 0:
                logger.info(f"Epoch {epoch+1}/{args.epochs}, Step {epoch_step} (Global: {global_step}) - Loss: {loss.item():.4f} - 计算时间: {log_comp_time:.4f} 秒")
                
                # 记录到 wandb (只在主进程中)
                if rank == 0 and args.use_wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/computation_time": log_comp_time,
                        "step": epoch_step,
                    }, step=global_step)
                
                log_comp_time = 0.0
            
            # 每隔 eval_interval 步，在验证集上评估
            if global_step % args.eval_interval == 0 and eval_dataloader is not None:
                eval_results = evaluate(model, eval_dataloader, device, world_size, task_type)
                
                if rank == 0:
                    # 根据任务类型输出不同的评估指标
                    if task_type == "classification":
                        logger.info(f"评估结果 [步数 {global_step}] - 验证损失: {eval_results['loss']:.4f}, 准确率: {eval_results['accuracy']:.4f}")
                    else:
                        logger.info(f"评估结果 [步数 {global_step}] - 验证损失: {eval_results['loss']:.4f}, 困惑度: {eval_results['perplexity']:.4f}")
                    
                    if args.use_wandb:
                        wandb_log = {"eval/loss": eval_results['loss']}
                        if "accuracy" in eval_results:
                            wandb_log["eval/accuracy"] = eval_results['accuracy']
                        if "perplexity" in eval_results:
                            wandb_log["eval/perplexity"] = eval_results['perplexity']
                        wandb.log(wandb_log, step=global_step)
            
            # 每隔 sync_interval 步，同步一次模型参数
            if global_step % args.sync_interval == 0:
                comm_time = sync_model_optimized(
                    model, original_snapshot, outer_optimizer, 
                    world_size, logger, comm_delay
                    )
                comm_time_total += comm_time
                
                # 记录通信时间到 wandb (只在主进程中)
                if rank == 0 and args.use_wandb:
                    wandb.log({
                        "communication_time": comm_time,
                    }, step=global_step)
            
            global_step += 1
            epoch_step += 1
        
        # 每个epoch结束时输出一次总结
        if rank == 0:
            logger.info(f"完成 Epoch {epoch+1}/{args.epochs}, 当前步数: {global_step}")
            
        if global_step >= args.total_steps:
                # 进行最终评估
            final_metrics = {}
            if eval_dataloader is not None:
                final_eval_results = evaluate(model, eval_dataloader, device, world_size, task_type)
                for k, v in final_eval_results.items():
                    final_metrics[f"final/eval_{k}"] = v
                
                final_metrics.update({
                    "final/total_comp_time": avg_comp_time,
                    "final/total_comm_time": avg_comm_time,
                    "final/avg_comp_time_per_step": avg_comp_time / global_step,
                    "final/avg_comm_time_per_sync": avg_comm_time / (global_step // args.sync_interval),
                    "final/avg_step_time": (avg_comp_time + avg_comm_time) / global_step,
                    "final/comp_time_ratio": avg_comp_time / (avg_comp_time + avg_comm_time),
                })
                if rank == 0:
                    logger.info(f"最终评估结果: {final_metrics}")
                    if args.use_wandb:
                        wandb.log(final_metrics, step=global_step)
            break
    
    # 汇总所有GPU的时间统计数据
    metrics = torch.tensor([comp_time_total, comm_time_total, global_step], dtype=torch.float64, device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    
    # 除以进程数得到平均值
    avg_comp_time = metrics[0].item() / world_size
    avg_comm_time = metrics[1].item() / world_size
    
    # 同步 global_step (应该都一样，但为安全起见)
    global_step = int(metrics[2].item() / world_size)
    
    # 只在主进程(rank 0)输出平均统计信息
    if rank == 0:
        logger.info(f"===== 平均统计信息（所有 {world_size} 个GPU） =====")
        logger.info(f"总计算时间(平均): {avg_comp_time:.4f} 秒")
        logger.info(f"总通信时间(平均): {avg_comm_time:.4f} 秒")
        logger.info(f"平均计算时间: {avg_comp_time / global_step:.4f} 秒")
        logger.info(f"平均通信时间: {avg_comm_time / (global_step // args.sync_interval):.4f} 秒")
        logger.info(f"平均步长时间: {(avg_comp_time + avg_comm_time) / global_step:.4f} 秒")
        logger.info(f"计算时间占比: {avg_comp_time / (avg_comp_time + avg_comm_time):.2%}")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='分布式训练 BERT 模型')
    parser.add_argument('--log_dir', type=str, default=None, 
                        help='日志目录')
    # 添加训练相关参数
    parser.add_argument('--model_name', type=str, default='bert',
                        help='预训练模型名称')
    parser.add_argument('--dataset_name', type=str, default='sst2',
                        help='数据集名称')
    parser.add_argument('--epochs', type=int, default=3,
                        help='训练的轮数')
    parser.add_argument('--sync_interval', type=int, default=50,
                        help='模型参数同步的间隔步数')
    parser.add_argument('--total_steps', type=int, default=1500,
                        help='总训练步数')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批处理大小')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='学习率')
    parser.add_argument('--outer_lr', type=float, default=0.4,
                        help='外部学习率')
    parser.add_argument('--use_nesterov', action='store_true',
                        help='是否对外层使用 Nesterov 动量')
    parser.add_argument('--bandwidth', type=float, default=None,
                    help='模拟的网络带宽 (Mbps)，不设置则使用实际带宽')
    parser.add_argument('--log_interval', type=int, default=50,
                    help='日志输出的间隔步数')
    # 添加评估相关参数
    parser.add_argument('--eval_interval', type=int, default=50,
                        help='验证集评估的间隔步数')
    parser.add_argument('--eval_batch_size', type=int, default=32,
                        help='验证时的批处理大小')
    # 添加 wandb 相关参数
    parser.add_argument('--wandb_project', type=str, default='distrain',
                    help='Weights & Biases 项目名称')
    parser.add_argument('--wandb_name', type=str, default=None,
                    help='Weights & Biases 实验名称')
    parser.add_argument('--use_wandb', action='store_true',
                    help='是否使用 Weights & Biases 进行日志记录')
    args = parser.parse_args()
    
    # 初始化分布式训练相关环境
    local_rank, rank, world_size = init_distributed()
    device = torch.device("cuda", local_rank)
    
    # 设置 logger
    logger = setup_logging(rank, args.log_dir)
    logger.info(f"初始化完成。当前 Rank: {rank}, 总进程数: {world_size}")

    # 初始化 wandb (只在主进程中)
    if rank == 0 and args.use_wandb:
        # 如果没有指定实验名称，自动生成一个
        if args.wandb_name is None:
            args.wandb_name = f"diloco_{args.model_name}_{args.dataset_name}_sync{args.sync_interval}"
            if args.bandwidth:
                args.wandb_name += f"_bw{args.bandwidth}"
            if args.outer_lr != 1.0:
                args.wandb_name += f"_olr{args.outer_lr}"
                if args.use_nesterov:
                    args.wandb_name += "_nesterov"
        
        # 初始化 wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            dir='/data/yzhu/distrain/runs',
            config=vars(args)
        )
        logger.info(f"Weights & Biases 初始化完成。项目: {args.wandb_project}, 实验: {args.wandb_name}")
    
    # 记录训练参数
    if rank == 0:
        logger.info(f"训练参数: {args}")

    # 加载数据集和对应的模型与分词器（每个节点各自加载独立副本）
    train_dataloader, eval_dataloader, tokenizer, model, task_type = load_data_and_model(
        args.dataset_name, args.model_name, 
        args.batch_size, args.eval_batch_size,
        rank, world_size
    )
    model.to(device)
    
    # 配置优化器
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    # 计算通信延迟
    comm_delay = calc_comm_delay(model, world_size, logger, simulated_bandwidth_mbps=args.bandwidth)
    
    # 使用传入的参数进行训练，传入训练和验证数据加载器
    train(model, (train_dataloader, eval_dataloader), optimizer,
          device, logger, world_size, rank, comm_delay,
          task_type, args)
    
    logger.info("训练结束。")
    
    # 在主进程中记录结束时间和关闭 wandb
    if rank == 0:
        with open(os.path.join(args.log_dir, "config.txt"), "a") as f:
            f.write(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Training parameters: {args}\n")
        
        if args.use_wandb:
            wandb.finish()
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
