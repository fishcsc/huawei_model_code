import os
os.environ['HF_HOME']='/data/hfhub'
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
os.environ['CUDA_LAUNCH_BLOCKING']='1'
import time
import torch
import torch.distributed as dist
from torch.optim import AdamW, SGD
import argparse
import wandb  # Add wandb import
from utils.common import *    # 从 utils/util.py 中导入函数
from utils.load_data_model import *  # 从 utils/load_data_model.py 中导入函数


# 批量 all-reduce 操作以提高通信效率，实现传算分离
def sync_model_optimized(model, shard_tracker, sync_shard_idx, world_size, logger,
                         comm_delay, num_shards, alpha):
    """
    分块同步模型参数 (Corrected Logic)
    model: 当前工作模型 (state at t_now)
    shard_tracker: 包含旧状态的字典
      - params: state at t_rec - H (base for outer gradient)
      - staged_params: state at t_rec (recorded delay_steps ago)
    """
    cur_shard = shard_tracker[sync_shard_idx]
    comm_time = 0.0 # Initialize comm_time

    # --- 1. Calculate Outer Gradient Delta: Δm,p = θ(t_rec - H)_m,p - θ(t_rec)_m,p ---
    # Ensure staged_params exist (should have been recorded 'delay_steps' ago)
    if cur_shard["staged_params"] is None:
         # This case should ideally not happen if logic is correct, but good to handle.
         logger.error(f"Error: staged_params for shard {sync_shard_idx} is None at receive time.")
         # Decide how to handle: skip sync? raise error? For now, let's log and skip.
         return 0.0 # Return 0 comm time as nothing happened

    with torch.no_grad():
        # p_old corresponds to θ(t_rec - H)_m,p
        # p_rec corresponds to θ(t_rec)_m,p
        sync_grads = [p_old.data - p_rec.data
                      for p_old, p_rec in zip(cur_shard["params"], cur_shard["staged_params"])]

    # --- 2. Communicate and Average Delta: Δp = (1/M) * Σ Δm,p ---
    comm_start = time.time()
    # Batch communication - flatten gradients for this shard
    try:
        flat_grads = torch.cat([g.flatten() for g in sync_grads])
    except RuntimeError as e:
        logger.error(f"Error flattening gradients for shard {sync_shard_idx}: {e}")
        # Log shapes for debugging
        for i, g in enumerate(sync_grads):
            logger.error(f"  Grad {i} shape: {g.shape}, dtype: {g.dtype}, device: {g.device}")
        return 0.0 # Skip if flattening fails

    dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
    flat_grads.div_(world_size) # flat_grads now holds averaged Δp

    # Unflatten the averaged delta back into sync_grads list structure
    # sync_grads will now hold the averaged Δp, structured like the parameters
    offset = 0
    for grad in sync_grads:
        numel = grad.numel()
        if offset + numel > flat_grads.numel():
             logger.error(f"Error unflattening: offset {offset} + numel {numel} > flat_grads size {flat_grads.numel()}")
             return 0.0 # Skip if unflattening calculation is wrong
        grad.copy_(flat_grads[offset:offset + numel].view_as(grad))
        offset += numel
    comm_time = time.time() - comm_start
    del flat_grads # Free memory

    # --- 3. Apply Outer Optimization: θ_outer = OuterOpt(θ(t_rec - H)_p, Δp) ---
    # The optimizer updates cur_shard["params"] (which holds θ(t_rec - H))
    # using sync_grads (which holds Δp) as the gradient.
    if cur_shard['outer_optimizer']:
        # Assign the averaged delta (Δp stored in sync_grads) as the gradient
        # for the parameters the outer optimizer manages (cur_shard["params"])
        for param, avg_delta in zip(cur_shard["params"], sync_grads):
            # Ensure grad buffer exists and copy avg_delta into it
            if param.grad is None:
                param.grad = avg_delta.clone() # Create grad buffer if needed
            else:
                param.grad.copy_(avg_delta)

        # Perform the optimizer step (updates cur_shard["params"])
        cur_shard['outer_optimizer'].step()
        cur_shard['outer_optimizer'].zero_grad() # Clean up gradients
        # cur_shard["params"] now holds θ_outer(t_now)_p

    else: # Equivalent to simple averaging (outer_lr=1.0 means SGD with lr=1.0)
          # θ_outer = θ(t_rec - H) - Δp
        with torch.no_grad():
            for param, avg_delta in zip(cur_shard["params"], sync_grads):
                param.data.sub_(avg_delta.data)
        # cur_shard["params"] now holds θ_outer(t_now)_p

    del sync_grads # Free memory associated with the averaged delta list

    # --- 4. Merge: θ(t_now)_m,p = α*θ(t_now)_m,p + (1-α)*θ_outer ---
    # Mix the locally computed parameters (model) with the globally updated ones (cur_shard["params"])
    start_idx = cur_shard["start_idx"]
    end_idx = cur_shard["end_idx"]
    globally_updated_params = cur_shard["params"] # Alias for clarity

    with torch.no_grad():
        model_params_list = list(model.parameters())
        # Get current local params for the shard
        current_local_params = model_params_list[start_idx:end_idx]

        for local_p, global_p in zip(current_local_params, globally_updated_params):
            local_p.data.mul_(alpha).add_(global_p.data, alpha=1 - alpha)
        del model_params_list # Dereference list

    # --- 5. Prepare State for Next Cycle ---
    # Update cur_shard["params"] to hold θ(t_rec) for the *next* synchronization's base state.
    # This state was stored in staged_params.
    with torch.no_grad():
        for param, staged_param in zip(cur_shard["params"], cur_shard["staged_params"]):
            param.data.copy_(staged_param.data)
        # staged_params are now effectively 'consumed' for this cycle.
        # We can optionally clear the reference, but it will be overwritten anyway.
        # cur_shard["staged_params"] = None # Optional memory optimization

    # --- Logging and Return ---
    if comm_delay:
        # Simulate delay based on config, dividing total delay by shards for average effect
        actual_delay = comm_delay / num_shards
        logger.info(f"分片 {sync_shard_idx+1} 模拟通信时间: {actual_delay:.4f} 秒")
        return actual_delay
    else:
        # Return measured all-reduce time
        logger.info(f"分片 {sync_shard_idx+1} 通信时间 (all-reduce): {comm_time:.4f} 秒")
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
    - 模型以 train 模式运行
    - 实现传、算分离的分块同步策略
    - 模拟通信延迟，延迟应用更新
    - 分别统计计算和通信的时间
    """
    model.train()
    global_step = 0
    comp_time_total = 0.0
    comm_time_total = 0.0
    
    log_comp_time = 0.0

    # 获取验证数据加载器
    train_dataloader, eval_dataloader = dataloader

    # 初始化分片追踪器
    num_shards = args.num_shards
    all_params = list(model.parameters())
    params_per_shard = len(all_params) // num_shards
    shard_tracker = {}
    
    # 每个分片的sync时间点
    sync_shard_interval = args.sync_interval // num_shards
    sync_time_points = [i * sync_shard_interval + args.offset for i in range(num_shards)]
    receive_time_points = [point + args.delay_steps for point in sync_time_points]
    if rank == 0: logger.info(f"每个分片的同步时间点: {sync_time_points}\n每个分片的接收时间点: {receive_time_points}")
    
    # 为每个分片创建跟踪信息
    model_params_list = list(model.parameters())
    for shard_idx in range(num_shards):
        start_idx = shard_idx * params_per_shard
        end_idx = start_idx + params_per_shard if shard_idx < num_shards - 1 else len(all_params)
        
        # 初始化分片跟踪器 - 使用深拷贝确保参数副本独立
        current_params = [p.data.clone() for p in model_params_list[start_idx:end_idx]]
        shard_tracker[shard_idx] = {
                "start_idx": start_idx,  # 分片起始索引
                "end_idx": end_idx,  # 分片结束索引
                "params": current_params,  # 发送中的参数副本，即 t-\tau，后续会放到 old_params 中
                "staged_params": None,  # 上一次发送时的参数，即 t-\tau-H
                "sent_at_step": 0,  # 发送时的步数
            }
        outer_optimizer = None if args.outer_lr == 1.0 else SGD(
            shard_tracker[shard_idx]["params"] , lr=args.outer_lr, 
            momentum=0.9, nesterov=bool(args.use_nesterov)
        )
        shard_tracker[shard_idx]["outer_optimizer"] = outer_optimizer
    del model_params_list
    if rank == 0:  logger.info(f"模型共 {len(all_params)} 个参数组，分为 {num_shards} 个分片")

    for epoch in range(args.epochs):
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
            
            # 检查是否需要发送或接收模型分片
            if global_step % args.sync_interval in receive_time_points:
                # print(f"当前步数: {global_step} 准备接收")
                sync_shard_idx = (global_step - args.delay_steps) % args.sync_interval // sync_shard_interval
                comm_time = sync_model_optimized(
                    model, shard_tracker, sync_shard_idx, world_size, logger,
                    comm_delay, num_shards, args.alpha)
                
                comm_time_total += comm_time
                
                # 记录通信时间到 wandb (只在主进程中)
                if rank == 0 and args.use_wandb:
                    wandb.log({
                        "communication_time": comm_time,
                        "sync_shard_idx": sync_shard_idx,
                    }, step=global_step)
            
            # 检查是否记录分片。注意第一个interval期间使用最开始的模型参数即可
            if global_step % args.sync_interval in sync_time_points:
                # print(f"当前步数: {global_step} 准备记录")
                sync_shard_idx = global_step % args.sync_interval // sync_shard_interval
                shard_tracker[sync_shard_idx]["sent_at_step"] = global_step
                # 使用clone()创建参数的深拷贝而不仅是引用
                model_params_list = list(model.parameters())
                start_idx = shard_tracker[sync_shard_idx]["start_idx"]
                end_idx = shard_tracker[sync_shard_idx]["end_idx"]
                shard_tracker[sync_shard_idx]["staged_params"] = [p.data.clone() for p in model_params_list[start_idx:end_idx]]
                del model_params_list
                logger.info(f"分片 {sync_shard_idx+1} 已记录并发送，当前步数: {global_step}")
  
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
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                        help='预训练模型名称')
    parser.add_argument('--dataset_name', type=str, default='sst2',
                        help='数据集名称')
    parser.add_argument('--epochs', type=int, default=3,
                        help='训练的轮数')
    parser.add_argument('--sync_interval', type=int, default=50,
                        help='模型参数同步的间隔步数')
    parser.add_argument('--total_steps', type=int, default=10000,
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
    # 新增参数
    parser.add_argument('--num_shards', type=int, default=5,
                        help='模型参数分块数量')
    parser.add_argument('--delay_steps', type=int, default=10,
                        help='通信延迟的步数')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='本地模型与更新模型的混合比例')
    parser.add_argument('--offset', type=int, default=0,
                        help='分片发送时间点的偏移量')
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

    # 记录训练参数
    if rank == 0:
        logger.info(f"训练参数: {args}")

    # 初始化 wandb (只在主进程中)
    if rank == 0 and args.use_wandb:
        # 如果没有指定实验名称，自动生成一个
        if args.wandb_name is None:
            args.wandb_name = f"{args.model_name}_{args.dataset_name}_streaming_ns{args.num_shards}_ds{args.delay_steps}"
            if args.outer_lr != 1.0:
                args.wandb_name += f"_olr{args.outer_lr}"
                if args.use_nesterov:
                    args.wandb_name += "_nesterov"
            if args.alpha != 0.5:
                args.wandb_name += f"_a{args.alpha}"
        
        # 初始化 wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            dir='/data/yzhu/distrain/runs',
            config=vars(args)
        )
        logger.info(f"Weights & Biases 初始化完成。项目: {args.wandb_project}, 实验: {args.wandb_name}")

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
