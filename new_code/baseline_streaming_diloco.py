import time
import torch
import torch.distributed as dist
from torch.optim import AdamW, SGD
from torch.amp import autocast, GradScaler
import argparse
import wandb  # Add wandb import
from transformers import get_cosine_schedule_with_warmup
from utils.common import *    # 从 utils/util.py 中导入函数
from utils.load_data_model import *  # 从 utils/load_data_model.py 中导入函数
from utils.arg_utils import parse_args
from utils.checkpoint_utils import save_streaming_diloco_checkpoint, load_streaming_diloco_checkpoint, get_latest_checkpoint


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

    # --- 1. Calculate Outer Gradient Delta: Δm,p = θ(t_rec - H)_m,p - θ(t_rec)_m,p ---
    with torch.no_grad():
        sync_grads = [p_old.data - p_rec.data
                      for p_old, p_rec in zip(cur_shard["params"], cur_shard["staged_params"])]

    # --- 2. Communicate and Average Delta: Δp = (1/M) * Σ Δm,p ---
    comm_start = time.time()
    # Batch communication - flatten gradients for this shard
    try:
        flat_grads = torch.cat([g.flatten() for g in sync_grads])
    except RuntimeError as e:
        logger.error(f"Error flattening gradients for shard {sync_shard_idx}: {e}")
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
    if cur_shard['outer_optimizer']:
        for param, avg_delta in zip(cur_shard["params"], sync_grads):
            if param.grad is None:
                param.grad = avg_delta.clone() # Create grad buffer if needed
            else:
                param.grad.copy_(avg_delta)
        # Perform the optimizer step (updates cur_shard["params"])
        cur_shard['outer_optimizer'].step()
        cur_shard['outer_optimizer'].zero_grad() # Clean up gradients
    else: # Equivalent to simple averaging (outer_lr=1.0 means SGD with lr=1.0)
        with torch.no_grad():
            for param, avg_delta in zip(cur_shard["params"], sync_grads):
                param.data.sub_(avg_delta.data)
        # cur_shard["params"] now holds θ_outer(t_now)_p
    del sync_grads # Free memory associated with the averaged delta list

    # --- 4. Merge: θ(t_now)_m,p = α*θ(t_now)_m,p + (1-α)*θ_outer ---
    start_idx = cur_shard["start_idx"]
    end_idx = cur_shard["end_idx"]
    globally_updated_params = cur_shard["params"] 
    with torch.no_grad():
        model_params_list = list(model.parameters())
        # Get current local params for the shard
        current_local_params = model_params_list[start_idx:end_idx]
        for local_p, global_p in zip(current_local_params, globally_updated_params):
            local_p.data.mul_(alpha).add_(global_p.data, alpha=1 - alpha)
        del model_params_list 

    # --- 5. Prepare State for Next Cycle ---
    with torch.no_grad():
        for param, staged_param in zip(cur_shard["params"], cur_shard["staged_params"]):
            param.data.copy_(staged_param.data)
        cur_shard["staged_params"] = None # Optional memory optimization

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

def train(model, dataloader, optimizer, scheduler,
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
    micro_step = 0
    comp_time_total = 0.0
    comm_time_total = 0.0
    
    log_comp_time = 0.0
    running_loss = 0.0

    # 获取验证数据加载器
    train_dataloader, eval_dataloader = dataloader

    # 初始化分片追踪器
    num_shards = args.num_shards
    all_params = list(model.parameters())
    params_per_shard = len(all_params) // num_shards
    shard_tracker = {}
    
    # 每个分片的sync时间点
    sync_shard_interval = args.sync_interval // num_shards
    # Calculate base sync points within an interval
    base_sync_points = [i * sync_shard_interval + args.offset for i in range(num_shards)]
    if rank == 0: logger.info(f"每个分片在间隔内的相对同步时间点: {base_sync_points}")
    
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
                "params": current_params,  # 上一次发送时的参数，即 t-\tau-H
                "staged_params": None,  # 发送中的参数副本，即 t-\tau，后续会放到 old_params 中
                "sent_at_step": -1,  # 发送时的步数
                "next_receive_step": -1,  # 下一次应该接收的步数
            }
        outer_optimizer = None if args.outer_lr == 1.0 else SGD(
            shard_tracker[shard_idx]["params"] , lr=args.outer_lr, 
            momentum=0.9, nesterov=bool(args.use_nesterov)
        )
        shard_tracker[shard_idx]["outer_optimizer"] = outer_optimizer
    del model_params_list
    if rank == 0:  logger.info(f"模型共 {len(all_params)} 个参数组，分为 {num_shards} 个分片")
    
    # 初始化 AMP GradScaler
    if args.use_amp and args.amp_type == torch.float16:  
        scaler = GradScaler(enabled=True)
        logger.info("Initialized GradScaler for AMP.")
    else:
        scaler = GradScaler(enabled=False)
    
    # 从检查点恢复训练（如果指定）
    best_metric = float('inf')  # 用于跟踪最佳验证结果
    epoch = 0
    if args.resume and args.checkpoint_dir:
        # 查找当前rank的最新检查点
        resume_path = get_latest_checkpoint(args.checkpoint_dir, logger, rank)
        
        if resume_path:
            logger.info(f"Rank {rank}: 正在从检查点恢复训练: {resume_path}")
            # 加载检查点状态
            training_state = load_streaming_diloco_checkpoint(
                resume_path, model, optimizer, scheduler,
                shard_tracker, scaler, rank, device, False, logger
            )
            # 更新训练状态
            if training_state:
                epoch = training_state.get("epoch", 0)
                global_step = training_state.get("global_step", 0)
                micro_step = training_state.get("micro_step", 0)
                comp_time_total = training_state.get("comp_time_total", 0.0)
                comm_time_total = training_state.get("comm_time_total", 0.0)
                logger.info(f"Rank {rank}: 恢复训练成功，当前轮次: {epoch}, 当前步数: {global_step}")
            else:
                logger.info(f"Rank {rank}: 检查点加载失败，将从头开始训练")
        else:
            logger.info(f"Rank {rank}: 未找到有效检查点，将从头开始训练")
    
    # 训练前清零梯度
    optimizer.zero_grad(set_to_none=True)

    for epoch_idx in range(epoch, args.epochs):
        epoch_step = 0  # 记录每个epoch内的step数
        logger.info(f"开始 Epoch {epoch_idx+1}/{args.epochs}")
        
        # 设置数据加载器的 epoch，确保恢复训练时的数据顺序正确
        if hasattr(train_dataloader, 'sampler') and hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch_idx)
            logger.info(f"设置 sampler epoch: {epoch_idx}")
        else:
            logger.info("当前 dataloader 没有可设置 epoch 的 sampler")
        
        for batch in train_dataloader:
            
            if global_step >= args.total_steps:
                break
            
            # 将 batch 数据移动到 GPU 上
            batch = {k: v.to(device) for k, v in batch.items()}
            start_comp = time.time()
            
            # 前向计算、损失计算、反向传播与优化
            with autocast(device_type=device.type, enabled=args.use_amp, dtype=args.amp_type):
                if task_type == "language_modeling":
                    outputs = model(**batch)
                elif task_type == "classification":
                    outputs = model(input_ids=batch["input_ids"],
                                   attention_mask=batch["attention_mask"],
                                   labels=batch["label"])
                loss = outputs.loss
            
            # 缩放损失以适应梯度累积
            running_loss += loss.item()
            scaled_loss = loss / args.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            
            micro_step += 1
            comp_time = time.time() - start_comp
            comp_time_total += comp_time
            log_comp_time += comp_time
            
            # log一下看看速度
            if rank == 0:
                logger.info(f"Micro step {micro_step}, Loss: {loss.item():.4f}, Time: {comp_time:.4f} seconds")
            
            # 只有在积累了足够的梯度后才更新参数
            if micro_step % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # model.clip_grad_norm_(1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                
                # 每隔 log_interval 步，输出一次日志
                if global_step % args.log_interval == 0:
                    # 使用累积的损失值进行记录
                    avg_loss = running_loss / args.gradient_accumulation_steps
                    # Get current learning rate and scale factor
                    current_lr = scheduler.get_last_lr()[0]
                    current_scale = scaler.get_scale()
                    
                    logger.info(f"Epoch {epoch_idx+1}/{args.epochs}, Step {epoch_step} (Global: {global_step}) - "
                                f"Loss: {avg_loss:.4f}, LR: {current_lr:.6f}, "
                                f"Scale: {current_scale:.1f}, CompTime: {log_comp_time:.4f}s")
                    
                    # 记录到 wandb (只在主进程中)
                    if rank == 0 and args.use_wandb:
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/learning_rate": current_lr,
                            "train/computation_time": log_comp_time,
                            "train/grad_scale": current_scale,
                            "step": global_step,
                        }, step=global_step)
                    
                    log_comp_time = 0.0
                    running_loss = 0.0
                
                # 每隔 eval_interval 步，在验证集上评估
                current_metric = None
                is_best = False
                if rank==0 and global_step % args.eval_interval == 0 and eval_dataloader is not None:
                    eval_results = evaluate(model, eval_dataloader, device, 
                                            task_type, args.use_amp, args.amp_type)

                    # 根据任务类型输出不同的评估指标
                    if task_type == "classification":
                        current_metric = eval_results['loss']  # 或使用 accuracy
                        logger.info(f"评估结果 [步数 {global_step}] - 验证损失: {eval_results['loss']:.4f}, 准确率: {eval_results['accuracy']:.4f}")
                    else:
                        current_metric = eval_results['loss']  # 或使用 perplexity
                        logger.info(f"评估结果 [步数 {global_step}] - 验证损失: {eval_results['loss']:.4f}, 困惑度: {eval_results['perplexity']:.4f}")
                    
                    # 检查是否为最佳模型
                    if current_metric < best_metric:
                        best_metric = current_metric
                        is_best = True
                    
                    if args.use_wandb:
                        wandb_log = {"eval/loss": eval_results['loss']}
                        if "accuracy" in eval_results:
                            wandb_log["eval/accuracy"] = eval_results['accuracy']
                        if "perplexity" in eval_results:
                            wandb_log["eval/perplexity"] = eval_results['perplexity']
                        wandb.log(wandb_log, step=global_step)
                
                # 每隔 checkpoint_interval 步，保存检查点
                if args.checkpoint_dir and global_step % args.checkpoint_interval == 0:
                    # 所有rank都保存检查点
                    logger.info(f"Rank {rank}: 保存检查点，当前步数: {global_step}")
                    save_streaming_diloco_checkpoint(
                        args.checkpoint_dir, model, optimizer, scheduler,
                        shard_tracker, scaler,
                        epoch_idx, global_step, micro_step,
                        comp_time_total, comm_time_total,
                        current_metric, is_best, rank,
                        False, args.max_checkpoints, logger
                    )
                
                global_step += 1
                epoch_step += 1
            
                # 检查是否需要发送或接收模型分片
                for shard_idx in range(num_shards):
                    # Check if we need to receive this shard
                    if shard_tracker[shard_idx]["next_receive_step"] == global_step:
                        comm_time = sync_model_optimized(
                            model, shard_tracker, shard_idx, world_size, logger,
                            comm_delay, num_shards, args.alpha)
                        
                        comm_time_total += comm_time
                        
                        # 记录通信时间到 wandb (只在主进程中)
                        if rank == 0 and args.use_wandb:
                            wandb.log({
                                "communication_time": comm_time,
                                "sync_shard_idx": shard_idx,
                            }, step=global_step)

                    # Check if we need to record/send this shard
                    # Send current params in staged_params
                    relative_step = global_step % args.sync_interval
                    if relative_step == base_sync_points[shard_idx]:
                        shard_tracker[shard_idx]["sent_at_step"] = global_step
                        # Calculate next receive step based on current send
                        shard_tracker[shard_idx]["next_receive_step"] = global_step + args.delay_steps
                        
                        # 使用clone()创建参数的深拷贝而不仅是引用
                        model_params_list = list(model.parameters())
                        start_idx = shard_tracker[shard_idx]["start_idx"]
                        end_idx = shard_tracker[shard_idx]["end_idx"]
                        shard_tracker[shard_idx]["staged_params"] = [p.data.clone() for p in model_params_list[start_idx:end_idx]]
                        del model_params_list
                        logger.info(f"分片 {shard_idx+1} 已记录并发送，当前步数: {global_step}, 将在步数 {shard_tracker[shard_idx]['next_receive_step']} 接收")
                
        # 每个epoch结束时输出一次总结
        if rank == 0:
            logger.info(f"完成 Epoch {epoch_idx+1}/{args.epochs}, 当前步数: {global_step}")
        
        # 每个epoch结束保存检查点
        if args.checkpoint_dir:
            logger.info(f"Rank {rank}: 保存 Epoch {epoch_idx+1} 检查点")
            save_streaming_diloco_checkpoint(
                args.checkpoint_dir, model, optimizer, scheduler,
                shard_tracker, scaler,
                epoch_idx, global_step, micro_step,
                comp_time_total, comm_time_total,
                None, False, rank,
                False, args.max_checkpoints, logger
            )
        
        if global_step >= args.total_steps:
            if eval_dataloader is not None:
                # 进行最终评估
                final_metrics = {}
                final_eval_results = evaluate(model, eval_dataloader, device, 
                                             task_type, args.use_amp, args.amp_type)
                for k, v in final_eval_results.items():
                    final_metrics[f"final/eval_{k}"] = v
                
                logger.info(f"最终评估结果: {final_metrics}")
                if args.use_wandb and rank == 0:
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
        logger.info(f"实际批量大小: {args.batch_size * args.gradient_accumulation_steps} (批量大小 {args.batch_size} * 梯度累积步数 {args.gradient_accumulation_steps})")

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 初始化分布式训练相关环境
    local_rank, rank, world_size = init_distributed()
    device = torch.device("cuda", local_rank)
    torch.set_float32_matmul_precision("high")
    args.amp_type = torch.bfloat16 if args.amp_type == 'bf16' else torch.float16
    
    # 设置梯度累积步数
    if args.effective_batch_size is not None:
        per_device_batch_size = args.batch_size
        total_batch_size = args.effective_batch_size * world_size
        args.gradient_accumulation_steps = args.effective_batch_size // per_device_batch_size
        if rank == 0:
            print(f"Effective batch size: {args.effective_batch_size}")
            print(f"Per-device batch size: {per_device_batch_size}")
            print(f"Total batch size (across all devices): {total_batch_size}")
            print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
            print(f"Actual effective batch size: {total_batch_size * args.gradient_accumulation_steps}")
    
    # 设置 logger
    logger = setup_logging(rank, args.log_dir)
    logger.info(f"初始化完成。当前 Rank: {rank}, 总进程数: {world_size}")
    logger.info(f"梯度累积步数: {args.gradient_accumulation_steps}, 实际批量大小: {args.batch_size * args.gradient_accumulation_steps * world_size} (全局)")

    # 检查是否需要恢复训练，获取WandB run ID
    loaded_wandb_run_id = None
    global_step_to_resume = None
    if args.resume and args.checkpoint_dir and rank == 0:
        latest_checkpoint_path = get_latest_checkpoint(args.checkpoint_dir, logger, rank)
        # 只有rank 0需要检查wandb ID
        if latest_checkpoint_path:
            logger.info(f"Rank 0: Found potential checkpoint for resume: {latest_checkpoint_path}")
            try:
                # 只加载检查点元数据以获取 ID，避免加载整个模型到 CPU
                ckpt_data = torch.load(latest_checkpoint_path, map_location='cpu')
                loaded_wandb_run_id = ckpt_data.get("wandb_run_id")
                global_step_to_resume = ckpt_data.get("global_step", None)
                if loaded_wandb_run_id:
                    logger.info(f"Rank 0: Found WandB run ID in checkpoint: {loaded_wandb_run_id}, global step: {global_step_to_resume}")
                else:
                    logger.warning("Rank 0: Checkpoint found, but no WandB run ID saved within it. Starting a new WandB run.")
                del ckpt_data # 释放内存
            except Exception as e:
                logger.error(f"Rank 0: Failed to load checkpoint metadata for WandB ID: {e}", exc_info=True)
    
    # 初始化 wandb (只在主进程中)
    if rank == 0 and args.use_wandb:
        wandb_config = vars(args)
        wandb_id_to_use = loaded_wandb_run_id if args.resume else None # 仅在 resume 时尝试使用旧 ID
        resume_status = "must" if wandb_id_to_use else None
        
        # 如果没有指定实验名称，自动生成一个
        if args.wandb_name is None:
            args.wandb_name = f"streaming_{args.model_name}_{args.dataset_name}_sync{args.sync_interval}_ns{args.num_shards}_ds{args.delay_steps}"
            if args.outer_lr != 1.0:
                args.wandb_name += f"_olr{args.outer_lr}"
                if args.use_nesterov:
                    args.wandb_name += "_nesterov"
            if args.alpha != 0.5:
                args.wandb_name += f"_a{args.alpha}"
            if args.gradient_accumulation_steps > 1:
                args.wandb_name += f"_acc{args.gradient_accumulation_steps}"
        
        try:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                # id=wandb_id_to_use if wandb_id_to_use else None,
                dir='/data/yzhu/distrain/runs/streaming_diloco',
                config=wandb_config,
                # resume=resume_status,
                resume_from=f"{wandb_id_to_use}?_step={global_step_to_resume}" if (wandb_id_to_use and global_step_to_resume) else None
            )
            logger.info(f"Weights & Biases 初始化完成。项目: {args.wandb_project}, 实验: {args.wandb_name}")
            # 如果是恢复运行，且实际恢复的ID与加载的ID不同，记录一下
            if resume_status == "must" and wandb.run and wandb.run.id != loaded_wandb_run_id:
                 logger.warning(f"WandB 恢复时使用了新的 ID: {wandb.run.id} (预期 ID: {loaded_wandb_run_id})")
            # 将最终的 run id 保存回 args，以便保存到检查点
            if wandb.run: args.wandb_run_id = wandb.run.id
            
        except Exception as e:
            logger.error(f"WandB 初始化失败: {e}", exc_info=True)
            args.use_wandb = False # 初始化失败则禁用

    # 记录训练参数
    if rank == 0:
        logger.info("--- 最终训练参数 ---")
        for k, v in vars(args).items():
             logger.info(f"{k}: {v}")
        logger.info("--------------------")
        # 保存配置到文件
        config_path = os.path.join(args.log_dir, "config.txt")
        try:
            with open(config_path, "w") as f:
                 for k, v in vars(args).items():
                     f.write(f"{k}: {v}\n")
                 f.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            logger.info(f"训练参数已保存到: {config_path}")
        except IOError as e:
             logger.error(f"无法保存参数到文件: {e}")

    # 加载数据集和对应的模型与分词器（每个节点各自加载独立副本）
    train_dataloader, eval_dataloader, tokenizer, model, task_type = load_data_and_model(
        args.dataset_name, args.model_name, 
        args.batch_size, args.eval_batch_size,
        rank, world_size
    )
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型和数据加载完成。模型参数量: {num_params:,}")
    logger.info(f"预计模型大小：{num_params * (2 if args.use_amp else 4) / 1e6:.2f} MB")
    logger.info(f"任务类型: {task_type}")
    args.task_type = task_type # 将检测到的任务类型存入args
    
    # 配置优化器和学习率调度器
    if args.weight_decay > 0:
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.total_steps,
    )
    
    # 计算通信延迟
    comm_delay = calc_comm_delay(model, world_size, logger, simulated_bandwidth_mbps=args.bandwidth)
    
    # 使用传入的参数进行训练，传入训练和验证数据加载器
    train_start_time = time.time()
    try:
        train(model, (train_dataloader, eval_dataloader), optimizer,
              scheduler, device, logger, world_size, rank, comm_delay,
              task_type, args)
    except Exception as e:
         logger.error(f"训练主函数捕获到错误: {e}", exc_info=True)
    finally:
        # --- Cleanup ---
        train_duration = time.time() - train_start_time
        logger.info(f"总训练时长: {train_duration:.2f} 秒 ({train_duration/3600:.2f} 小时)")
        if rank == 0:
            # Append end time to config file
            try:
                with open(os.path.join(args.log_dir, "config.txt"), "a") as f:
                    f.write(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Total duration: {train_duration:.2f} seconds\n")
            except IOError as e:
                logger.error(f"Could not write end time to config file: {e}")

            if args.use_wandb and wandb.run is not None:
                logger.info("Finishing WandB run...")
                wandb.finish()

        logger.info("Destroying process group...")
        dist.destroy_process_group()
        logger.info("Process group destroyed.")

if __name__ == "__main__":
    main()
