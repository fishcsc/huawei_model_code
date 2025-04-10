import time
import argparse
import torch
import torch.distributed as dist
from torch.optim import AdamW, SGD
from torch.amp import autocast, GradScaler
from copy import deepcopy
import wandb  # Add wandb import
from transformers import get_cosine_schedule_with_warmup

from utils.common import *    # common, 包括 logging，分布式训练初始化等
from utils.load_data_model import *  # 加载数据集和模型
from utils.checkpoint_utils import load_diloco_checkpoint, save_diloco_checkpoint, get_latest_checkpoint
from utils.arg_utils import parse_args


# 批量 all-reduce 操作以提高通信效率
def sync_model_optimized(model, original_model, outer_optimizer, world_size, logger, comm_delay=None):
    """
    Synchronizes model parameters across distributed processes.

    Handles two cases:
    1. If outer_optimizer is provided (DiLoCo-like): Calculates the update direction
       (original_param - current_param), aggregates it across workers, applies
       it to the original_model using the outer_optimizer, and copies the
       updated parameters back to the model.
    2. If outer_optimizer is None: Directly averages the model parameters across
       all workers using all_reduce.

    Args:
        model: The current model instance on the worker.
        original_model: A snapshot of the model before local steps (used with outer_optimizer).
        outer_optimizer: The optimizer for the global update step (e.g., SGD).
        world_size: Total number of distributed processes.
        logger: Logger instance.
        comm_delay: Optional simulated communication delay in seconds.

    Returns:
        The communication time in seconds.
    """
    sync_comm_time = 0.0
    with torch.no_grad():
        if outer_optimizer:
            # --- DiLoCo-like Synchronization ---
            grads_for_sync = []
            # Calculate the effective gradient (update direction) for the outer step
            for param, original_param in zip(model.parameters(), original_model.parameters()):
                # grad = original_param.data - param.data # Original DiLoCo direction
                grad_update = original_param.data - param.data
                # Assign this difference to the .grad field of the parameters
                # in original_model so the outer_optimizer can use it.
                # Ensure the grad attribute exists and is compatible
                if original_param.grad is None:
                    original_param.grad = torch.zeros_like(original_param.data)
                original_param.grad.copy_(grad_update)
                grads_for_sync.append(original_param.grad.data) # Collect grads for all-reduce

            # --- Batch Communication ---
            comm_start_sync = time.time()
            if grads_for_sync: # Proceed only if there are gradients to sync
                # Flatten all gradients into a single tensor for efficient all-reduce
                flat_grads = torch.cat([g.flatten() for g in grads_for_sync])
                # Aggregate gradients across all workers
                dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
                # Average the gradients
                flat_grads.div_(world_size)

                # --- Unflatten Gradients ---
                # Copy the averaged gradients back to the original_model's .grad attributes
                offset = 0
                for grad_sync in grads_for_sync:
                    numel = grad_sync.numel()
                    # Ensure grad_sync is used to determine the view shape
                    grad_sync.copy_(flat_grads[offset:offset+numel].view_as(grad_sync))
                    offset += numel
            sync_comm_time = time.time() - comm_start_sync

            # --- Outer Optimizer Step ---
            outer_optimizer.step()
            outer_optimizer.zero_grad(set_to_none=True) # Use set_to_none=True for potential memory savings

            # --- Update Worker Model ---
            # Copy the globally updated parameters from original_model back to the worker model
            for param, original_param in zip(model.parameters(), original_model.parameters()):
                param.data.copy_(original_param.data)

        else:
            # --- Direct Averaging Synchronization (No Outer Optimizer) ---
            comm_start_sync = time.time()
            # Directly average the parameters of the model across all workers
            # Note: With AMP, model parameters might be FP16. Averaging FP16 directly
            # can lead to precision loss. A more robust approach might involve
            # casting to FP32 before all-reduce and back, or synchronizing FP32
            # master weights if the optimizer maintains them (AdamW does).
            # For simplicity, we keep the direct FP16 averaging here.
            for param in model.parameters():
                dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
                param.data.div_(world_size)
            sync_comm_time = time.time() - comm_start_sync

    # --- Communication Delay Simulation ---
    if comm_delay:
        logger.info(f"Simulating communication time: {comm_delay:.4f} seconds")
        # Override measured time with simulated delay
        sync_comm_time = comm_delay
    else:
        logger.info(f"Synchronization communication time: {sync_comm_time:.4f} seconds")

    return sync_comm_time

def train(model, dataloader, optimizer, scheduler,
          device, logger, world_size, rank, comm_delay=None, 
          task_type="classification", args=None):
    """
    训练主循环，训练、发送、更新、评估、保存检查点等。
    """
    model.train()
    global_step = 0  # 实际步数（每accumulation_steps个小批量算一步）
    micro_step = 0   # 小批量计数
    comp_time_total = 0.0
    comm_time_total = 0.0
    log_comp_time = 0.0
    running_loss = 0.0  # 累积期间的总损失
    
    # 获取训练和验证数据加载器
    train_dataloader, eval_dataloader = dataloader
    # 保存一份参数计算外梯度
    original_snapshot = deepcopy(model)  
    if args.outer_lr != 1.0:
        outer_optimizer = SGD(
            original_snapshot.parameters(), lr=args.outer_lr, momentum=0.9, 
            nesterov=bool(args.use_nesterov)
        )
    else:
        outer_optimizer = None

    # 使用 AMP 进行混合精度训练
    if args.use_amp and args.amp_type == torch.float16:  
        scaler = GradScaler(enabled=True)
        logger.info("Initialized GradScaler for AMP.")
    else:
        scaler = GradScaler(enabled=False)
    
    # 从检查点恢复训练（如果指定）
    best_metric = float('inf')  # 用于跟踪最佳验证结果
    if args.resume and args.checkpoint_dir:
        # 直接从checkpoint_dir找当前rank的最新检查点
        resume_path = get_latest_checkpoint(args.checkpoint_dir, logger, rank)
        
        if resume_path:
            logger.info(f"Rank {rank}: 正在从检查点恢复训练: {resume_path}")
            # 加载检查点状态
            training_state = load_diloco_checkpoint(
                resume_path, model, optimizer, scheduler,
                original_snapshot, outer_optimizer, scaler,
                rank, device, False, logger
            )
            # 更新训练状态
            if training_state:
                global_step = training_state.get("global_step", 0)
                micro_step = training_state.get("micro_step", 0)
                comp_time_total = training_state.get("comp_time_total", 0.0)
                comm_time_total = training_state.get("comm_time_total", 0.0)
                logger.info(f"Rank {rank}: 恢复训练成功，当前步数: {global_step}")
            else:
                logger.info(f"Rank {rank}: 检查点加载失败，将从头开始训练")
        else:
            logger.info(f"Rank {rank}: 未找到有效检查点，将从头开始训练")
    
    # 梯度累积初始化时清零梯度
    optimizer.zero_grad()
    
    try:  # 添加异常处理以便在训练中断时保存检查点
        for epoch in range(args.epochs):
            epoch_step = 0  # 记录每个epoch内的step数（实际步数）
            logger.info(f"开始 Epoch {epoch+1}/{args.epochs}")
            
            # 设置数据加载器的 epoch，确保恢复训练时的数据顺序正确
            if hasattr(train_dataloader, 'sampler') and hasattr(train_dataloader.sampler, 'set_epoch'):
                train_dataloader.sampler.set_epoch(epoch)
                logger.info(f"设置 sampler epoch: {epoch}")
            else:
                logger.info("当前 dataloader 没有可设置 epoch 的 sampler")
            
            for batch in train_dataloader:
                if global_step >= args.total_steps:
                    logger.info(f"到达 ({args.total_steps}) 步数，停止训练。")
                    break
                
                # 将 batch 数据移动到 GPU 上
                batch = {k: v.to(device) for k, v in batch.items()}
                start_comp = time.time()
                
                # 前向计算、损失计算、反向传播
                with autocast(device_type=device.type, enabled=args.use_amp, dtype=args.amp_type):
                    if task_type == "language_modeling":
                        outputs = model(**batch)
                    elif task_type == "classification":
                        outputs = model(input_ids=batch["input_ids"],
                                      attention_mask=batch["attention_mask"],
                                      labels=batch["label"])
                    else:
                        raise ValueError(f"Unknown task_type: {task_type}")
                    loss = outputs.loss
                
                # 缩放损失以适应梯度累积
                running_loss += loss.item() 
                scaled_loss = loss / args.gradient_accumulation_steps
                scaler.scale(scaled_loss).backward()

                micro_step += 1
                comp_time_micro = time.time() - start_comp
                comp_time_total += comp_time_micro
                log_comp_time += comp_time_micro
                
                # log一下看看速度
                if rank == 0:
                    logger.info(f"Micro step {micro_step}, Loss: {loss.item():.4f}, Time: {comp_time_micro:.4f} seconds")
                
                # 只有在积累了足够的梯度后才更新参数
                if micro_step % args.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    
                    # 每隔 log_interval 步，输出一次日志
                    if global_step % args.log_interval == 0:
                        # 使用累积的损失值进行记录
                        avg_loss = running_loss / (args.gradient_accumulation_steps * args.log_interval)
                        # Get current learning rate and scale factor
                        current_lr = scheduler.get_last_lr()[0]
                        current_scale = scaler.get_scale()

                        logger.info(f"Epoch {epoch+1}, Step: {global_step}/{args.total_steps}, "
                                    f"Loss: {avg_loss:.4f}, LR: {current_lr:.6f}, "
                                    f"Scale: {current_scale:.1f}, CompTime(avg): {log_comp_time/args.log_interval:.4f}s")
                        
                        # Log to WandB if enabled (only on rank 0)
                        if rank == 0 and args.use_wandb:
                            wandb.log({
                                "train/loss": avg_loss,
                                "train/learning_rate": current_lr,
                                "train/computation_time_interval_avg": log_comp_time / args.log_interval,
                                "train/grad_scale": current_scale,
                                "step": global_step,
                            }, step=global_step)
                        
                        log_comp_time = 0.0
                        running_loss = 0.0
                    
                    # 每隔 eval_interval 步，在验证集上评估
                    current_metric = None
                    is_best = False
                    if rank == 0 and global_step % args.eval_interval == 0 and eval_dataloader is not None:
                        # 根据任务类型输出不同的评估指标
                        eval_results = evaluate(model, eval_dataloader, device, 
                                                task_type, args.use_amp, args.amp_type)
                        
                        # 记录验证指标并检查是否为最佳模型
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
                        save_diloco_checkpoint(
                            args.checkpoint_dir, model, optimizer, scheduler,
                            original_snapshot, outer_optimizer, scaler,
                            epoch, global_step, micro_step,
                            comp_time_total, comm_time_total,
                            current_metric, is_best, rank,
                            False, args.max_checkpoints, logger
                        )
                    
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
            logger.info(f"Rank {rank}: 完成 Epoch {epoch+1}/{args.epochs}, 当前步数: {global_step}")
            if args.checkpoint_dir:
                save_diloco_checkpoint(
                    args.checkpoint_dir, model, optimizer, scheduler,
                    original_snapshot, outer_optimizer, scaler,
                    epoch, global_step, micro_step,
                    comp_time_total, comm_time_total,
                    None, False, rank,
                    False, args.max_checkpoints, logger
                )
            
            if global_step >= args.total_steps:
                if rank == 0 and eval_dataloader is not None:
                    # 进行最终评估
                    final_metrics = {}
                    final_eval_results = evaluate(model, eval_dataloader, device, 
                                                task_type, args.use_amp, args.amp_type)
                    for k, v in final_eval_results.items():
                        final_metrics[f"final/eval_{k}"] = v
                    
                    logger.info(f"最终评估结果: {final_metrics}")
                    if args.use_wandb:
                        wandb.log(final_metrics, step=global_step)
                break
    except Exception as e:
        logger.error(f"Rank {rank}: 训练过程中发生异常: {e}", exc_info=True)
        raise e 
    finally:
        # 保存最终检查点
        if args.checkpoint_dir:
            logger.info(f"Rank {rank}: 保存最终检查点: {global_step}")
            save_diloco_checkpoint(
                args.checkpoint_dir, model, optimizer, scheduler,
                original_snapshot, outer_optimizer, scaler,
                epoch, global_step, micro_step,
                comp_time_total, comm_time_total,
                None, False, rank,
                False, args.max_checkpoints, logger
            )
    
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
    # 如果设置了effective_batch_size，计算gradient_accumulation_steps
    if args.effective_batch_size is not None:
        per_device_batch_size = args.batch_size
        total_batch_size = args.effective_batch_size * world_size
        # args.gradient_accumulation_steps = max(1, args.effective_batch_size // total_batch_size)
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

    # 检查是否需要恢复训练
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
                    logger.info(f"Rank 0: Found WandB run ID in checkpoint: {loaded_wandb_run_id},global step: {global_step_to_resume}")
                else:
                    logger.warning("Rank 0: Checkpoint found, but no WandB run ID saved within it. Starting a new WandB run.")
                del ckpt_data # 释放内存
            except Exception as e:
                logger.error(f"Rank 0: Failed to load checkpoint metadata for WandB ID: {e}", exc_info=True)

    # dist.barrier()
    
    # 初始化 wandb (只在主进程中)
    if rank == 0 and args.use_wandb:
        wandb_config = vars(args)
        wandb_id_to_use = loaded_wandb_run_id if args.resume else None # 仅在 resume 时尝试使用旧 ID
        resume_status = "must" if wandb_id_to_use else None
        # if loaded_wandb_run_id is None:
        # 如果没有指定实验名称，自动生成一个
        if args.wandb_name is None:
            args.wandb_name = f"diloco_{args.model_name}_{args.dataset_name}_sync{args.sync_interval}"
            args.wandb_name += f"_bw{args.bandwidth}" if args.bandwidth else ""
            if args.outer_lr != 1.0:
                args.wandb_name += f"_olr{args.outer_lr}"
                if args.use_nesterov:
                    args.wandb_name += "_nesterov"
            if args.gradient_accumulation_steps > 1:
                args.wandb_name += f"_acc{args.gradient_accumulation_steps}"
        try:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                # id=wandb_id_to_use if wandb_id_to_use else None,
                dir="/data/yzhu/distrain/runs/diloco", 
                config=wandb_config,
                # resume=resume_status,
                resume_from=f"{wandb_id_to_use}?_step={global_step_to_resume}" if (wandb_id_to_use and global_step_to_resume) else None
            )
            logger.info(f"Weights & Biases 初始化完成。项目: {args.wandb_project}, 名称: {args.wandb_name}, "
                        f"ID: {wandb.run.id if wandb.run else 'N/A'}, Resume: {resume_status}")
            # 如果是恢复运行，且实际恢复的ID与加载的ID不同（可能wandb服务器上没有该ID了），记录一下
            if resume_status == "must" and wandb.run and wandb.run.id != loaded_wandb_run_id:
                 logger.warning(f"WandB 恢复时使用了新的 ID: {wandb.run.id} (预期 ID: {loaded_wandb_run_id})")
            # 将最终的 run id 保存回 args，以便保存到检查点
            if wandb.run: args.wandb_run_id = wandb.run.id

        except Exception as e:
            logger.error(f"WandB 初始化失败: {e}", exc_info=True)
            args.use_wandb = False # 初始化失败则禁用
            
    # 记录最终参数 (包括动态计算的)
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
    logger.info(f"预计模型大小：{num_params * (2 if args.use_amp else 2 ) / 1e6:.2f} MB")
    logger.info(f"任务类型: {task_type}")
    args.task_type = task_type # 将检测到的任务类型存入args

    # 配置优化器
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
         logger.error(f"训练主函数捕获到错误: {e}", exc_info=True) # Log traceback
         # Optionally re-raise or handle cleanup
        #  raise e
    finally:
        # --- Cleanup ---
        train_duration = time.time() - train_start_time
        logger.info(f"总训练时长: {train_duration:.2f} 秒 ({train_duration/3600:.2f} 小时)")
        if rank == 0:
            # Append end time to config file
            config_path = os.path.join(args.log_dir, f"config_rank{rank}.txt")
            try:
                with open(config_path, "a") as f:
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
