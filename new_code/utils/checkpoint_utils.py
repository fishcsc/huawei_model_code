import os
import time
import glob
import torch
import torch.distributed as dist
from pathlib import Path
import logging
import re
import shutil
import wandb
from typing import Dict, Any, Optional, Tuple, Union


def get_latest_checkpoint(checkpoint_dir: str, logger: Optional[logging.Logger] = None, rank: int = 0) -> str:
    """
    获取指定目录中最新的检查点目录和对应rank的文件路径。
    
    Args:
        checkpoint_dir: 检查点根目录
        logger: 日志记录器
        rank: 进程的rank，用于查找特定rank的检查点
    
    Returns:
        str: 最新检查点文件的路径，如果没有找到则返回空字符串
    """
    if not os.path.exists(checkpoint_dir):
        if logger:
            logger.warning(f"检查点目录不存在: {checkpoint_dir}")
        return ""
    
    # 查找所有步数检查点目录
    checkpoint_dirs = glob.glob(os.path.join(checkpoint_dir, "step_*"))
    if not checkpoint_dirs:
        if logger:
            logger.info(f"在目录 {checkpoint_dir} 中未找到检查点目录")
        return ""
    
    # 按修改时间排序
    checkpoint_dirs.sort(key=os.path.getmtime, reverse=True)
    latest_dir = checkpoint_dirs[0]
    
    # 在最新的目录中查找特定rank的检查点文件
    rank_file = os.path.join(latest_dir, f"rank_{rank}.pt")
    if not os.path.exists(rank_file):
        if logger:
            logger.info(f"未找到rank {rank}的检查点文件: {rank_file}")
        return ""
    
    if logger:
        logger.info(f"找到最新检查点: {rank_file}")
    
    return rank_file


def _clean_old_checkpoints(checkpoint_dir: str, max_keep: int, logger: Optional[logging.Logger] = None) -> None:
    """
    清理旧的检查点目录，只保留最近的N个。
    
    Args:
        checkpoint_dir: 检查点根目录
        max_keep: 要保留的最大检查点数量
        logger: 日志记录器
    """
    # 查找所有步数检查点目录
    checkpoint_dirs = glob.glob(os.path.join(checkpoint_dir, "step_*"))
    
    # 如果检查点数量小于等于保留数量，无需删除
    if len(checkpoint_dirs) <= max_keep:
        return
    
    # 按修改时间排序
    checkpoint_dirs.sort(key=os.path.getmtime, reverse=True)
    
    # 删除旧的检查点目录
    for old_dir in checkpoint_dirs[max_keep:]:
        if logger:
            logger.info(f"删除旧检查点目录: {old_dir}")
        shutil.rmtree(old_dir)


def save_diloco_checkpoint(
    checkpoint_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    original_snapshot: Optional[torch.nn.Module] = None,
    outer_optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    epoch: int = 0,
    global_step: int = 0,
    micro_step: int = 0,
    comp_time_total: float = 0.0,
    comm_time_total: float = 0.0,
    metric_value: Optional[float] = None,
    is_best: bool = False,
    rank: int = 0,
    save_model_only: bool = False,
    max_checkpoints: int = 3,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    保存检查点，包括模型、优化器、调度器等状态。每个rank保存自己的状态到对应step的目录中。
    
    Args:
        checkpoint_dir: 保存检查点的根目录路径
        model: 当前模型
        optimizer: 主优化器
        scheduler: 学习率调度器
        original_snapshot: DiLoCo原始模型快照
        outer_optimizer: 外部优化器(如果使用DiLoCo)
        scaler: GradScaler实例(用于AMP)
        epoch: 当前轮次
        global_step: 当前全局步数
        micro_step: 当前微步数(梯度累积相关)
        comp_time_total: 总计算时间
        comm_time_total: 总通信时间
        metric_value: 用于命名的指标值(如验证精度)
        is_best: 是否为最佳模型
        rank: 当前进程的rank
        save_model_only: 是否只保存模型(不保存优化器等)
        max_checkpoints: 保留的最大检查点数量
        logger: 日志记录器
        
    Returns:
        str: 保存的检查点文件路径
    """
    # 确保根目录存在
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 构建检查点目录名 - 只使用step，不包含metric值
    step_dir_name = f"step_{global_step}"
    
    # 确保所有rank使用相同的目录名
    dist.barrier()
    
    # 创建步数目录
    step_dir = os.path.join(checkpoint_dir, step_dir_name)
    os.makedirs(step_dir, exist_ok=True)
    
    # 构建rank特定的文件名
    checkpoint_file = f"rank_{rank}.pt"
    tmp_checkpoint_path = os.path.join(step_dir, f"tmp_{checkpoint_file}")
    checkpoint_path = os.path.join(step_dir, checkpoint_file)
    
    # 准备要保存的状态字典
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "micro_step": micro_step,
        "comp_time_total": comp_time_total,
        "comm_time_total": comm_time_total,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if not save_model_only else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler and not save_model_only else None,
        "rank": rank,  # 保存rank信息
        # 保存随机数状态以确保可复现性
        "rng_states": {
            "python": None,  # Python不能序列化
            "numpy": None,   # 可能会很大，暂不保存
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
    }
    
    # 添加DiLoCo特有状态
    if original_snapshot is not None:
        checkpoint["original_snapshot_state_dict"] = original_snapshot.state_dict()
    
    if outer_optimizer is not None and not save_model_only:
        checkpoint["outer_optimizer_state_dict"] = outer_optimizer.state_dict()
    
    # 添加AMP scaler状态
    if scaler is not None and not save_model_only:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    
    # 保存wandb run ID (只在rank 0保存)
    if rank == 0 and wandb.run is not None:
         checkpoint["wandb_run_id"] = wandb.run.id
         if logger: logger.info(f"Saving WandB run ID: {wandb.run.id}")
    
    if logger:
        logger.info(f"Rank {rank}: 保存检查点到 {checkpoint_path}")
    
    # 保存临时文件然后重命名，防止保存过程中断造成文件损坏
    torch.save(checkpoint, tmp_checkpoint_path)
    os.replace(tmp_checkpoint_path, checkpoint_path)
    
    # 如果是最佳模型并且是rank 0，更新best_step.txt和保存最佳模型
    if is_best and rank == 0:
        # 创建best_model目录
        best_dir = os.path.join(checkpoint_dir, "best_model")
        os.makedirs(best_dir, exist_ok=True)
        best_path = os.path.join(best_dir, "rank_0.pt")
        shutil.copy2(checkpoint_path, best_path)
        
        # 创建一个best_step.txt文件记录最佳步数和指标值
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(os.path.join(checkpoint_dir, "best_step.txt"), "w") as f:
            f.write(f"step: {global_step}\n")
            f.write(f"metric: {metric_value if metric_value is not None else 'N/A'}\n")
            f.write(f"time: {current_time}\n")
        
        if logger:
            logger.info(f"Rank {rank}: 保存最佳模型到 {best_path}, 指标值: {metric_value}")
    dist.barrier()
    if rank != 0:
        # 保存各自的检查点到best_model目录
        best_dir = os.path.join(checkpoint_dir, "best_model", f"rank_{rank}.pt")
        shutil.copy2(checkpoint_path, best_dir)
    
    # 清理旧检查点目录 (只需要rank 0执行一次即可)
    if rank == 0 and max_checkpoints > 0:
        _clean_old_checkpoints(checkpoint_dir, max_checkpoints, logger)
    
    # 确保所有进程同步
    dist.barrier()
    
    return checkpoint_path


def load_diloco_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    original_snapshot: Optional[torch.nn.Module] = None,
    outer_optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    rank: int = 0,
    map_location: str = "cpu",
    load_model_only: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    加载检查点，恢复模型、优化器、调度器等状态。每个rank加载自己的状态。
    
    Args:
        checkpoint_path: 检查点目录路径或文件路径
        model: 要恢复的模型
        optimizer: 要恢复的主优化器
        scheduler: 要恢复的学习率调度器
        original_snapshot: 要恢复的DiLoCo原始模型快照
        outer_optimizer: 要恢复的外部优化器
        scaler: 要恢复的GradScaler实例
        rank: 当前进程的rank
        map_location: 加载模型的设备
        load_model_only: 是否只加载模型(不加载优化器等)
        logger: 日志记录器
        
    Returns:
        Dict[str, Any]: 包含训练状态的字典(epoch, global_step等)
    """
    # 如果是文件路径，直接使用
    if os.path.isfile(checkpoint_path):
        checkpoint_file = checkpoint_path
    # 如果是目录路径，且是step_*格式，查找对应rank的文件
    elif os.path.isdir(checkpoint_path) and os.path.basename(checkpoint_path).startswith("step_"):
        checkpoint_file = os.path.join(checkpoint_path, f"rank_{rank}.pt")
    # 如果是根目录，查找最新检查点
    else:
        checkpoint_file = get_latest_checkpoint(checkpoint_path, logger, rank)
    
    if not os.path.exists(checkpoint_file):
        if logger:
            logger.error(f"检查点文件不存在: {checkpoint_file}")
        return {}
    
    if logger:
        logger.info(f"Rank {rank}: 加载检查点: {checkpoint_file}")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_file, map_location=map_location)
    
    # 验证rank匹配
    checkpoint_rank = checkpoint.get("rank", -1)
    if checkpoint_rank != rank and checkpoint_rank != -1:
        if logger:
            logger.warning(f"加载的检查点rank ({checkpoint_rank}) 与当前进程rank ({rank}) 不匹配!")
    
    # 恢复模型状态
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # 恢复DiLoCo特有状态
    if original_snapshot is not None and "original_snapshot_state_dict" in checkpoint:
        original_snapshot.load_state_dict(checkpoint["original_snapshot_state_dict"])
    
    # 恢复优化器、调度器和AMP状态
    if not load_model_only:
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if outer_optimizer is not None and "outer_optimizer_state_dict" in checkpoint:
            outer_optimizer.load_state_dict(checkpoint["outer_optimizer_state_dict"])
        
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
    
    # 恢复随机数状态以确保可复现性
    if "rng_states" in checkpoint:
        rng_states = checkpoint["rng_states"]
        if rng_states["torch"] is not None:
            torch.set_rng_state(torch.tensor(rng_states["torch"], device="cpu", dtype=torch.uint8))
        if torch.cuda.is_available() and rng_states["cuda"] is not None:
            cuda_rng_list = rng_states["cuda"]
            cuda_rng_list = [torch.tensor(state, device="cpu", dtype=torch.uint8) for state in cuda_rng_list]
            torch.cuda.set_rng_state_all(cuda_rng_list)
    
    # 提取训练状态
    training_state = {
        "epoch": checkpoint.get("epoch", 0),
        "global_step": checkpoint.get("global_step", 0),
        "micro_step": checkpoint.get("micro_step", 0),
        "comp_time_total": checkpoint.get("comp_time_total", 0.0),
        "comm_time_total": checkpoint.get("comm_time_total", 0.0),
    }
    
    # 只有rank 0获取wandb_run_id
    if rank == 0:
        training_state["wandb_run_id"] = checkpoint.get("wandb_run_id")
    
    if logger:
        logger.info(f"Rank {rank}: 恢复到轮次 {training_state['epoch']}, 微步 {training_state['micro_step']}, 步数 {training_state['global_step']}")
    
    # 确保所有进程同步
    dist.barrier()
    
    return training_state


def save_streaming_diloco_checkpoint(
    checkpoint_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    shard_tracker: Dict[int, Dict[str, Any]], 
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    epoch: int = 0,
    global_step: int = 0,
    micro_step: int = 0,
    comp_time_total: float = 0.0,
    comm_time_total: float = 0.0,
    metric_value: Optional[float] = None,
    is_best: bool = False,
    rank: int = 0,
    save_model_only: bool = False,
    max_checkpoints: int = 3,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    保存流式DiLoCo检查点，包括模型、优化器、调度器和分片追踪器状态。
    
    Args:
        checkpoint_dir: 保存检查点的根目录路径
        model: 当前模型
        optimizer: 主优化器
        scheduler: 学习率调度器
        shard_tracker: 分片跟踪器状态
        scaler: GradScaler实例(用于AMP)
        epoch: 当前轮次
        global_step: 当前全局步数
        micro_step: 当前微步数(梯度累积相关)
        comp_time_total: 总计算时间
        comm_time_total: 总通信时间
        metric_value: 用于命名的指标值(如验证精度)
        is_best: 是否为最佳模型
        rank: 当前进程的rank
        save_model_only: 是否只保存模型(不保存优化器等)
        max_checkpoints: 保留的最大检查点数量
        logger: 日志记录器
        
    Returns:
        str: 保存的检查点文件路径
    """
    # 确保根目录存在
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 构建检查点目录名 - 只使用step，不包含metric值
    step_dir_name = f"step_{global_step}"
    
    # 确保所有rank使用相同的目录名
    dist.barrier()
    
    # 创建步数目录
    step_dir = os.path.join(checkpoint_dir, step_dir_name)
    os.makedirs(step_dir, exist_ok=True)
    
    # 构建rank特定的文件名
    checkpoint_file = f"rank_{rank}.pt"
    tmp_checkpoint_path = os.path.join(step_dir, f"tmp_{checkpoint_file}")
    checkpoint_path = os.path.join(step_dir, checkpoint_file)
    
    # 准备要保存的shard_tracker状态 - 需要特殊处理其中的tensor
    serializable_shard_tracker = {}
    for shard_idx, shard_info in shard_tracker.items():
        serializable_shard = {
            "start_idx": shard_info["start_idx"],
            "end_idx": shard_info["end_idx"],
            "sent_at_step": shard_info["sent_at_step"],
            "next_receive_step": shard_info["next_receive_step"],
            # 保存参数的深拷贝
            "params": [p.data.cpu().clone() for p in shard_info["params"]],
            # 保存staged_params的深拷贝(如果存在)
            "staged_params": None if shard_info["staged_params"] is None else 
                            [p.data.cpu().clone() for p in shard_info["staged_params"]],
        }
        # 如果存在outer_optimizer，保存其状态字典
        if shard_info.get("outer_optimizer") is not None and not save_model_only:
            serializable_shard["outer_optimizer_state_dict"] = shard_info["outer_optimizer"].state_dict()
        
        serializable_shard_tracker[shard_idx] = serializable_shard
    
    # 准备要保存的状态字典
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "micro_step": micro_step,
        "comp_time_total": comp_time_total,
        "comm_time_total": comm_time_total,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if not save_model_only else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler and not save_model_only else None,
        "shard_tracker": serializable_shard_tracker,
        "rank": rank,  # 保存rank信息
        # 保存随机数状态以确保可复现性
        "rng_states": {
            "python": None,  # Python不能序列化
            "numpy": None,   # 可能会很大，暂不保存
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
    }
    
    # 添加AMP scaler状态
    if scaler is not None and not save_model_only:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    
    # 保存wandb run ID (只在rank 0保存)
    if rank == 0 and wandb.run is not None:
         checkpoint["wandb_run_id"] = wandb.run.id
         if logger: logger.info(f"Saving WandB run ID: {wandb.run.id}")
    
    if logger:
        logger.info(f"Rank {rank}: 保存检查点到 {checkpoint_path}")
    
    # 保存临时文件然后重命名，防止保存过程中断造成文件损坏
    torch.save(checkpoint, tmp_checkpoint_path)
    os.replace(tmp_checkpoint_path, checkpoint_path)
    
    # 如果是最佳模型并且是rank 0，更新best_step.txt和保存最佳模型
    if is_best and rank == 0:
        # 创建best_model目录
        best_dir = os.path.join(checkpoint_dir, "best_model")
        os.makedirs(best_dir, exist_ok=True)
        best_path = os.path.join(best_dir, "rank_0.pt")
        shutil.copy2(checkpoint_path, best_path)
        
        # 创建一个best_step.txt文件记录最佳步数和指标值
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(os.path.join(checkpoint_dir, "best_step.txt"), "w") as f:
            f.write(f"step: {global_step}\n")
            f.write(f"metric: {metric_value if metric_value is not None else 'N/A'}\n")
            f.write(f"time: {current_time}\n")
        
        if logger:
            logger.info(f"Rank {rank}: 保存最佳模型到 {best_path}, 指标值: {metric_value}")
    dist.barrier()
    if rank != 0:
        # 保存各自的检查点到best_model目录
        best_dir = os.path.join(checkpoint_dir, "best_model", f"rank_{rank}.pt")
        shutil.copy2(checkpoint_path, best_dir)
    # 清理旧检查点目录 (只需要rank 0执行一次即可)
    if rank == 0 and max_checkpoints > 0:
        _clean_old_checkpoints(checkpoint_dir, max_checkpoints, logger)
    
    # 确保所有进程同步
    dist.barrier()
    
    return checkpoint_path


def load_streaming_diloco_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    shard_tracker: Dict[int, Dict[str, Any]],
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    rank: int = 0,
    map_location: str = "cpu",
    load_model_only: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    加载流式DiLoCo检查点，恢复模型、优化器、调度器和分片追踪器状态。
    
    Args:
        checkpoint_path: 检查点目录路径或文件路径
        model: 要恢复的模型
        optimizer: 要恢复的主优化器
        scheduler: 要恢复的学习率调度器
        shard_tracker: 分片跟踪器，将被检查点中的状态更新
        scaler: 要恢复的GradScaler实例
        rank: 当前进程的rank
        map_location: 加载模型的设备
        load_model_only: 是否只加载模型(不加载优化器等)
        logger: 日志记录器
        
    Returns:
        Dict[str, Any]: 包含训练状态的字典(epoch, global_step等)
    """
    # 如果是文件路径，直接使用
    if os.path.isfile(checkpoint_path):
        checkpoint_file = checkpoint_path
    # 如果是目录路径，且是step_*格式，查找对应rank的文件
    elif os.path.isdir(checkpoint_path) and os.path.basename(checkpoint_path).startswith("step_"):
        checkpoint_file = os.path.join(checkpoint_path, f"rank_{rank}.pt")
    # 如果是根目录，查找最新检查点
    else:
        checkpoint_file = get_latest_checkpoint(checkpoint_path, logger, rank)
    
    if not os.path.exists(checkpoint_file):
        if logger:
            logger.error(f"检查点文件不存在: {checkpoint_file}")
        return {}
    
    if logger:
        logger.info(f"Rank {rank}: 加载检查点: {checkpoint_file}")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_file, map_location=map_location)
    
    # 验证rank匹配
    checkpoint_rank = checkpoint.get("rank", -1)
    if checkpoint_rank != rank and checkpoint_rank != -1:
        if logger:
            logger.warning(f"加载的检查点rank ({checkpoint_rank}) 与当前进程rank ({rank}) 不匹配!")
    
    # 恢复模型状态
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # 恢复分片追踪器状态
    if "shard_tracker" in checkpoint:
        checkpoint_shard_tracker = checkpoint["shard_tracker"]
        device = next(model.parameters()).device
        
        for shard_idx, ckpt_shard_info in checkpoint_shard_tracker.items():
            if shard_idx in shard_tracker:
                # 恢复基本状态信息
                shard_tracker[shard_idx]["start_idx"] = ckpt_shard_info["start_idx"]
                shard_tracker[shard_idx]["end_idx"] = ckpt_shard_info["end_idx"]
                shard_tracker[shard_idx]["sent_at_step"] = ckpt_shard_info["sent_at_step"]
                shard_tracker[shard_idx]["next_receive_step"] = ckpt_shard_info["next_receive_step"]
                
                # 恢复参数状态 - 将CPU tensor移动到正确的设备
                for i, param_cpu in enumerate(ckpt_shard_info["params"]):
                    shard_tracker[shard_idx]["params"][i].data.copy_(param_cpu.to(device))
                
                # 恢复staged_params状态(如果存在)
                if ckpt_shard_info["staged_params"] is not None:
                    if shard_tracker[shard_idx]["staged_params"] is None:
                        # 如果当前不存在staged_params，创建新的
                        shard_tracker[shard_idx]["staged_params"] = [
                            param_cpu.to(device).clone() for param_cpu in ckpt_shard_info["staged_params"]
                        ]
                    else:
                        # 如果已存在staged_params，更新值
                        for i, param_cpu in enumerate(ckpt_shard_info["staged_params"]):
                            shard_tracker[shard_idx]["staged_params"][i].data.copy_(param_cpu.to(device))
                
                # 恢复outer_optimizer状态(如果存在)
                if "outer_optimizer_state_dict" in ckpt_shard_info and shard_tracker[shard_idx].get("outer_optimizer") is not None:
                    shard_tracker[shard_idx]["outer_optimizer"].load_state_dict(ckpt_shard_info["outer_optimizer_state_dict"])
    
    # 恢复优化器、调度器和AMP状态
    if not load_model_only:
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
    
    # 恢复随机数状态以确保可复现性
    if "rng_states" in checkpoint:
        rng_states = checkpoint["rng_states"]
        if rng_states["torch"] is not None:
            torch.set_rng_state(torch.tensor(rng_states["torch"], device="cpu", dtype=torch.uint8))
        if torch.cuda.is_available() and rng_states["cuda"] is not None:
            cuda_rng_list = rng_states["cuda"]
            cuda_rng_list = [torch.tensor(state, device="cpu", dtype=torch.uint8) for state in cuda_rng_list]
            torch.cuda.set_rng_state_all(cuda_rng_list)
    
    # 提取训练状态
    training_state = {
        "epoch": checkpoint.get("epoch", 0),
        "global_step": checkpoint.get("global_step", 0),
        "micro_step": checkpoint.get("micro_step", 0),
        "comp_time_total": checkpoint.get("comp_time_total", 0.0),
        "comm_time_total": checkpoint.get("comm_time_total", 0.0),
    }
    
    # 只有rank 0获取wandb_run_id
    if rank == 0:
        training_state["wandb_run_id"] = checkpoint.get("wandb_run_id")
    
    if logger:
        logger.info(f"Rank {rank}: 恢复到轮次 {training_state['epoch']}, 步数 {training_state['global_step']}")
    
    # 确保所有进程同步
    dist.barrier()
    
    return training_state

