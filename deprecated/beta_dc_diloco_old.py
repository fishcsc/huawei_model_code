import os
os.environ['HF_HOME']='/data/hfhub'
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
os.environ['CUDA_LAUNCH_BLOCKING']='1'
import time
import torch
import torch.distributed as dist
from torch.optim import AdamW, SGD
import argparse
from utils.common import *    # 从 utils/util.py 中导入函数
from utils.delay_compensate import *    # 延迟补偿相关内容


# 更新海森矩阵的估计
def update_hessian_est(shard_tracker, model, rho=0.01):
    for i, shard in enumerate(shard_tracker.values()):
        start_idx = shard["start_idx"]
        end_idx = shard["end_idx"]
        hessian_list = shard["hessian_est"]
        
        idx_local = 0
        for p in list(model.parameters())[start_idx:end_idx]:
            if p.grad is not None:
                grad_sq = p.grad.data.square()
                hessian_list[idx_local].mul_(1 - rho).add_(grad_sq, alpha=rho)
            idx_local += 1

# 批量 all-reduce 操作以提高通信效率，实现传算分离
def sync_model_optimized(model, shard_tracker, sync_shard_idx, world_size, logger,
                        comm_delay, num_shards, alpha, args=None):
    """
    分块同步模型参数
    model: 当前工作模型
    shard_tracker: 用于外层优化的模型副本
    """
    cur_shard = shard_tracker[sync_shard_idx]
    with torch.no_grad():
        # 获取当前分片的参数范围
        start_idx = cur_shard["start_idx"]
        end_idx = cur_shard["end_idx"]
        model_params_list = list(model.parameters())
        sync_grads = [p1.data - p2.data for p1, p2 in zip(cur_shard["params"], model_params_list[start_idx:end_idx])]
        del model_params_list
        comm_start = time.time()
        # 批量通信 - 将当前分片的所有梯度拼接
        flat_grads = torch.cat([g.flatten() for g in sync_grads])
        dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
        flat_grads.div_(world_size)
        # 将梯度复制回参数的梯度
        offset = 0
        for grad in sync_grads:
            numel = grad.numel()
            grad.copy_(flat_grads[offset:offset+numel].view_as(grad))
            offset += numel
        comm_time = time.time() - comm_start
        del flat_grads, sync_grads
        
    if cur_shard['outer_optimizer']:
        # 使用外层优化器对当前分片的参数进行更新
        cur_shard['outer_optimizer'].step()
        cur_shard['outer_optimizer'].zero_grad()
    else:
        # 不适用优化器，直接对分片参数进行 all-reduce
        with torch.no_grad():
            for param in cur_shard["params"]:
                dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
                param.data.div_(world_size)
    
    with torch.no_grad():
        # TODO: Delay compensate。
        # streaming diloco 直接混合更新本地模型参数
        # 我们的方法会考虑到通信延迟带来的影响。具体来说我们会考虑到混合所用到的参数是比较老的。
        # 为了对延迟进行补偿，单纯混合可能并不是最好的选择。
        # 考虑使用泰勒展开对梯度进行补偿。
        model_params_list = list(model.parameters())
        for i, (param, updated_param, staged) in enumerate(zip(
            model_params_list[start_idx:end_idx],
            cur_shard["params"],
            cur_shard["staged_params"])):
            # 一阶变化量
            # param_diff = param.data - cur_shard["staged_params"][i].data
            param_diff = param.data - staged.data
            # 一阶补偿
            if args.first_order:
                updated_param.data.add_(param_diff, alpha=args.fbeta)
            # 二阶补偿
            if args.second_order:
                updated_param.data.add_(cur_shard["hessian_est"][i] * param_diff, 
                    alpha=args.sbeta)
            
            # Then do your original mixing
            param.data.mul_(alpha).add_(updated_param.data, alpha=1-alpha)
            
        # 将 staged_params 移动到当前分片的 params 中
        for cur_param, staged_param in zip(cur_shard["params"], cur_shard["staged_params"]):
            cur_param.data.copy_(staged_param.data)
            staged_param = None

    if comm_delay:
        logger.info(f"分片 {sync_shard_idx+1} 模拟通信时间: {comm_delay/num_shards:.4f} 秒")
        return comm_delay/num_shards
    else:
        logger.info(f"分片 {sync_shard_idx+1} 通信时间 (all-reduce): {comm_time:.4f} 秒")
        return comm_time    

def train(model, dataloader, optimizer,
          device, logger, world_size, rank, comm_delay=None, 
          args=None):
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
        # 如果使用二阶补偿，初始化海森矩阵的估计
        if args.second_order:
            shard_tracker[shard_idx]["hessian_est"] = [torch.zeros_like(p.data) for p in current_params]
        else:
            shard_tracker[shard_idx]["hessian_est"] = None
    del model_params_list
    
    if rank == 0:  logger.info(f"模型共 {len(all_params)} 个参数组，分为 {num_shards} 个分片")

    for epoch in range(args.epochs):
        epoch_step = 0  # 记录每个epoch内的step数
        logger.info(f"开始 Epoch {epoch+1}/{args.epochs}")
        
        for batch in dataloader:
            
            if global_step >= args.total_steps:
                break
            
            # 将 batch 数据移动到 GPU 上
            batch = {k: v.to(device) for k, v in batch.items()}
            start_comp = time.time()
            
            # 检查是否需要发送或接收模型分片
            if global_step % args.sync_interval in receive_time_points:
                print(f"当前步数: {global_step} 准备接收")
                sync_shard_idx = (global_step - args.delay_steps) % args.sync_interval // sync_shard_interval
                comm_time = sync_model_optimized(
                    model, shard_tracker, sync_shard_idx, world_size, logger,
                    comm_delay, num_shards, args.alpha, args)
                
                comm_time_total += comm_time
                
            # 前向计算、损失计算、反向传播与优化
            outputs = model(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["label"])
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if args.second_order:
                # 计算海森矩阵的估计
                update_hessian_est(shard_tracker, model, args.rho)
            
            comp_time = time.time() - start_comp
            comp_time_total += comp_time
            log_comp_time += comp_time
            
            # 每隔 log_interval 步，输出一次日志
            if global_step % args.log_interval == 0:
                logger.info(f"Epoch {epoch+1}/{args.epochs}, Step {epoch_step} (Global: {global_step}) - Loss: {loss.item():.4f} - 计算时间: {log_comp_time:.4f} 秒")
                log_comp_time = 0.0
                
            # 检查是否记录分片。注意第一个interval期间使用最开始的模型参数即可
            if global_step % args.sync_interval in sync_time_points:
                print(f"当前步数: {global_step} 准备记录")
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
    parser.add_argument('--epochs', type=int, default=1,
                        help='训练的轮数')
    parser.add_argument('--sync_interval', type=int, default=50,
                        help='模型参数同步的间隔步数')
    parser.add_argument('--total_steps', type=int, default=100,
                        help='总训练步数')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批处理大小')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='学习率')
    parser.add_argument('--outer_lr', type=float, default=0.4,
                        help='外部学习率')
    parser.add_argument('--use_nesterov', type=int, default=0,
                        help='是否对外层使用 Nesterov 动量 (0 或 1)')
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
    # 延迟补偿
    parser.add_argument('--first_order', type=int, default=0,
                        help='是否使用一阶补偿 (0 或 1)')
    parser.add_argument('--fbeta', type=float, default=0.5,
                        help='一阶补偿的超参数')
    parser.add_argument('--second_order', type=int, default=0,
                        help='是否使用二阶补偿 (0 或 1)')
    parser.add_argument('--sbeta', type=float, default=0.5,
                        help='二阶补偿的超参数')
    parser.add_argument('--rho', type=float, default=0.01,
                        help='二阶补偿的 hessian EMA rho 参数')
    
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

    # 加载数据集和对应的模型与分词器（每个节点各自加载独立副本）
    dataloader, tokenizer, model = load_data_and_model(args.dataset_name, args.model_name, args.batch_size, rank, world_size)
    model.to(device)
    
    # 配置优化器
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    # 计算通信延迟
    comm_delay = calc_comm_delay(model, world_size, logger, simulated_bandwidth_mbps=args.bandwidth)
    
    # 使用传入的参数进行训练
    train(model, dataloader, optimizer,
          device, logger, world_size, rank, comm_delay,
          args)
    
    logger.info("训练结束。")
    
    # 在主进程中记录结束时间
    if rank == 0:
        with open(os.path.join(args.log_dir, "config.txt"), "a") as f:
            f.write(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Training parameters: {args}\n")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
