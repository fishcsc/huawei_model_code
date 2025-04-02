import os
os.environ['HF_HOME']='/data/hfhub'
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
os.environ['CUDA_LAUNCH_BLOCKING']='1'
import time
import logging
import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from transformers import BertForSequenceClassification, BertTokenizerFast
from datasets import load_dataset
import argparse
import copy
import threading

from utils.common import *    # 从 utils/util.py 中导入函数
from utils.delay_compensate import *  # 从 utils/delay_compensate.py 中导入补偿函数


# 非阻塞参数同步函数
def async_sync_model(model_snapshot, world_size, logger, comm_delay=None):
    """
    非阻塞方式执行模型参数同步
    """
    # 创建一个线程来执行同步，这样主线程可以继续计算
    def sync_thread():
        comm_start = time.time()
        
        # 使用模型快照进行同步
        with torch.no_grad():
            for snapshot_param in model_snapshot.parameters():
                # 使用模型快照的参数进行all-reduce
                sync_param = snapshot_param.data.clone()
                dist.all_reduce(sync_param, op=dist.ReduceOp.SUM)
                sync_param /= world_size
                
                # 将结果存回快照中
                snapshot_param.data.copy_(sync_param)
        
        comm_time = time.time() - comm_start
        
        # 模拟通信延迟
        if comm_delay:
            # time.sleep(comm_delay)
            logger.info(f"模拟通信时间: {comm_delay:.4f} 秒")
            comm_time = comm_delay
        else:
            logger.info(f"通信时间 (async all-reduce): {comm_time:.4f} 秒")
    
    # 启动同步线程
    sync_thread = threading.Thread(target=sync_thread)
    sync_thread.start()
    return sync_thread

def train(model, dataloader, optimizer, device, logger, world_size, rank, comm_delay=None, args=None):
    """
    训练过程：
    - 模型以 train 模式运行；
    - 每个节点独立进行前向、反向传播及优化更新；
    - 每训练 sync_interval 个step后，异步同步模型参数；
    - 同步从step t开始，在step t+delay_steps接收到参数；
    - 支持vanilla和dc两种延迟补偿方法。
    """
    model.train()
    global_step = 0
    comp_time_total = 0.0
    comm_time_total = 0.0
    
    log_comp_time = 0.0
    
    # 用于延迟补偿的变量
    sync_in_progress = False             # 是否有同步正在进行
    sync_thread = None                   # 同步线程
    last_sync_step = None                # 上次发起同步的步数
    pending_sync_receive = []            # 待接收的同步点
    sync_complete = {}                   # 跟踪同步是否完成
    
    # 延迟补偿用 lambda
    if args.dc_method == 'g':
        lmd = args.dc_lambda
    
    logger.info(f"训练配置: 延迟步数={args.delay_steps}, 延迟补偿方法={args.dc_method}, λ={args.dc_lambda}")

    for epoch in range(args.epochs):
        epoch_step = 0
        logger.info(f"开始 Epoch {epoch+1}/{args.epochs}")
        
        for batch in dataloader:
            if global_step >= args.total_steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            # 检查是否有需要接收的同步结果
            receive_step = global_step - args.delay_steps
            if receive_step in pending_sync_receive:
                logger.info(f"步骤 {global_step}: 接收来自步骤 {receive_step} 的同步结果")
                pending_sync_receive.remove(receive_step)
                
                # 接收完成同步的模型参数
                if sync_complete.get(receive_step, False):
                    # 根据不同方法应用参数更新
                    if args.dc_method == 'vanilla':
                        # Vanilla方法：直接将同步的参数应用到当前模型，本质上覆盖且浪费了10步
                        # 测试通过
                        with torch.no_grad():
                            for param, sync_param in zip(model.parameters(), sync_snapshot.parameters()):
                                param.data.copy_(sync_param.data)
                        logger.info(f"步骤 {global_step}: 应用vanilla同步，来自步骤 {receive_step}")
                    elif args.dc_method == 's_diloco':
                        # 将模型以参数 alpha 进行线性加权平均
                        # 测试通过
                        dc_streaming_diloco(sync_snapshot.parameters(), model.parameters(), args.dc_lambda)
                        logger.info(f"步骤 {global_step}: 应用streaming_diloco同步，来自步骤 {receive_step}")
                    elif args.dc_method == 'braindead':
                        # 脑死方法：假设所有参数都是iid，直接把 w_t - w_t^{global} 作为补偿量无脑加到 w_t_tau 上
                        # 测试通过
                        dc_braindead(original_snapshot.parameters(), model.parameters(), sync_snapshot.parameters())
                        logger.info(f"步骤 {global_step}: 应用dc_braindead同步，来自步骤 {receive_step}")
                    elif args.dc_method == 'g':
                        # 将 w_t 在本地数据上计算一次梯度得到 g_t，应用 g'_t = g_t + λ * g_t ⊙ g_t ⊙ (w_{t+τ} - w_t)
                        temp_grad = get_temp_grad(sync_snapshot, batch)
                        # 使用模型参数 W_{t+\tau} 在 t 时刻的泰勒展开来计算延迟补偿
                        if args.dc_adp_lambda:
                            # 自适应调整 lambda
                            next_lmd_const=dc_expansion_g_adp(
                                sync_snapshot.parameters(),
                                model.parameters(),
                                temp_grad,
                                lmd=lmd,
                                lr=optimizer.param_groups[0]['lr']
                            )
                            lmd = next_lmd_const * args.dc_lambda
                        else:
                            # 固定 lambda
                            dc_expansion_g(
                                sync_snapshot.parameters(),
                                model.parameters(),
                                temp_grad,
                                lmd=lmd,
                                lr=optimizer.param_groups[0]['lr']
                            )
                        logger.info(f"步骤 {global_step}: 应用taylor_g同步，来自步骤 {receive_step}, λ={lmd:.4f}")
                    # 清理不再需要的快照
                    del original_snapshot, sync_snapshot
                    if args.dc_method == 'g' or args.dc_method == 'w':
                        # 清理临时梯度
                        del temp_grad
                        logger.info(f"步骤 {global_step}: 清理临时梯度，进入下一步")
                        global_step += 1
                        continue

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
            global_step += 1
            epoch_step += 1
            
            # 每隔 log_interval 步，输出一次日志
            if global_step % args.log_interval == 0:
                logger.info(f"Epoch {epoch+1}/{args.epochs}, Step {epoch_step} (Global: {global_step}) - Loss: {loss.item():.4f} - 计算时间: {log_comp_time:.4f} 秒")
                log_comp_time = 0.0
            
            # 每隔 sync_interval 步，异步同步一次模型参数
            if global_step % args.sync_interval == 0:
                logger.info(f"步骤 {global_step}: 开始异步同步")
                
                # 保存当前步骤用于延迟补偿
                sync_snapshot = copy.deepcopy(model)
                original_snapshot = copy.deepcopy(model)
                # 启动异步同步
                sync_thread = async_sync_model(sync_snapshot, world_size, logger, comm_delay)
                
                # 记录此次同步，等待后续接收结果
                last_sync_step = global_step
                pending_sync_receive.append(global_step)
                
                # 设置一个回调，在同步完成后标记状态
                def sync_done():
                    sync_complete[global_step] = True
                    logger.info(f"步骤 {global_step} 的同步已完成")
                
                # 这里我们简单模拟一下回调机制
                def simulate_sync_callback():
                    time.sleep(0.1)  # 模拟一些延迟
                    sync_done()
                
                callback_thread = threading.Thread(target=simulate_sync_callback)
                callback_thread.start()
                
                # 模拟通信时间计入总通信时间
                if comm_delay:
                    comm_time_total += comm_delay
                
        # 每个epoch结束时输出一次总结
        if rank == 0:
            logger.info(f"完成 Epoch {epoch+1}/{args.epochs}, 当前步数: {global_step}")
            
        if global_step >= args.total_steps:
            break
    
    # 确保所有同步线程完成
    if sync_thread and sync_thread.is_alive():
        sync_thread.join()
    
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
    parser.add_argument('--timestamp', type=str, default=None, 
                        help='用于日志目录命名的时间戳，格式为 YYYYMMDD-HHMMSS')
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
    parser.add_argument('--bandwidth', type=float, default=None,
                    help='模拟的网络带宽 (Mbps)，不设置则使用实际带宽')
    parser.add_argument('--log_interval', type=int, default=50,
                    help='日志输出的间隔步数')
    # 新增参数
    parser.add_argument('--delay_steps', type=int, default=10,
                        help='延迟步数τ，表示从发送到接收的延迟')
    parser.add_argument('--dc_method', type=str, default='vanilla',
                        help='延迟补偿方法: vanilla或dc')
    parser.add_argument('--dc_lambda', type=float, default=0.01,
                        help='延迟补偿系数λ')
    parser.add_argument('--dc_adp_lambda', type=int, default=0)
    args = parser.parse_args()
    
    # 初始化分布式训练相关环境
    local_rank, rank, world_size = init_distributed()
    device = torch.device("cuda", local_rank)
    
    # 使用传入的时间戳作为 run_name
    run_name = args.timestamp if args.timestamp else None
    logger, run_name = setup_logging(rank, 'dc_diloco', run_name)
    logger.info(f"初始化完成。当前 Rank: {rank}, 总进程数: {world_size}")

    # 记录训练参数
    if rank == 0:
        logger.info(f"训练参数: epochs={args.epochs}, sync_interval={args.sync_interval}, "
                   f"total_steps={args.total_steps}, batch_size={args.batch_size}, "
                   f"learning_rate={args.learning_rate}, bandwidth={args.bandwidth}, "
                   f"delay_steps={args.delay_steps}, dc_method={args.dc_method}, "
                   f"dc_lambda={args.dc_lambda}, dc_adp_lambda={args.dc_adp_lambda} ")

    # 加载 tokenizer 和模型（每个节点各自加载独立副本）
    dataloader, tokenizer, model = load_data_and_model(args.dataset_name, args.model_name, args.batch_size, rank, world_size)
    model.to(device)
    
    # 配置优化器
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # 计算通信延迟
    comm_delay = calc_comm_delay(model, world_size, logger, simulated_bandwidth_mbps=args.bandwidth)
    
    # 使用传入的参数进行训练
    train(model, dataloader, optimizer, device, logger, world_size, rank, comm_delay, args)
    
    logger.info("训练结束。")
    
    # 在主进程中记录结束时间
    if rank == 0:
        log_dir = os.path.join("logs", "dc_diloco", run_name)
        with open(os.path.join(log_dir, "config.txt"), "a") as f:
            f.write(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Training parameters: epochs={args.epochs}, sync_interval={args.sync_interval}, "
                   f"total_steps={args.total_steps}, batch_size={args.batch_size}, "
                   f"dataset_name={args.dataset_name}, model_name={args.model_name}, "
                   f"learning_rate={args.learning_rate}, bandwidth={args.bandwidth}, "
                   f"delay_steps={args.delay_steps}, dc_method={args.dc_method}, "
                   f"dc_lambda={args.dc_lambda}, log_interval={args.log_interval}, "
                   f"dc_adp_lambda={args.dc_adp_lambda}  \n")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()