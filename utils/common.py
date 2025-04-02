import os
import time
import logging
import random
import copy
import torch
import torch.distributed as dist


def approx_hessian(grads):
    """
    用梯度的平方近似Hessian矩阵对角线元素。
    接收梯度列表并返回每个参数的Hessian近似值列表。
    """
    return [(g.clone().detach() ** 2) for g in grads]  # 对每个梯度张量计算平方

def get_temp_grad(w_t, batch):
    """
    计算模型参数 w_t 在 batch 上的梯度。
    注意，仅计算一次梯度，不反向传播，不影响现有计算图。
    """
    # 移除torch.no_grad()上下文，因为我们需要计算梯度
    
    # 根据模型类型处理输入格式
    if isinstance(batch, dict) and "label" in batch:
        # BERT 等分类模型
        outputs = w_t(input_ids=batch["input_ids"], 
                      attention_mask=batch["attention_mask"], 
                      labels=batch["label"])
        loss = outputs.loss
    else:
        # tinyllama 自回归
        outputs = w_t(input_ids=batch["input_ids"], 
                      attention_mask=batch["attention_mask"])
        loss = outputs.loss
    
    # 计算梯度，设置create_graph=False以不创建新的计算图
    # 这样计算的梯度不会参与后续的反向传播
    grads = torch.autograd.grad(loss, w_t.parameters(), create_graph=False)
    
    # 返回分离的梯度，确保不会影响主计算图
    return [g.detach().clone() for g in grads]

def init_distributed():
    """
    初始化分布式环境，使用 NCCL 后端 适用于单机多 GPU 。
    环境变量 LOCAL_RANK 必须通过启动命令传入。
    返回 local_rank, 当前进程 rank, 总进程数 world_size。
    """
    random.seed(2025)   # 设置随机种子
    torch.manual_seed(2025)
    torch.cuda.manual_seed_all(2025)
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return local_rank, rank, world_size

def setup_logging(rank, log_dir):
    """
    配置日志：每个进程记录到独立文件，同时输出到控制台。
    日志存储到 log_dir 目录下
    """
    # 创建目录结构
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger("NodeLogger")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(f"%(asctime)s - Rank {rank} - %(levelname)s - %(message)s")

    # 确保处理程序不会重复添加
    if logger.handlers:
        logger.handlers = []

    # 文件日志
    log_path = os.path.join(log_dir, f"node_{rank}.log")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # 控制台日志
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # 记录运行配置
    if rank == 0:
        config_path = os.path.join(log_dir, "config.txt")
        with open(config_path, "w") as f:
            f.write(f"Run dir: {log_dir}\n")
            f.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not specified')}\n")
    
    return logger
        
def calc_comm_delay(model, world_size, logger, simulated_bandwidth_mbps=None):
    """
    计算模型参数同步的通信时间。
    """
    if simulated_bandwidth_mbps is None:
        return None
    # 计算参数总量（字节）
    total_params_bytes = sum(p.numel() * 4 for p in model.parameters())  # 4 bytes per float32
    
    # 计算理论上的通信量（字节）: 2*(N-1)/N * 参数总量
    comm_volume_bytes = 2 * (world_size - 1) / world_size * total_params_bytes
    
    # 转换为MB
    comm_volume_mb = comm_volume_bytes / (1024 * 1024)
    
    # 模拟的带宽延迟（秒）= 数据量(MB) / 带宽(MB/s)
    simulated_delay = comm_volume_mb / (simulated_bandwidth_mbps / 8)  # 除以8将Mbps转换为MB/s
    
    logger.info(f"模型大小: {total_params_bytes/1024/1024:.2f}MB, Ring All Reduce通信量: {comm_volume_mb:.2f}MB")
    logger.info(f"模拟 {simulated_bandwidth_mbps}Mbps 带宽，延迟 {simulated_delay:.2f}秒")
    return simulated_delay

def shard_model(model, num_shards):
    """
    将模型按层粒度均匀划分，返回每个shard的深拷贝
    """
    shards = []
    for i in range(num_shards):
        shard = torch.nn.ModuleList()
        for j, layer in enumerate(model.children()):
            if j % num_shards == i:
                with torch.no_grad():
                    shard.append(copy.deepcopy(layer))
        shards.append(shard)
    return shards
