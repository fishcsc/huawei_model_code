import torch
import torch.nn as nn
import copy
from common import *

def test_shard_model():
    # 创建测试模型，包含不同层类型
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10)
    )
    num_shards = 4
    shards = shard_model(model, num_shards)
    print(f"分片数量: {len(shards)}")
    print(f"每个分片的层数: {[len(shard) for shard in shards]}")
    print(f"分片名称: {[shard.__class__.__name__ for shard in shards]}")
    # 检查分片数量是否正确
    assert len(shards) == num_shards, "分片数量应等于num_shards"
    
    # 检查每个分片的层数
    # 原模型有5层，分成2片：shard0含层0,2,4（3层），shard1含层1,3（2层）
    assert len(shards[0]) == 3, "Shard 0 应有3层"
    assert len(shards[1]) == 2, "Shard 1 应有2层"
    
    # 检查层类型是否正确
    for layer in shards[0]:
        assert isinstance(layer, nn.Linear), "Shard 0 应仅包含Linear层"
    for layer in shards[1]:
        assert isinstance(layer, nn.ReLU), "Shard 1 应仅包含ReLU层"
    
    # 检查是否为深拷贝
    # 修改原模型的层0权重
    original_weight = model[0].weight.clone()
    new_weight = torch.randn_like(original_weight)
    model[0].weight.data = new_weight
    
    # 验证分片中的层未被修改（深拷贝）
    assert torch.equal(shards[0][0].weight, original_weight), "分片中的层应保持原权重，不受原模型修改影响"
    
    # 反向验证：修改分片中的层，检查原模型是否不变
    shard_layer = shards[0][0]
    shard_new_weight = torch.randn_like(shard_layer.weight)
    shard_layer.weight.data = shard_new_weight
    assert not torch.equal(model[0].weight, shard_new_weight), "原模型不应受分片层修改影响"

# 运行测试
test_shard_model()
print("所有测试通过！")