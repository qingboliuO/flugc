import torch
import sys
# print("Python version:", sys.version)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(torch.cuda.is_available())
# print("CUDA version:", torch.version.cuda)
# print("PyTorch version:", torch.__version__)
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveFeatureDropout(nn.Module):
    def __init__(self, num_features):
        super(AdaptiveFeatureDropout, self).__init__()
        num_groups = num_features // 100  # 假设num_features能够被100整除
        # 初始化每组特征的权重，初始权重平均分配，使得它们的和为1
        initial_weights = torch.full((num_groups,), 1 / num_groups)
        self.feature_weights = nn.Parameter(initial_weights)

    def forward(self, x, base_drop_rate):
        # 使用softmax归一化权重以确保权重之和为1
        normalized_weights = F.softmax(self.feature_weights, dim=0)
        # 计算每组特征的丢弃概率
        drop_probs = base_drop_rate * normalized_weights * (x.size(1) // 100)

        # 为每组特征生成丢弃mask
        group_drop_mask = torch.bernoulli(1 - drop_probs).to(x.device)

        # 将组mask扩展到每个特征
        drop_mask = group_drop_mask.repeat_interleave(100)

        # 应用mask
        x = x * drop_mask
        # 输出被遮蔽的特征数量
        num_dropped_features = (drop_mask == 0).sum().item()
        print(f"Number of dropped features: {num_dropped_features}")
        return x, drop_mask, normalized_weights

def drop_feature(x, drop_prob):
    # 创建dropout模块实例
    dropout = AdaptiveFeatureDropout(x.size(1))
    dropout.train()  # 确保模块处于训练模式以应用dropout

    # 将数据和模块放到相同的设备上
    device = x.device#将x即输入数据放到了gpu上
    dropout.to(device)#将dropout参数放到了gpu上

    # 应用dropout并返回结果
    dropped_x, drop_mask, weights = dropout(x, drop_prob)
    print("Current normalized weights:", weights)
    # 计算被遮蔽的100维特征向量的范围
    dropped_groups = (
                drop_mask.view(-1, 100).sum(dim=1) == 0).nonzero().squeeze()

    # 确保 dropped_groups 是一个列表
    if dropped_groups.ndim == 0:
        dropped_groups = [dropped_groups.item()]  # 如果是单个数字，转换为列表
    else:
        dropped_groups = dropped_groups.tolist()

    dropped_ranges = [f"{i * 100}-{(i + 1) * 100 - 1}" for i in dropped_groups]

    print(f"Dropped feature ranges: {dropped_ranges}")
    return dropped_x

# 示例使用
x = torch.randn(5, 800)  # 假设有5个样本，每个样本有10个特征
print(x)
x_1 = drop_feature(x, 0.1)  # 以0.2的基础丢弃率调用
x_2 = drop_feature(x, 0.5)  # 以0.3的基础丢弃率调用
print(x_1)
print(x_2)
