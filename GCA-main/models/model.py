import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import random
import math
# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的，如果这个设置为True，可能会使得运算效率降低
#     torch.backends.cudnn.benchmark = False
class ChannelAttention(nn.Module):#性能提升较大
    def __init__(self, num_channels, reduction_ratio=4):#2和4好像差不多，4稍微好一点
        super(ChannelAttention, self).__init__()
        self.num_channels = num_channels
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(num_channels // reduction_ratio, num_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()# Sigmoid激活函数，输出每个通道的重要性系数

    def forward(self, x):
        # setup_seed(42)
        # 确保输入的形状是[num_edges, num_channels]
        batch_size, num_channels = x.shape
        # 平均池化和最大池化
        avg_pooled = torch.mean(x, dim=0, keepdim=True)  # 维度保持为 [1, num_channels]
        max_pooled = torch.max(x, dim=0, keepdim=True)[0]  # 维度保持为 [1, num_channels]
        # 全连接层处理
        avg_out = self.fc(avg_pooled.squeeze(0))  # 压缩第一个维度然后处理
        max_out = self.fc(max_pooled.squeeze(0))

        # 生成通道注意力权重
        scale = self.sigmoid(avg_out + max_out)  # [num_channels]

        # 应用注意力权重
        return x * scale.unsqueeze(0).expand_as(x)  # 保持形状不变 [num_edges, num_channels]

class PositionalEncoding(nn.Module):
    def __init__(self, feature_dim):
        super(PositionalEncoding, self).__init__()
        self.feature_dim = feature_dim
        pe = torch.zeros(1, feature_dim)
        position = torch.arange(0, feature_dim, dtype=torch.float).unsqueeze(0)
        div_term = torch.exp(torch.arange(0, feature_dim // 2).float() * (-math.log(10000.0) / feature_dim))
        pe[0, 0::2] = torch.sin(position[:, 0::2] * div_term)
        pe[0, 1::2] = torch.cos(position[:, 1::2] * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe

# class NodeFeatureSelfAttention(nn.Module):
#     def __init__(self, feature_dim):
#         super(NodeFeatureSelfAttention, self).__init__()
#         self.query = nn.Linear(feature_dim, feature_dim)
#         self.key = nn.Linear(feature_dim, feature_dim)
#         self.value = nn.Linear(feature_dim, feature_dim)
#         self.position_encoding = PositionalEncoding(feature_dim)
#
#     def forward(self, x):
#         x = self.position_encoding(x)
#         attention_outputs = []
#         for node_features in x:
#             Q = self.query(node_features.unsqueeze(0))
#             K = self.key(node_features.unsqueeze(0))
#             V = self.value(node_features.unsqueeze(0))
#
#             attention_scores = torch.matmul(Q, K.transpose(-2, -1))
#             attention_scores = F.softmax(attention_scores, dim=-1)
#
#             attention_output = torch.matmul(attention_scores, V)
#             attention_outputs.append(attention_output.squeeze(0))
#
#         attention_outputs = torch.stack(attention_outputs, dim=0)
#         return attention_outputs
class NodeFeatureSelfAttention(nn.Module):
    def __init__(self, feature_dim):
        super(NodeFeatureSelfAttention, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.position_encoding = PositionalEncoding(feature_dim)

    def forward(self, x):
        # Adding positional encoding
        x = self.position_encoding(x)  # Shape: [num_nodes, feature_dim]

        # Computing Q, K, V matrices
        Q = self.query(x)  # Shape: [num_nodes, feature_dim]
        K = self.key(x)    # Shape: [num_nodes, feature_dim]
        V = self.value(x)  # Shape: [num_nodes, feature_dim]

        # Transpose K for matmul
        Q = Q.unsqueeze(2)  # Shape: [num_nodes, feature_dim, 1]
        K = K.unsqueeze(1)  # Shape: [num_nodes, 1, feature_dim]

        # Compute attention scores
        attention_scores = torch.matmul(Q, K) / math.sqrt(x.size(-1))  # Shape: [num_nodes, feature_dim, feature_dim]
        attention_scores = F.softmax(attention_scores, dim=-1)  # Shape: [num_nodes, feature_dim, feature_dim]

        # Compute attention output
        attention_output = torch.matmul(attention_scores, V.unsqueeze(2)).squeeze(2)  # Shape: [num_nodes, feature_dim]
        return attention_output
class GCNDecoder(torch.nn.Module):
    def __init__(self,out_feats):# out_feats，这是解码器将处理的输入特征的维度。
        super().__init__()
        # self.reduce_dim1 = nn.Linear(112800, 2048)
        # self.reduce_dim1 = nn.Linear(112800, 1024)#没有encoder的情况
        # self.reduce_dim1 = nn.Linear(2048, 512)#有encoder的情况
        # self.reduce_dim1 = nn.Linear(2048, 512)
        # self.reduce_dim2 = nn.Linear(512, 2 * out_feats)
        self.fc1 = nn.Linear(2 * out_feats,  2 * out_feats)#输入的特征维度是两倍的 out_feats，
        # 相当于输入起始节点和目标节点的聚合后的嵌入向量，预测这两个抗原的距离
        self.bn1 = nn.BatchNorm1d( 2 * out_feats)
        # 通常涉及两个节点（源节点和目标节点）的特征
        self.fc2 = nn.Linear(2 * out_feats, 2 * out_feats)
        self.bn2 = nn.BatchNorm1d(2 * out_feats)
        self.fc3 = nn.Linear(out_feats, out_feats)
        self.bn3 = nn.BatchNorm1d(out_feats)
        self.fc4 = nn.Linear(2 * out_feats, 2 * out_feats)
        self.bn4 = nn.BatchNorm1d(2 * out_feats)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)#随机丢弃10%的特征
        self.final = nn.Linear(2*out_feats, 1)
        #第二个线性层，将特征维度从 out_feats 减少到1，用于最终输出，很可能就是抗原距离。
        self.channel_attention = ChannelAttention(2 * out_feats)
        self.NodeFeatureSelfAttention = NodeFeatureSelfAttention(2 * out_feats)
        self.l1_lambda = 0.001  # L1正则化系数
        self.reduce_dim1 = nn.Linear(2 * out_feats, out_feats)
        self.reduce_bn1 = nn.BatchNorm1d(out_feats)
        self.reduce_dim2 = nn.Linear(out_feats, out_feats // 2)
        self.reduce_bn2 = nn.BatchNorm1d(out_feats // 2)
        self.reduce_dim3 = nn.Linear(out_feats // 2, 1)
        # self.reduce_bn3 = nn.BatchNorm1d(out_feats // 4)
        self.reduce_dim4 = nn.Linear(out_feats // 4, out_feats // 8)
        self.reduce_bn4 = nn.BatchNorm1d(out_feats // 8)
        self.reduce_dim5 = nn.Linear(out_feats // 8, 1)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.1)
    def forward(self, x,edge_index):#这里的x是GCNEncoder得到的嵌入向量，不是初始的图节点特征
        #这里只用到了节点特征和要预测边的起始节点目标节点，和序列预测的不同就是聚合了邻居信息
        # setup_seed(42)
        x_src = x[edge_index[0]]#从 x 中提取所有边的源节点的特征
        x_dst = x[edge_index[1]]#从 x 中提取所有边的目标节点的特征
        edge_x = torch.cat((x_src, x_dst), dim=1)#将源节点和目标节点的特征沿特征维度拼接，为每条边创建一个联合特征向量。
        print("edge_x shape before fc4:", edge_x.shape)
        #edge_x的形状是边数量*2feature维度=边数量*1132
        # 多次降维
        # identity = edge_x  # 保存原始特征作为残差连接的一部分
        # edge_x = self.NodeFeatureSelfAttention(edge_x)
        # 线性变换和ReLU激活
        edge_x = self.fc4(edge_x)
        edge_x = self.bn4(edge_x)
        edge_x = self.relu(edge_x)
        edge_x = self.dropout1(edge_x)
        # edge_x += identity  # 添加原始输入特征
        # edge_x = self.NodeFeatureSelfAttention(edge_x)
        # edge_x = self.channel_attention(edge_x)



        # edge_x = F.relu(self.reduce_dim1(edge_x))
        # # edge_x = F.relu(self.reduce_dim2(edge_x))
        # edge_x = self.reduce_dim2(edge_x)#不用序列后就不需要dim1和dim2降维，因为encoder已经降维

        # edge_x = self.channel_attention(edge_x)
        # 第一层处理
        # First residual block
        out1 = self.fc1(edge_x)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        out1 = self.dropout2(out1)
        out1 =edge_x + out1
        # 第二层处理并引入残差连接
        # Second residual block
        out2 = self.fc2(out1)
        out2 = self.bn2(out2)
        out2 = self.relu(out2)
        # out2 += out1
        out2 = self.dropout3(out2)
        out2 += out1
        # out = out1 + out2  # 加上第一层的输出作为残差
        out = self.final(out2)
        # out = self.final(edge_x)
        # out = self.final(out1)
        # out = self.final(edge_x)
        # edge_x = self.reduce_dim1(edge_x)
        # edge_x = self.reduce_dim1(edge_x)
        # edge_x = self.reduce_bn1(edge_x)
        # edge_x = self.relu(edge_x)
        # edge_x = self.dropout(edge_x)

        # edge_x = self.reduce_dim2(out1)
        # edge_x = self.reduce_bn2(edge_x)
        # edge_x = self.relu(edge_x)
        # edge_x = self.dropout(edge_x)
        #
        # out = self.reduce_dim3(edge_x)
        # edge_x = self.reduce_bn3(edge_x)
        # edge_x = self.relu(edge_x)
        # edge_x = self.dropout(edge_x)

        # out = self.reduce_dim4(edge_x)
        # edge_x = self.reduce_bn4(edge_x)
        # edge_x = self.relu(edge_x)
        # edge_x = self.dropout(edge_x)

        # out = self.reduce_dim5(edge_x)
        # out = self.fc(edge_x)#将联合特征向量通过定义的顺序模型进行处理，得到每条边的最终输出。
        out = torch.flatten(out)#将输出展平。如果输出包括多个边，这确保输出是一个一维数组，每个元素对应一条边的结果。

        return out


