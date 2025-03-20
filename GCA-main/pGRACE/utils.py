import math
import numpy as np
import scipy.sparse as sp
from scipy.special import iv
from scipy.sparse.linalg import eigsh
import os.path as osp
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import SpectralEmbedding
# from libKMCUDA import kmeans_cuda
from tqdm import tqdm
from matplotlib import cm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.optim import Adam
from torch.utils.data import random_split
from torch_geometric.nn import GCNConv, SGConv, SAGEConv, GATConv, GraphConv, GINConv
from torch_geometric.utils import sort_edge_index, degree, add_remaining_self_loops, remove_self_loops, get_laplacian, \
    to_undirected, to_dense_adj, to_networkx
from torch_geometric.datasets import KarateClub
from torch_scatter import scatter
import torch_sparse

# import networkx as nx
import matplotlib.pyplot as plt


def get_base_model(name: str):
    def gat_wrapper(in_channels, out_channels):
        return GATConv(
            in_channels=in_channels,
            out_channels=out_channels // 4,
            heads=4
        )

    def gin_wrapper(in_channels, out_channels):
        mlp = nn.Sequential(
            nn.Linear(in_channels, 2 * out_channels),
            nn.ELU(),
            nn.Linear(2 * out_channels, out_channels)
        )
        return GINConv(mlp)

    base_models = {
        'GCNConv': GCNConv,
        'SGConv': SGConv,
        'SAGEConv': SAGEConv,
        'GATConv': gat_wrapper,
        'GraphConv': GraphConv,
        'GINConv': gin_wrapper
    }

    return base_models[name]


def get_activation(name: str):
    activations = {
        'relu': F.relu,
        'hardtanh': F.hardtanh,
        'elu': F.elu,
        'leakyrelu': F.leaky_relu,
        'prelu': torch.nn.PReLU(),
        'rrelu': F.rrelu
    }

    return activations[name]


def compute_pr(edge_index, damp: float = 0.85, k: int = 10):#节点特征3种中心性中的一种，用于计算图中各节点的PageRank中心性的
    num_nodes = edge_index.max().item() + 1
    #计算图中节点的总数。这是通过获取边索引 edge_index 中的最大值并加1来实现的，确保节点编号从0开始连续
    deg_out = degree(edge_index[0])
    #使用 degree 函数计算图中每个节点的出度。这里 edge_index[0] 表示所有边的起始节点，通过计算每个节点作为起始节点出现的次数来得到出度。
    x = torch.ones((num_nodes, )).to(edge_index.device).to(torch.float32)
    #初始化所有节点的PageRank分数为1，并确保这个初始分数向量 x 位于与边索引相同的设备上，数据类型为浮点数
    for i in range(k):#迭代计算PageRank分数
        edge_msg = x[edge_index[0]] / deg_out[edge_index[0]]
        #对每条边，将起点的PageRank分数除以其出度，计算出每条边传递的消息。这表示每个节点将其PageRank分数平均分配给它的所有邻居
        agg_msg = scatter(edge_msg, edge_index[1], reduce='sum')
        #使用 scatter 函数将上述消息按照边的终点索引进行聚合。即，每个节点收到其所有入边的消息总和。
        x = (1 - damp) * x + damp * agg_msg
        #根据经典的PageRank更新公式，更新每个节点的PageRank分数。这里 damp 是阻尼因子，通常设置为0.85，表示继承原始分数的比例，而新的分数则由邻居传入的聚合消息决定。
    return x
#函数返回经过 k 轮迭代后的所有节点的PageRank分数。这些分数反映了节点在图中的重要性，其中更高的分数表示更高的重要性或影响力。
#这种实现方式有效地计算了图中每个节点的PageRank值，可用于评估节点的中心性或在图结构数据中进行其他相关分析。

def eigenvector_centrality(data, max_iter=1000, tol=1e-6):##节点特征3种中心性中的一种，节点特征度中心性不用定义函数
    #计算图中每个节点的特征向量中心性，利用了 NetworkX 库的功能
    graph = to_networkx(data)
    #这行代码将输入的 data（通常是一个包含边信息和可能的节点特征的数据结构，如PyTorch Geometric的 Data 对象）转换成一个 NetworkX 图对象。to_networkx 函数通常会处理节点和边的信息，确保图的构建正确反映了原始数据的连接结构。
    # x = nx.eigenvector_centrality_numpy(graph)
    try:
        # 计算图中每个节点的特征向量中心性
        x = nx.eigenvector_centrality_numpy(graph, max_iter=max_iter, tol=tol)
    except nx.NetworkXError as e:
        print(f"NetworkX error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
    #利用 NetworkX 的 eigenvector_centrality_numpy 函数计算图中每个节点的特征向量中心性。这个函数使用 NumPy 实现来高效地计算中心性，基于图的邻接矩阵的最大特征值对应的特征向量。
    x = [x[i] for i in range(data.num_nodes)]
    #这行代码遍历所有节点（从 0 到 data.num_nodes - 1），并从 x 字典中提取出每个节点的特征向量中心性分数。这一步确保最终的输出与输入数据中的节点顺序一致，并将字典格式转换为列表格式，方便后续处理
    return torch.tensor(x, dtype=torch.float32).to(data.edge_index.device)
#将包含特征向量中心性分数的列表 x 转换为 PyTorch 张量，并设置数据类型为 torch.float32
#这种中心性测量考虑了节点的连接模式以及连接到的邻居的重要性，使得连接到高中心性节点的节点也具有较高的中心性

def generate_split(num_samples: int, train_ratio: float, val_ratio: float):
    train_len = int(num_samples * train_ratio)
    val_len = int(num_samples * val_ratio)
    test_len = num_samples - train_len - val_len

    train_set, test_set, val_set = random_split(torch.arange(0, num_samples), (train_len, test_len, val_len))

    idx_train, idx_test, idx_val = train_set.indices, test_set.indices, val_set.indices
    train_mask = torch.zeros((num_samples,)).to(torch.bool)
    test_mask = torch.zeros((num_samples,)).to(torch.bool)
    val_mask = torch.zeros((num_samples,)).to(torch.bool)

    train_mask[idx_train] = True
    test_mask[idx_test] = True
    val_mask[idx_val] = True

    return train_mask, test_mask, val_mask

