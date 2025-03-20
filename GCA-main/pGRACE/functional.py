import torch
from torch_geometric.utils import degree, to_undirected

from pGRACE.utils import compute_pr, eigenvector_centrality


# def drop_feature(x, drop_prob):#随机丢弃，无偏重
#     drop_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < drop_prob
#     #uniform_()。这个函数是 torch.Tensor 类的一个方法，它将张量的元素填充为从均匀分布 [min, max) 中抽取的随机数
#     x = x.clone()
#     x[:, drop_mask] = 0#选择了所有样本的将被丢弃的特征，并将它们设为零。
#
#     return x


def drop_feature_weighted(x, w, p: float, threshold: float = 0.7):#可选的阈值参数 threshold
    w = w / w.mean() * p
    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_prob = w.repeat(x.size(0)).view(x.size(0), -1)
    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)

    x = x.clone()
    x[drop_mask] = 0.
    return x

#最终调用的丢弃特征的函数，第二个参数是下面的丢弃种类，3种中心性中的一种
def drop_feature_weighted_2(x, w, p: float, threshold: float = 0.7):#自适应丢弃节点特征，节点和边都是这一个公式
    #w是(w.max() - w) / (w.max() - w.mean())   这个w越大计算得到的w越小，乘以p后概率越小，表示被丢弃的概率更小
    w = w / w.mean() * p#这一行首先计算每一维权重 w 的均值，并将每个权重除以这个均值，然后乘以丢弃概率 p。这样做的目的是使维度权重标准化，使得权重平均值等于 p
    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    #这一行将所有超过阈值 threshold 的权重值设置为 threshold。这是为了避免过高的权重使得某些特征几乎总是被保留，从而可以保持一定的随机性和平衡。
    drop_prob = w
    #得到论文中公式4，即每个节点每一维特征的丢弃概率
    #这一行将权重 w 重复 x.size(0)（即数据的样本数量）次，以生成一个与输入 x 相同形状的丢弃概率矩阵。每个特征对应的丢弃概率在所有样本中是一致的
    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)
    #利用伯努利分布根据 drop_prob 生成一个随机的二值（0或1，即False或True）矩阵，表示是否丢弃对应的特征
    x = x.clone()
    x[:, drop_mask] = 0.#复制输入特征并应用丢弃掩码：

    return x


def feature_drop_weights(x, node_c):#基于节点中心性的特征丢弃权重
    x = x.to(torch.bool).to(torch.float32)#node_c是3种节点中心性中的一种，还有两种在utils里面
    w = x.t() @ node_c
    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())

    return s#赋值给了drop_feature_weighted_2中的w


def feature_drop_weights_dense(x, node_c):
    x = x.abs()
    w = x.t() @ node_c
    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())

    return s#与上面一样，不过是用于连续特征











#最终调用的丢弃边的函数，第二个参数是下面的丢弃种类，3种中心性中的一种
def drop_edge_weighted(edge_index, edge_weights, p: float, threshold: float = 1.):#返回丢弃边后的索引
    edge_weights = edge_weights / edge_weights.mean() * p #edge_weights是下面3种边中心性中的一种
    #这一行代码首先计算边权重 edge_weights 的均值，然后将每个边的权重除以这个均值，并乘以丢弃概率 p。这一步骤是为了调整权重，使其平均值为 p，标准化权重以适应不同的丢弃策略。
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    #这一行将所有超过阈值 threshold 的权重设置为 threshold。这是为了防止某些边的权重过高，从而几乎永远不会被丢弃，保持一定的随机性和平衡
    sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)
    #使用伯努利分布根据调整后的 edge_weights 生成选择掩码 sel_mask。这里先通过 1. - edge_weights 计算保留边的概率，然后根据这个概率生成一个随机的二值（True/False）矩阵，表示是否保留对应的边
    return edge_index[:, sel_mask]
    #使用 sel_mask 从原始的边索引 edge_index 中过滤出应该保留的边。edge_index[:, sel_mask] 使用掩码来选择应保留的列（即边

def degree_drop_weights(edge_index):#边的丢弃，3种中心性中的一种
    edge_index_ = to_undirected(edge_index)
    #这行代码将有向图的边索引 edge_index 转换为无向图的边索引 edge_index_。这通常是通过添加反向边来完成的，确保图中的每条边都是无向的
    deg = degree(edge_index_[1])
    #使用 degree 函数计算无向图中每个节点的度（即每个节点的连接数）。这里 edge_index_[1] 指的是边索引数组的第二行，包含所有边的目标节点索引
    deg_col = deg[edge_index[1]].to(torch.float32)
    #从 deg 数组中索引原始边索引 edge_index[1]（目标节点索引）的度，并将结果转换为浮点数。这样做是为了后续的计算需要，特别是对数和除法操作
    s_col = torch.log(deg_col)
    #对 deg_col 中每个元素（即每条边目标节点的度）取自然对数。对数转换有助于缩小不同度值之间的差距，使得权重分布更加均匀。
    weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())

    return weights


def pr_drop_weights(edge_index, aggr: str = 'sink', k: int = 10):#边的丢弃，3种中心性中的一种
    pv = compute_pr(edge_index, k=k)
    #函数计算整个图的PageRank值。k 参数通常用于指定迭代次数或者某些算法中的参数，用于控制PageRank计算的精度或者收敛速度
    pv_row = pv[edge_index[0]].to(torch.float32)
    pv_col = pv[edge_index[1]].to(torch.float32)
    #这两行从PageRank结果 pv 中提取出每条边的起点（source）和终点（sink）的PageRank值。edge_index[0] 和 edge_index[1] 分别表示边的起点和终点索引。结果转换为浮点数类型，以便进行接下来的计算。
    s_row = torch.log(pv_row)
    s_col = torch.log(pv_col)
    #对起点和终点的PageRank值取自然对数。这一步有助于缩小数值范围并平滑分布，使权重计算更为稳定
    if aggr == 'sink':
        s = s_col
    elif aggr == 'source':
        s = s_row
    elif aggr == 'mean':
        s = (s_col + s_row) * 0.5
    else:
        s = s_col
    #根据 aggr 参数选择聚合方式。sink 使用终点（sink）的PageRank对数值，source 使用起点（source）的PageRank对数值，
    # mean 使用起点和终点对数值的平均。如果没有指定或不在选项中，默认使用终点的值
    weights = (s.max() - s) / (s.max() - s.mean())
    #计算最终的权重。首先使用 s.max() - s 归一化PageRank对数值，使得高PageRank对数值的边获得较低的权重值（因为高PageRank表示节点重要性高，不希望丢弃）。
    # 接着除以 (s.max() - s.mean()) 进一步归一化，确保权重值分布合理
    return weights


def evc_drop_weights(data):#边的丢弃，3种中心性中的一种
    evc = eigenvector_centrality(data)
    #使用 eigenvector_centrality 函数计算图中每个节点的特征向量中心性。这通常涉及到找到图邻接矩阵的最大特征值对应的特征向量，并将其作为中心性的度量。
    evc = evc.where(evc > 0, torch.zeros_like(evc))
    #将特征向量中心性中非正的值（小于或等于0的值）替换为0。这一步是必要的，因为中心性值需要是非负的，且对数函数的输入必须是正数
    evc = evc + 1e-8
    #为每个节点的中心性值加上一个很小的正数（1e-8），这是为了避免在接下来的对数运算中对0取对数导致数值错误。
    s = evc.log()
#对处理后的中心性值取自然对数。对数变换有助于处理中心性值的潜在极端差异，使其更加适合后续的计算
    edge_index = data.edge_index
    s_row, s_col = s[edge_index[0]], s[edge_index[1]]
    #从 data.edge_index 中提取边的起点和终点索引，然后使用这些索引从对数化的中心性值 s 中提取对应的起点和终点的中心性对数值。
    s = s_col
    #在此场景中，选择使用边的终点的中心性对数值 s_col 作为权重计算的基础。这一选择基于特定的应用场景或偏好。
    return (s.max() - s) / (s.max() - s.mean())