import argparse
import os.path as osp
# import seaborn as sns
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import random
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
from sklearn.manifold import TSNE
from time import perf_counter as t
from graphData import graphDataset
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GATConv
import os
from model import Encoder, GATModel, drop_feature
from models.model import GCNDecoder
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from torch_geometric.transforms import Compose, NormalizeFeatures, ToDevice, RandomLinkSplit
from pGRACE.functional import drop_edge_weighted, \
    degree_drop_weights, \
    evc_drop_weights, pr_drop_weights, \
    feature_drop_weights, drop_feature_weighted_2, feature_drop_weights_dense
from pGRACE.utils import compute_pr, eigenvector_centrality
epoch_counter = 0
def setup_seed(seed):
    torch.manual_seed(seed)#用于设置CPU生成随机数的种子
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的，如果这个设置为True，可能会使得运算效率降低
    torch.backends.cudnn.benchmark = False  # 用来启用cudnn自动寻找最合适当前配置的高效算法，以优化运行效率
# class GraphModel(nn.Module):
#     def __init__(self, in_channels, num_hidden, activation, base_model, k=2, skip=True):
#         super(GraphModel, self).__init__()
#         self.encoder = Encoder(in_channels=in_channels, out_channels=num_hidden, activation=activation, base_model=base_model, k=k, skip=skip)
#         self.decoder = GCNDecoder(out_feats=num_hidden)
#
#     def forward(self, x, edge_index):
#         # 使用编码器处理输入
#         encoded_x = self.encoder(x, edge_index)
#         # 使用解码器处理编码后的输出
#         output = self.decoder(encoded_x, edge_index)
#         return output
def encoder_train(model: GATModel, x, edge_index):#edge_index通常是一个 [2, num_edges] 大小的矩阵
    # global epoch_counter
    # drop_weights = evc_drop_weights(data).to(device)
    model.train()#训练模式 encoder_model = GATModel=model
    encoder_optimizer.zero_grad()
    # edge_index_1 = drop_edge_weighted(data.edge_index, drop_weights, p=0.6)  # 这里的1，2指向某个参数文件的不同丢弃概率，上面以经绝对了用哪种丢弃方法
    # edge_index_2 = drop_edge_weighted(data.edge_index, drop_weights, p=0.55)#大致准确但是数量有点误差
    #丢弃边，如果是无向图应该一下丢弃两条边
    # if epoch_counter < 2:
    #     print(f'Epoch {epoch_counter + 1}')
    #     print('edge_index_1:', edge_index_1)
    #     print('edge_index_2:', edge_index_2)

    # epoch_counter += 1
    edge_index_1 = dropout_adj(edge_index, p=0.5)[0]#原来0.5
    edge_index_2 = dropout_adj(edge_index, p=0.5)[0]#原来0.6
    # 通过 dropout_adj 函数从原始的 edge_index 中随机丢弃一部分边。dropout_adj 函数通常返回一个元组，其中第一个元素是处理后的边索引。
    # p=drop_edge_rate_1 和 p=drop_edge_rate_2 分别定义了两种不同的丢边概率，概率是参数在yaml文件中存储
    #丢弃节点特征
    # x_1 = drop_feature_weighted_2(data.x, feature_weights,
    #                               0.15)
    # x_2 = drop_feature_weighted_2(data.x, feature_weights,
    #                               0.1)
    # x_1 = model.drop_feature(x, 0.1)#定义了两种不同的丢特征概率，用于两个视图
    # x_2 = model.drop_feature(x, 0.15)#566序列0.2和0.3好
    x_1 = drop_feature(x, 0.1)  # 定义了两种不同的丢特征概率，用于两个视图
    x_2 = drop_feature(x, 0.15)  # 566序列0.2和0.3好#jiah1n1 0.2 0.3效果好
    #分别生成两个视图聚合后的嵌入向量
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)#模型的参数就是encoder的参数即聚合时的权重
    #这里的z1z2是GCNModel的输出，没有经过projection的全连接层
    Contrastive_loss = model.loss(z1, z2, batch_size=0)#调用模型的损失函数计算两个视图的嵌入向量 z1 和 z2 之间的损失
    #这里的z1, z2经过了projection的全连接层，计算损失前要对输入数据经过一些形状变幻
    #batch_size=0表示一次性加载所有节点
    Contrastive_loss.backward()
    # for name, param in model.named_parameters():
    #     if param.grad is not None:
    #         print(f'Gradient for {name}: {param.grad.norm()}')
    encoder_optimizer.step()#优化encoder生成嵌入向量过程中的权重
    encoder_scheduler.step()
    return Contrastive_loss.item()
if __name__ == '__main__':
    seed = 42  # 可以选择任意一个喜欢的数字作为种子 42效果不错
    torch.manual_seed(seed)  # 用于设置CPU生成随机数的种子
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的，如果这个设置为True，可能会使得运算效率降低
    torch.backends.cudnn.benchmark = False
    # os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 这里设置为"0"表示使用机器上的第一个GPU。
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # gData = graphDataset("nature586B")#566H1N1
    # gData = graphDataset("nature566H3N2不对称")
    # gData = graphDataset("nature566H3N2")
    # gData = graphDataset("nature566B")
    gData = graphDataset("nature566H1N1")
    # gData = graphDataset("nature566H1N1不对称")
    # gData = graphDataset("nature585BYamagata")
    # gData = graphDataset("nature585BVictoria")
    # gData = graphDataset("nature585BVictoria不对称")
    # gData = graphDataset("nature585BYamagata不对称")
    # gData = graphDataset("jiaH3N2")
    # gData = graphDataset("nature566H1N1不对称")
    # gData = graphDataset("nature566H3N2不对称")
    print("Node features (x) shape:", gData.data.x.shape)
    print("Edge index shape:", gData.data.edge_index.shape)
    # data = gData.to(device)  # 将数据对象 data 移动到之前设置的 device
    data = gData.data  # 直接访问加载好的数据对象
    data.to(device)
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.edge_attr = data.edge_attr.to(device)
    # print(data.virus_names)
    num_features = data.num_features#56400
    num_hidden = 256#编码器输出维度，128比256和512效果好
    num_proj_hidden = 64
    encoder_learning_rate = 0.0008
    decoder_learning_rate = 0.0005#解码器学习率#原理0.0005
    weight_decay_encoder = 0.0005  #L2 正则化
    weight_decay_decoder = 0.001       #5e-4
    base_model = GATConv
    num_layers = 2#原来2
    tau = 0.3

    global drop_weights
    # drop_weights = evc_drop_weights(data).to(device)
    # drop_weights = pr_drop_weights(data.edge_index, aggr='sink', k=200).to(
    #     device)

    # drop_weights = degree_drop_weights(data.edge_index).to(device)

    # node_pr = compute_pr(data.edge_index)
    # feature_weights = feature_drop_weights_dense(data.x, node_c=node_pr).to(
    #     device)

    # node_evc = eigenvector_centrality(data)  # 计算特征向量中心性值
    # feature_weights = feature_drop_weights_dense(data.x, node_c=node_evc).to(
    #     device)
    encoder = Encoder(num_features, num_hidden, F.relu, base_model=base_model,
                      k=num_layers).to(device)  # 2次聚合，同时随机初始化了参数
    encoder_model = GATModel(encoder, num_hidden, num_proj_hidden,num_features,tau).to(
        device)
    # for name, param in encoder_model.named_parameters():
    #     print(name, param.size())
    encoder_optimizer = torch.optim.Adam(encoder_model.parameters(), lr=encoder_learning_rate, weight_decay=weight_decay_encoder)
    encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer,
                                                        step_size=200,
                                                        gamma=0.9)
    start = t()
    prev = start
    # 记录训练开始的时间和前一个epoch的结束时间，t() 是一个时间记录函数，比如time.time()
    for epoch in range(1, 1000):  # 进行从1到num_epochs的循环，10月20号之前是2000
        # 在每个epoch中，调用train函数执行一次训练，该函数返回计算得到的损失。
        # setup_seed(seed)
        encoder_loss = encoder_train(encoder_model, data.x,
                                     data.edge_index)  # 训练的时候好像没有用标签
        # 记录当前时间
        now = t()
        print(
            f'(T) | Epoch={epoch:03d}, encoderContrastive_loss={encoder_loss:.4f}, '  # 显示当前轮次和损失
            f'this epoch {now - prev:.4f}, total {now - start:.4f}')
        # 显示本轮耗时和总耗时。
        prev = now
        # 更新前一个epoch结束的时间为当前时间

        # encoder_scheduler.step()
        # print("Feature weights after epoch {}: {}".format(epoch,
        #                                                   encoder_model.dropout.feature_weights))
        # print("Feature weights gradients: ",
        #       encoder_model.dropout.feature_weights.grad)
    print("=== Final ===")  # 经过很多个encoder优化出更好的gcnencoder编码器生成嵌入向量
    encoder_model.eval()
    # torch.save(encoder_model.state_dict(), 'encoderjia_model.pth')
    # pretrain_data = graphDataset("H3N2").data.to(device)  # 加载预训练数据集到gpu
    # encoder_model.load_state_dict(torch.load('encoderjiaH1N1best_model.pth'))
    # encoder_model.load_state_dict(torch.load('encoderjiaH3N2_model.pth'))   #这里加载的模型参数一样，后面结果就会一样
    with torch.no_grad():
        z1 = encoder_model(data.x, data.edge_index)  # 预训练数据集新生成的嵌入向量
        # z2 = encoder_model(pretrain_data.x,
        #                    pretrain_data.edge_index)  # 目标数据集新生成的嵌入向量
    print("Shape of z:", z1.shape)  # torch.Size([50, 128])
    data.x = z1
    z1_numpy = z1.cpu().detach().numpy()
    # 假设 z1 和 data.virus_names 已定义
    # z1_numpy = z1.cpu().detach().numpy()
    #
    # # 优化t-SNE参数适应更多聚类
    # tsne = TSNE(n_components=2, perplexity=15, method='barnes_hut',
    #             init='pca', n_iter=2000, early_exaggeration=11)
    # z1_2d = tsne.fit_transform(z1_numpy)
    #
    # # 设置15个聚类
    # # kmeans = KMeans(n_clusters=15, random_state=42)
    # kmeans = KMeans(n_clusters=8, random_state=42)
    # clusters = kmeans.fit_predict(z1_2d)
    #
    # # 使用扩展的15色方案（高对比度纯色）
    # colors = [
    #     '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    #     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    #     '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    # ]
    #
    # # 创建对应深色版本（通过HSV调整）
    # dark_colors = [
    #     '#184e7d', '#d45f07', '#1f7a1f', '#a01d1d', '#6b4896',
    #     '#633a37', '#b25998', '#5a5a5a', '#8c8f1a', '#0f8c9c',
    #     '#8da5c4', '#d4995a', '#6bb06d', '#d47a7a', '#9580b0'
    # ]
    #
    # plt.figure(figsize=(10, 10), dpi=300)
    # plt.grid(False)
    #
    # # 完全移除坐标轴
    # # plt.axis('off')
    # plt.xticks([])  # 去除x轴刻度
    # plt.yticks([])  # 去除y轴刻度
    # for i in range(5):
    #     idxs = np.where(clusters == i)[0]
    #     cluster_points = z1_2d[idxs]
    #     cluster_center = np.mean(cluster_points, axis=0)
    #
    #     # 绘制实心数据点
    #     plt.scatter(z1_2d[idxs, 0], z1_2d[idxs, 1],
    #                 c=colors[i], s=80,
    #                 edgecolor='black', linewidth=0.8,
    #                 alpha=1, zorder=3)
    #
    #     # 绘制深色簇中心（比原色深两个色阶）
    #     plt.scatter(cluster_center[0], cluster_center[1],
    #                 c=dark_colors[i], s=330,
    #                 edgecolor='black', linewidth=0.8,
    #                 zorder=4, marker='o')
    #     plt.scatter(cluster_center[0], cluster_center[1],
    #                 c='none',  # Use transparent fill
    #                 s=120,  # Slightly smaller size
    #                 edgecolor='#FF0000', linewidth=0.8,
    #                 zorder=5,  # Higher zorder to appear on top
    #                 marker='^')  # Pentagon or any other marker
    #
    #
    #
    #     # 添加标签（仅在中心点显示）
    #     closest_idx = np.argmin(
    #         np.sum((cluster_points - cluster_center) ** 2, axis=1))
    #     plt.annotate(data.virus_names[idxs[closest_idx]],
    #                  (cluster_center[0], cluster_center[1]),
    #                  ha='center', va='center',
    #                  fontsize=13, weight='normal',
    #                  color='black', zorder=6)



    # z1_numpy = z1.cpu().detach().numpy()
    # tsne = TSNE(n_components=2, perplexity=15, method='barnes_hut',
    #             init='pca', n_iter=2000, early_exaggeration=11)
    # z1_2d = tsne.fit_transform(z1_numpy)
    #
    # # Set up 8 clusters
    # kmeans = KMeans(n_clusters=8, random_state=42)
    # clusters = kmeans.fit_predict(z1_2d)
    #
    # # Use the same color schemes as before
    # colors = [
    #     '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    #     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    #     '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    # ]
    #
    # dark_colors = [
    #     '#184e7d', '#d45f07', '#1f7a1f', '#a01d1d', '#6b4896',
    #     '#633a37', '#b25998', '#5a5a5a', '#8c8f1a', '#0f8c9c',
    #     '#8da5c4', '#d4995a', '#6bb06d', '#d47a7a', '#9580b0'
    # ]
    #
    # # Create a list to store data for Excel
    # excel_data = []
    # center_viruses = []
    #
    # plt.figure(figsize=(10, 10), dpi=300)
    # plt.grid(False)
    # plt.xticks([])
    # plt.yticks([])
    #
    # for i in range(8):
    #     idxs = np.where(clusters == i)[0]
    #     cluster_points = z1_2d[idxs]
    #
    #     if len(idxs) > 0:
    #         # Calculate cluster center
    #         cluster_center = np.mean(cluster_points, axis=0)
    #
    #         # Find the virus closest to the cluster center
    #         distances = np.sum((cluster_points - cluster_center) ** 2, axis=1)
    #         closest_idx = idxs[np.argmin(distances)]
    #         center_virus_name = data.virus_names[closest_idx]
    #         center_viruses.append((closest_idx, center_virus_name))
    #
    #         # Add data to Excel list
    #         for idx in idxs:
    #             excel_data.append({
    #                 'virus_name': data.virus_names[idx],
    #                 'cluster': i,
    #                 'color': colors[i],
    #                 'is_center': idx == closest_idx
    #             })
    #
    #         # Plot points
    #         plt.scatter(z1_2d[idxs, 0], z1_2d[idxs, 1],
    #                     c=colors[i], s=80,
    #                     edgecolor='black', linewidth=0.8,
    #                     alpha=1, zorder=3)
    #
    #         # Plot cluster center
    #         plt.scatter(cluster_center[0], cluster_center[1],
    #                     c=dark_colors[i], s=330,
    #                     edgecolor='black', linewidth=0.8,
    #                     zorder=4, marker='o')
    #         plt.scatter(cluster_center[0], cluster_center[1],
    #                     c='none',
    #                     s=120,
    #                     edgecolor='#FF0000', linewidth=0.8,
    #                     zorder=5,
    #                     marker='^')
    #
    #         # Only annotate the center virus
    #         plt.annotate(center_virus_name,
    #                      (cluster_center[0], cluster_center[1]),
    #                      ha='center', va='center',
    #                      fontsize=10,
    #                      # Slightly larger font for better visibility
    #                      weight='bold',
    #                      color='black', zorder=6)
    #
    # plt.title('H3N2', fontsize=18, fontweight='bold', pad=20)
    # plt.show()
    #
    # # Create and save Excel file
    # df = pd.DataFrame(excel_data)
    # df.to_excel('virus_clusters.xlsx', index=False)

    # z1_numpy = z1.cpu().detach().numpy()
    # tsne = TSNE(n_components=2, perplexity=15, method='barnes_hut',
    #             init='pca', n_iter=2000, early_exaggeration=11)
    # z1_2d = tsne.fit_transform(z1_numpy)
    #
    # # Set up 8 clusters
    # kmeans = KMeans(n_clusters=8, random_state=42)
    # clusters = kmeans.fit_predict(z1_2d)
    #
    # # Use the same color schemes as before
    # colors = [
    #     '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    #     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    #     '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    # ]
    #
    # dark_colors = [
    #     '#184e7d', '#d45f07', '#1f7a1f', '#a01d1d', '#6b4896',
    #     '#633a37', '#b25998', '#5a5a5a', '#8c8f1a', '#0f8c9c',
    #     '#8da5c4', '#d4995a', '#6bb06d', '#d47a7a', '#9580b0'
    # ]
    #
    # # Create a list to store data for Excel
    # excel_data = []
    # center_viruses = []
    #
    # # List of special viruses to highlight with stars
    # special_viruses = [
    #     'A/Bangkok/1/1979', 'A/Beijing/353/1989', 'A/Brisbane/10/2007',
    #     'A/California/7/2004', 'A/Cambodia/e0826360/2020',
    #     'A/Croatia/10136RV/2023',
    #     'A/Darwin/6/2021', 'A/Darwin/9/2021', 'A/Fujian/411/2002',
    #     'A/Hong Kong/2671/2019', 'A/Hong Kong/45/2019', 'A/Hong Kong/4801/2014',
    #     'A/Johannesburg/33/1994', 'A/Kansas/14/2017', 'A/Leningrad/360/1986',
    #     'A/Massachusetts/18/2022', 'A/Moscow/10/99', 'A/Perth/16/2009',
    #     'A/Philippines/2/1982', 'A/Shangdong/9/1993', 'A/Sichuan/2/1987',
    #     'A/Singapore/INFIMH-16-0019/2016', 'A/South Australia/34/2019',
    #     'A/Switzerland/8060/2017', 'A/Switzerland/9715293/2013',
    #     'A/Sydney/5/1997',
    #     'A/Texas/50/2012', 'A/Thailand/8/2022', 'A/Victoria/361/2011',
    #     'A/Wellington/1/2004', 'A/Wisconsin/67/2005', 'A/Wuhan/359/1995'
    # ]
    #
    # # Find indices of special viruses
    # special_indices = []
    # for virus_name in special_viruses:
    #     try:
    #         idx = data.virus_names.index(virus_name)
    #         special_indices.append(idx)
    #     except ValueError:
    #         print(f"Warning: {virus_name} not found in dataset")
    #
    # plt.figure(figsize=(14, 14), dpi=300)  # Larger figure to accommodate labels
    # plt.grid(False)
    # plt.xticks([])
    # plt.yticks([])
    #
    # # Plot regular clusters first
    # for i in range(8):
    #     idxs = np.where(clusters == i)[0]
    #     cluster_points = z1_2d[idxs]
    #
    #     if len(idxs) > 0:
    #         # Calculate cluster center
    #         cluster_center = np.mean(cluster_points, axis=0)
    #
    #         # Find the virus closest to the cluster center
    #         distances = np.sum((cluster_points - cluster_center) ** 2, axis=1)
    #         closest_idx = idxs[np.argmin(distances)]
    #         center_virus_name = data.virus_names[closest_idx]
    #         center_viruses.append((closest_idx, center_virus_name))
    #
    #         # Add data to Excel list
    #         for idx in idxs:
    #             excel_data.append({
    #                 'virus_name': data.virus_names[idx],
    #                 'cluster': i,
    #                 'color': colors[i],
    #                 'is_center': idx == closest_idx,
    #                 'is_special': idx in special_indices
    #             })
    #
    #         # Plot points
    #         plt.scatter(z1_2d[idxs, 0], z1_2d[idxs, 1],
    #                     c=colors[i], s=80,
    #                     edgecolor='black', linewidth=0.8,
    #                     alpha=1, zorder=3)
    #
    #         # Plot cluster center
    #         plt.scatter(cluster_center[0], cluster_center[1],
    #                     c=dark_colors[i], s=330,
    #                     edgecolor='black', linewidth=0.8,
    #                     zorder=4, marker='o')
    #         plt.scatter(cluster_center[0], cluster_center[1],
    #                     c='none',
    #                     s=120,
    #                     edgecolor='#FF0000', linewidth=0.8,
    #                     zorder=5,
    #                     marker='^')
    #
    #         # Annotate the center virus
    #         plt.annotate(center_virus_name,
    #                      (cluster_center[0], cluster_center[1]),
    #                      ha='center', va='center',
    #                      fontsize=10,
    #                      weight='bold',
    #                      color='black', zorder=6)
    #
    # # Calculate the range of x and y coordinates for positioning labels
    # x_range = np.max(z1_2d[:, 0]) - np.min(z1_2d[:, 0])
    # y_range = np.max(z1_2d[:, 1]) - np.min(z1_2d[:, 1])
    #
    # # Now add the special virus markers with stars
    # for idx in special_indices:
    #     virus_name = data.virus_names[idx]
    #     x, y = z1_2d[idx, 0], z1_2d[idx, 1]
    #
    #     # Plot the star marker (pentagram) for special viruses
    #     plt.scatter(x, y,
    #                 c='yellow', s=200,
    #                 edgecolor='black', linewidth=0.8,
    #                 marker='*', zorder=10)
    #
    #     # Add the virus name label with an arrow pointing to the star
    #     # Create offset to prevent label overlap
    #     angle = np.random.uniform(0, 2 * np.pi)
    #     offset_distance = 0.06 * np.sqrt(x_range ** 2 + y_range ** 2)
    #     offset_x = np.cos(angle) * offset_distance
    #     offset_y = np.sin(angle) * offset_distance
    #
    #     plt.annotate(virus_name,
    #                  xy=(x, y),  # Point to annotate
    #                  xytext=(x + offset_x, y + offset_y),  # Label position
    #                  fontsize=8,
    #                  bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7),
    #                  arrowprops=dict(arrowstyle='->',
    #                                  connectionstyle='arc3,rad=0.2',
    #                                  color='black'),
    #                  ha='center', va='center',
    #                  zorder=11)

    # plt.title('H3N2', fontsize=18, fontweight='bold', pad=20)
    # plt.tight_layout()
    # plt.show()

    # Create and save Excel file
    # df = pd.DataFrame(excel_data)
    # df.to_excel('virus_clusters.xlsx', index=False)
    # plt.show()
    # 在图像上方中央添加标签

    # 设置画布背景
    # plt.gca().set_facecolor('#f0f0f0')
    # plt.title('H3N2', fontsize=18, fontweight='bold', pad=20)
    # plt.title('B_Victoria', fontsize=18, fontweight='bold', pad=20)
    # plt.title('H3N2', fontsize=18, fontweight='bold', pad=20)
    # plt.title('H1N1', fontsize=18, fontweight='bold', pad=20)
    # plt.title('H3N2', fontsize=18, fontweight='bold', pad=20)
    # 保存配置

    # plt.savefig('clustering_nature566H1N1不对称.pdf', dpi=600,
    #             pad_inches=0.02,
    #             transparent=False, facecolor='white', format='pdf')
    # plt.savefig('clustering_nature566H3N2不对称.pdf', dpi=600,
    #              pad_inches=0.1,
    #             transparent=False, facecolor='white', format='pdf')
    # plt.savefig('clustering_nature585BVictoria不对称.pdf', dpi=600,
    #              pad_inches=0.1,
    #             transparent=False, facecolor='white', format='pdf')
    # plt.savefig('clustering_nature585BYamagata不对称.pdf', dpi=600,
    #              pad_inches=0.1,
    #             transparent=False, facecolor='white', format='pdf')
    # plt.savefig('clustering_nature566H1N1.pdf', dpi=600,
    #              pad_inches=0.1,
    #             transparent=False, facecolor='white', format='pdf')
    # plt.savefig('clustering_nature566H3N2.pdf', dpi=600,
    #              pad_inches=0.1,
    #             transparent=False, facecolor='white', format='pdf')
    # print(111)
    # plt.savefig('clustering_nature585BVictoria.pdf', dpi=600,
    #              pad_inches=0.1,
    #             transparent=False, facecolor='white', format='pdf')
    # plt.savefig('clustering_nature585BYamagata.pdf', dpi=600,
    #              pad_inches=0.1,
    #             transparent=False, facecolor='white', format='pdf')



    # plt.savefig('clustering_optimized.eps', dpi=600,
    #             bbox_inches='tight', pad_inches=0.1,
    #             transparent=False, facecolor='white', format='eps')
    # plt.show()


    # # 首先，初始化GCNDecoder
    # decoder = GCNDecoder(num_hidden).to(device)  # num_hidden是encoder的输出维度
    # # 为GCNDecoder设置优化器 ，同时随机初始化了参数
    # decoder_optimizer = torch.optim.Adam(decoder.parameters(),
    #                                      lr=decoder_learning_rate,
    #                                      weight_decay=weight_decay_decoder)
    # model = GraphModel(num_features, num_hidden, F.relu,
    #                    base_model=base_model, k=num_layers, skip=True)
    data.x = z1
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_idx = 0
    total_mse = 0
    total_mae = 0
    total_r_squared = 0
    for train_idx, test_idx in kf.split(range(data.num_edges)):#循环5次
        print(f"Processing fold {fold_idx + 1}")
        fold_idx += 1
        inner_train_idx,inner_val_idx = train_test_split(train_idx, test_size=0.2,
                                                    random_state=42)##从训练集中划分出验证集，确保训练集，验证集，测试集互不交叉
        #inner_train_idx,inner_val_idx边的起始节点和终止节点
        inner_train_edge_attr = data.edge_attr[inner_train_idx]#边的实际值
        inner_val_edge_attr = data.edge_attr[inner_val_idx]
        test_edge_attr = data.edge_attr[test_idx]
        # print(inner_train_idx)
        # print(inner_val_idx)
        # print(test_idx)
        # print(data.edge_index[:, test_idx].shape)
        # model.to(device)
        # optimizer = torch.optim.Adam(model.parameters(), lr=encoder_learning_rate,
        #                                      weight_decay=weight_decay)
        # encoder = Encoder(num_features, num_hidden, F.relu,
        #                   base_model=base_model,
        #                   k=num_layers).to(device)  # 2次聚合，同时随机初始化了参数
        # encoder_optimizer = torch.optim.Adam(encoder.parameters(),
        #                                      lr=encoder_learning_rate,
        #                                      weight_decay=weight_decay)
        # # 首先，初始化GCNDecoder
        decoder = GCNDecoder(num_hidden).to(device)  # num_hidden是encoder的输出维度
        # 为GCNDecoder设置优化器 ，同时每一折重新随机初始化了参数,避免上一折的训练集在这一折的测试集中导致信息泄露
        decoder_optimizer = torch.optim.Adam(decoder.parameters(),
                                             lr=decoder_learning_rate,
                                             weight_decay=weight_decay_decoder)
        scheduler = torch.optim.lr_scheduler.StepLR(decoder_optimizer,
                                                    step_size=100,
                                                    gamma=0.9)
        best_val_loss_fine_tuning = float('inf')  # 为微调阶段也设置一个最佳损失记录器
        best_model_path_fine_tuning = 'best_decoder_model_fine_tuning.pth'  # 最佳模型保存路径
        for epoch in range(1, 1500):#经测试预训练模型参数加载成功
            # model.train()
            # optimizer.zero_grad()
            # encoder.train()
            decoder.train()
            # encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            # encoded_x = encoder(data.x, data.edge_index[:, inner_train_idx])
            # print(encoded_x.shape)
            # predicted_edge_attr = decoder(encoded_x,data.edge_index[:, inner_train_idx])
            predicted_edge_attr = decoder(data.x, data.edge_index[:, inner_train_idx])
            # 因为预测的是每条边的属性，我们直接使用data.edge_attr作为真实值计算MSE损失
            train_loss = F.mse_loss(predicted_edge_attr, inner_train_edge_attr)
            train_loss.backward()
            # encoder_optimizer.step()
            decoder_optimizer.step()  # 更新参数
            # optimizer.step()
            scheduler.step()
            # encoder.eval()
            # decoder.eval()
            decoder.eval()
            # 每5轮在测试集上评估一次（仅用于监控，不用测试集结果调参，不影响训练）
            if epoch % 5 == 0:
                decoder.eval()
                with torch.no_grad():
                    # 在测试集上进行预测
                    test_predicted_edge_attr = decoder(data.x,
                                                       data.edge_index[:,
                                                       test_idx])
                    test_loss = F.mse_loss(test_predicted_edge_attr,
                                           test_edge_attr)
                    # 计算 MAE
                    test_mae = mean_absolute_error(test_edge_attr.cpu().numpy(),
                                                   test_predicted_edge_attr.cpu().numpy())
                    # 计算 R²
                    test_r_squared = r2_score(test_edge_attr.cpu().numpy(),
                                              test_predicted_edge_attr.cpu().numpy())

                    print(
                        f'    [Test Evaluation at Epoch {epoch}] Test Loss: {test_loss.item():.4f}, MAE: {test_mae:.4f}, R²: {test_r_squared:.4f}')
            with torch.no_grad():
                # encoded_x = encoder(data.x, data.edge_index[:,inner_val_idx])
                # 对验证集进行预测
                # val_predicted_edge_attr = decoder(encoded_x,
                #                                    data.edge_index[:,inner_val_idx])
                # 计算测试损失
                val_predicted_edge_attr = decoder(data.x,
                                                  data.edge_index[:,
                                                  inner_val_idx])
                val_loss = F.mse_loss(val_predicted_edge_attr,inner_val_edge_attr)
            print(
                f'Decoder Training Epoch: {epoch:03d}, Train Loss: {train_loss.item():.4f}, val_loss Loss: {val_loss.item():.4f}')
            if val_loss < best_val_loss_fine_tuning:#如果验证集效果更好，使用验证集的参数
                best_val_loss_fine_tuning = val_loss
                # torch.save(decoder.state_dict(), best_model_path_fine_tuning)
                torch.save(decoder.state_dict(), best_model_path_fine_tuning)

        decoder.load_state_dict(torch.load(best_model_path_fine_tuning))
        # model.load_state_dict(torch.load(best_model_path_fine_tuning))
        with torch.no_grad():#结束后对测试集进行一次测试
            # encoded_x = encoder(data.x, data.edge_index[:, test_idx])
            # # 对测试集进行预测
            # test_predicted_edge_attr = decoder(encoded_x,
            #                                    data.edge_index[:, test_idx])
            # 计算测试损失
            test_predicted_edge_attr = decoder(data.x, data.edge_index[:, test_idx])
            test_loss = F.mse_loss(test_predicted_edge_attr,test_edge_attr)
            # 计算 MAE
            mae = mean_absolute_error(test_edge_attr.cpu().numpy(),
                                      test_predicted_edge_attr.cpu().numpy())
            # 计算 R²
            r_squared = r2_score(test_edge_attr.cpu().numpy(),
                                 test_predicted_edge_attr.cpu().numpy())

        # 输出当前epoch的训练损失和测试损失
        print(
            f'Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}')
        print(f'MAE: {mae:.4f}')
        print(f'R²: {r_squared:.4f}')




        # predicted = test_predicted_edge_attr.cpu().numpy().flatten()
        # actual = test_edge_attr.cpu().numpy().flatten()
        # # 样式配置
        # sns.set(style="white", context="paper", font_scale=1.2)
        # plt.figure(figsize=(6, 6), dpi=600)  # 稍小的画布更显精致
        # # 绘制散点图（调整颜色和大小）
        # sc = plt.scatter(actual, predicted,
        #                  c='#4C72B0',  # 更现代的科技蓝
        #                  alpha=1,  # 增加透明度显示密度分布
        #                  edgecolors='none',  # 去除边框
        #                  s=8,  # 更小的点尺寸
        #                  zorder=3)
        # # 理想参考线（改为更柔和的配色）
        # lims = [np.min([actual.min(), predicted.min()]),
        #         np.max([actual.max(), predicted.max()])]
        # plt.plot(lims, lims, '--',
        #          color='#D55E00',  # 互补色橙色系
        #          lw=1.2,
        #          zorder=2,
        #          label="Ideal")
        # # 坐标轴美化
        # plt.xlabel('Actual Values', fontsize=12, labelpad=8)
        # plt.ylabel('Predicted Values', fontsize=12, labelpad=8)
        # plt.xlim(lims)
        # plt.ylim(lims)
        #
        # # 网格和刻度优化
        # plt.gca().set_aspect('equal')
        # plt.tick_params(axis='both',
        #                 which='major',
        #                 labelsize=10,  # 更紧凑的刻度标签
        #                 length=3,
        #                 width=0.8)
        # plt.grid(True,
        #          linestyle=':',  # 改为点线
        #          color='lightgray',
        #          alpha=0.7,
        #          zorder=1)
        #
        # # 紧凑布局
        # plt.tight_layout(pad=1.5)
        # plt.savefig('scatterplot.pdf', format='pdf', bbox_inches='tight')
        # plt.close()

        total_mse += test_loss
        total_mae += mae
        total_r_squared += r_squared
        torch.cuda.empty_cache()
    average_mse = total_mse / 5
    average_mae = total_mae / 5
    average_r_squared = total_r_squared / 5
    print(
        f'Average MSE: {average_mse}, Average MAE: {average_mae}, Average R²: {average_r_squared}')
