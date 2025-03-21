import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, Dataset
from torch.utils.data import DataLoader

# 从指定文件里面加载数据，该文件是每个HA1对应的327*100 维特征表示
def load_vec(matrices_path):
    data = np.genfromtxt(matrices_path)
    print(data.shape)
    name = data[:, 0:1]  # 毒株名
    vec = data[:, 1:]  # 毒株嵌入，没有卷积过
    # print(vec.shape)
    num = vec.shape[0]
    # vec = vec.reshape(num, 100, 327)  # 253 个 100*327 矩阵
    matrices = vec.reshape(num, 327, 100)  # 253 个 327*100 矩阵
    # print(vec.shape)
    # print(vec[0])
    # print(vec[0].shape)
    matrices = torch.FloatTensor(vec)  # 转换为tensor  将特征向量重塑为 327x100 的矩阵，并转换为 PyTorch 的张量
    #
    # inputs = inputs.unsqueeze(1)  # 添加一维(253,1,100,327)  (253,1,327,100)
    return matrices


matrices_path = "data/H3N2/HA_vec.csv"##就是H3N2的氨基酸序列转化为327*100维嵌入向量后的结果
# result = load_vec(matrices_path)
# print(result.shape)

distances_csv_path = "data/H3N2/AH3N2_combine.csv"


def load_dis(dis_path):
    csv_data = pd.read_csv(dis_path)#从CSV文件中加载距离数据。
    return csv_data

def extract_features(matrices,csv_data):     #从嵌入向量矩阵中提取出每个毒株的嵌入向量，并把成对毒株的两个嵌入向量合并起来
    features_list = []                      #直接将两个特征矩阵合并为一个特征向量，并将所有特征和标签转换为 TensorDataset。
    labels_list = []
    #matrices是毒株嵌入向量数据，csv_data是毒株对的抗原距离
    # 对CSV中的每一行，提取矩阵特征并创建标签
    for _, row in csv_data.iterrows():
        matrix1 = matrices[int(row['strainName1'])]
        matrix2 = matrices[int(row['strainName2'])]#matrix1 和 matrix2 分别从 matrices 中提取出对应毒株的特征矩阵,即毒株的嵌入向量

        feature1 = matrix1
        feature2 = matrix2#将提取的毒株嵌入向量，即特征矩阵分别赋值给 feature1 和 feature2

        # 提取或计算特征,# 可以直接将两个毒株嵌入向量合并，也可以计算它们之间的差异或其他关系
        combined_feature = torch.cat((feature1, feature2), dim=0)

        # 将特征和标签添加到列表中
        features_list.append(combined_feature)
        labels_list.append(row['distance'])

        # 转换列表为tensor
        features_tensor = torch.stack(features_list)
        labels_tensor = torch.tensor(labels_list, dtype=torch.float32)

        # 创建TensorDataset
        dataset = TensorDataset(features_tensor, labels_tensor)


class MatrixDistanceDataset(Dataset):#以原始特征对的形式返回特征向量和距离标签,MatrixDistanceDataset更加灵活，因为它可以在 __getitem__ 中进行各种特征处理，而不仅限于合并。
    def __init__(self, matrices_path, distances_csv_path):
        #matrices_path是毒株嵌入向量数据，distances_csv_path是毒株对的抗原距离
        # 加载CSV文件
        # 加载毒株嵌入向量数据
        features_data = np.genfromtxt(matrices_path)  # 假设第一行是头部
        # 提取矩阵数据和毒株名
        # 假设毒株名在第一列，矩阵数据在剩余列
        self.names = features_data[:, 0:1]  # 毒株名
        self.features = torch.FloatTensor(features_data[:, 1:])  # 特征向量

        # 加载距离数据
        distances_data = pd.read_csv(distances_csv_path)
        self.index_pairs = list(zip(distances_data.iloc[:, 0], distances_data.iloc[:, 1]))
        self.distances = torch.FloatTensor(distances_data.iloc[:, 2].values)

    def __len__(self):
        # 返回CSV数据的长度，即矩阵对的数量
        return len(self.index_pairs)

    def __getitem__(self, idx):
        # 获取索引idx对应的数据对索引
        index_pair = self.index_pairs[idx]

        # 获取索引对应的特征向量
        feature1 = self.features[index_pair[0]]
        feature2 = self.features[index_pair[1]]

        # 获取索引对应的距离
        distance = self.distances[idx]

        # 返回特征向量对和距离
        return (feature1, feature2), distance


dataset = MatrixDistanceDataset(matrices_path, distances_csv_path)

# 创建DataLoader实例
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 使用DataLoader
for batch in dataloader:
    (features1, features2), distances = batch
    print(f"Batch features1 shape: {features1.shape}")
    print(f"Batch features2 shape: {features2.shape}")
    print(f"Batch distances shape: {distances.shape}")
    # 这里你可以将features1, features2和distances送入你的模型。