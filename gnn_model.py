import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, BatchNorm, global_mean_pool
import networkx as nx


def graph_to_data(graph, label):
    """
    将NetworkX图转换为PyTorch Geometric的Data对象
    :param graph: NetworkX图
    :param label: 标签（0或1）
    :return: Data对象
    """
    # 获取节点特征（这里使用度作为初始特征）
    degrees = [graph.degree(n) for n in graph.nodes()]
    x = torch.tensor(degrees, dtype=torch.float).view(-1, 1)
    
    # 转换边列表
    edges = list(graph.edges())
    # 添加反向边以构建无向图
    edge_index = torch.tensor(edges + [(v, u) for u, v in edges], dtype=torch.long).t().contiguous()
    
    # 标签
    y = torch.tensor([label], dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, y=y)


class GCN(torch.nn.Module):
    """
    图卷积网络模型
    """
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(1, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 2)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 第一层GCN
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # 第二层GCN
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # 第三层GCN
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        # 全局池化（取每个图的所有节点特征的平均值）
        x = global_mean_pool(x, batch)
        
        # 全连接层分类
        x = self.lin(x)
        
        return x


def load_model(model_path, device):
    """
    加载已保存的模型
    :param model_path: 模型文件路径
    :param device: 运行设备
    :return: 加载后的模型
    """
    model = GCN(hidden_channels=256).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"模型已从 {model_path} 加载")
    return model


def predict_graph(model, graph, device):
    """
    使用模型预测单个图是否可4染色
    :param model: 训练好的模型
    :param graph: NetworkX图
    :param device: 运行设备
    :return: 预测概率（0到1之间的连续值，表示图可4染色的概率）
    """
    # 将图转换为Data对象
    data = graph_to_data(graph, 0)  # 标签这里不重要，只用于预测
    data = data.to(device)
    
    model.eval()
    with torch.no_grad():
        out = model(data)
        prob = F.softmax(out, dim=1)
        pred_prob = prob[0][1].item()  # 获取可4染色（索引1）的概率
    
    return pred_prob


def demo(model_path, device):
    """
    演示如何加载和使用模型
    """
    print("\n=== 模型使用演示 ===")
    
    # 加载模型
    model = load_model(model_path, device)
    
    # 生成一个测试图
    test_graph = nx.erdos_renyi_graph(15, 0.3)
    
    # 使用模型进行预测
    pred_prob = predict_graph(model, test_graph, device)
    
    # 输出结果
    print(f"测试图节点数: {test_graph.number_of_nodes()}")
    print(f"测试图边数: {test_graph.number_of_edges()}")
    print(f"模型预测可4染色的概率: {pred_prob:.4f}")
    print(f"预测类别: {'可4染色' if pred_prob >= 0.5 else '不可4染色'}")
    
    # 如果我们有实际标签，可以比较
    # 注意：这里为了演示，我们不计算实际标签，直接输出预测结果
    print("\n=== 演示结束 ===")