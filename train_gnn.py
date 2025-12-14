import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, BatchNorm
import networkx as nx
import numpy as np
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

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

def load_and_preprocess_data(filename):
    """
    加载数据集并转换为PyTorch Geometric所需格式
    :param filename: 数据集文件名
    :return: 训练集、验证集和测试集的DataLoader
    """
    # 加载数据集
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"加载了 {len(dataset)} 个图")
    
    # 转换为Data对象
    data_list = [graph_to_data(graph, 1 if colorable else 0) for graph, colorable in dataset]
    
    # 划分数据集
    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2
    
    print(f"训练集大小: {len(train_data)}")
    print(f"验证集大小: {len(val_data)}")
    print(f"测试集大小: {len(test_data)}")
    
    # 创建DataLoader
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    return train_loader, val_loader, test_loader

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
        x = torch_geometric.nn.global_mean_pool(x, batch)
        
        # 全连接层分类
        x = self.lin(x)
        
        return x

def train(model, loader, optimizer, criterion, device):
    """
    训练模型
    """
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    """
    评估模型
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item()
            
            # 获取预测结果
            preds = out.argmax(dim=1).cpu().numpy()
            labels = data.y.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    # 计算准确率和F1分数
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return total_loss / len(loader), accuracy, f1

def test_model(model, loader, device):
    """
    测试模型
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            
            # 获取预测结果
            preds = out.argmax(dim=1).cpu().numpy()
            labels = data.y.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return accuracy, f1, conf_matrix, all_preds, all_labels

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
    :return: 预测结果（0或1）
    """
    # 将图转换为Data对象
    data = graph_to_data(graph, 0)  # 标签这里不重要，只用于预测
    data = data.to(device)
    
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1).item()
    
    return pred

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
    pred = predict_graph(model, test_graph, device)
    
    # 输出结果
    print(f"测试图节点数: {test_graph.number_of_nodes()}")
    print(f"测试图边数: {test_graph.number_of_edges()}")
    print(f"模型预测结果: {'可3染色' if pred == 1 else '不可3染色'}")
    
    # 如果我们有实际标签，可以比较
    # 注意：这里为了演示，我们不计算实际标签，直接输出预测结果
    print("\n=== 演示结束 ===")

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载和预处理数据
    train_loader, val_loader, test_loader = load_and_preprocess_data("test_graph_dataset.pkl")
    
    # 初始化模型
    model = GCN(hidden_channels=256).to(device)
    print(model)
    
    # 设置优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 训练模型
    num_epochs = 1000
    best_val_accuracy = 0.0
    best_model_path = "best_3color_model.pth"
    
    print("开始训练...")
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy, val_f1 = evaluate(model, val_loader, criterion, device)
        
        # 保存最佳模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"第 {epoch+1} 轮: 验证准确率提升至 {best_val_accuracy:.4f}，保存模型到 {best_model_path}")
        
        if (epoch+1) % 10 == 0:
            print(f"第 {epoch+1}/{num_epochs} 轮: 训练损失={train_loss:.4f}, 验证损失={val_loss:.4f}, 验证准确率={val_accuracy:.4f}, 验证F1={val_f1:.4f}")
    
    # 加载最佳模型并测试
    print("\n开始测试...")
    best_model = load_model(best_model_path, device)
    test_accuracy, test_f1, conf_matrix, all_preds, all_labels = test_model(best_model, test_loader, device)
    
    # 打印测试结果
    print(f"测试准确率: {test_accuracy:.4f}")
    print(f"测试F1分数: {test_f1:.4f}")
    print("混淆矩阵:")
    print(conf_matrix)
    
    # 统计测试集中可染色和不可染色的数量
    colorable_test = sum(all_labels)
    non_colorable_test = len(all_labels) - colorable_test
    print(f"\n测试集中可3染色图数: {colorable_test}")
    print(f"测试集中不可3染色图数: {non_colorable_test}")
    
    # 调用演示函数，展示如何加载和使用模型
    demo(best_model_path, device)

if __name__ == "__main__":
    # 解决循环导入问题
    import torch_geometric
    main()
