import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# 导入模型定义
from gnn_model import GCN, load_model, demo, graph_to_data, graph_conflict_rate, loss_fn


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
        loss = criterion(out, data.edge_index)
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
            loss = criterion(out, data.edge_index)
            total_loss += loss.item()
            
            # 获取预测结果
            # preds = out.argmax(dim=1).cpu().numpy()
            # labels = data.y.cpu().numpy()
            # all_preds.extend(preds)
            # all_labels.extend(labels)
    
    # 计算准确率和F1分数
    # accuracy = accuracy_score(all_labels, all_preds)
    # f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return total_loss / len(loader) #, accuracy, f1

def test_model(model, loader, device):
    """
    测试模型
    """
    model.eval()
    vals = []
    preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            rate = graph_conflict_rate(out, data.edge_index)
            vals.append(rate)
            
            # 获取预测结果
            labels = data.colorable.cpu().numpy()
            all_labels.extend(labels)
    
    # 计算指标
    # accuracy = accuracy_score(all_labels, all_preds)
    # f1 = f1_score(all_labels, all_preds, average='weighted')
    # conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return sum(vals) / len(vals), all_labels

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载和预处理数据
    train_loader, val_loader, test_loader = load_and_preprocess_data("test_graph_dataset.pkl")
    
    # 初始化模型
    # model = GCN(hidden_channels=256).to(device)
    model = GCN().to(device)
    print(model)
    
    # 设置优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    # num_epochs = 1000
    num_epochs = 100
    best_val_loss = 0.0
    best_model_path = "best_4color_model.pth"
    
    print("开始训练...")
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        
        # 保存最佳模型
        if epoch == 0 or val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"第 {epoch+1} 轮: 验证损失降低至 {val_loss:.8f}，保存模型到 {best_model_path}")
        
        if (epoch+1) % 10 == 0:
            print(f"第 {epoch+1}/{num_epochs} 轮: 训练损失={train_loss:.8f}, 验证损失={val_loss:.8f}")
    
    # 加载最佳模型并测试
    print("\n开始测试...")
    best_model = load_model(best_model_path, device)
    test_accuracy, all_labels = test_model(best_model, test_loader, device)
    
    # 打印测试结果
    print(f"测试准确率: {test_accuracy:.8f}")
    
    # 统计测试集中可染色和不可染色的数量
    colorable_test = sum(all_labels)
    non_colorable_test = len(all_labels) - colorable_test
    print(f"\n测试集中可4染色图数: {colorable_test}")
    print(f"测试集中不可4染色图数: {non_colorable_test}")
    
    # 调用演示函数，展示如何加载和使用模型
    demo(best_model_path, device)


if __name__ == "__main__":
    main()
