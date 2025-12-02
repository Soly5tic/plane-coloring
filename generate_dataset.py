import networkx as nx
import numpy as np
import z3
import random
import pickle
import os

from graph_coloring import is_colorable

def generate_random_graphs(num_graphs, min_nodes=5, max_nodes=50, edge_frac=3, chr=4, resume_file=None):
    """
    生成随机图数据集，支持断点续传
    :param num_graphs: 生成的图数量
    :param min_nodes: 最小节点数
    :param max_nodes: 最大节点数
    :param edge_frac: 边比例系数
    :param resume_file: 用于断点续传的文件路径，如果为None则从头开始
    :return: 列表，每个元素是(graph, is_colorable)元组
    """
    dataset = []
    start_idx = 0
    
    # 检查是否需要从断点续传
    if resume_file and os.path.exists(resume_file):
        dataset = load_dataset(resume_file)
        start_idx = len(dataset)
        print(f"从断点续传：已生成 {start_idx} 个图，还需生成 {num_graphs - start_idx} 个图")
    
    # 如果已经生成了足够的图，直接返回
    if start_idx >= num_graphs:
        print(f"已生成 {start_idx} 个图，超过了要求的 {num_graphs} 个图")
        return dataset[:num_graphs]
    
    i = start_idx
    while i < num_graphs:
        # 随机生成节点数
        num_nodes = random.randint(min_nodes, max_nodes)
        # 生成随机图
        edge_prob = edge_frac * (num_nodes - 1) / (num_nodes * (num_nodes - 1) / 2)
        graph = nx.erdos_renyi_graph(num_nodes, edge_prob)
        # 判断是否可k染色
        colorable = is_colorable(graph, chr)
        
        # 如果遇到超时（colorable为None），跳过当前图并重新生成
        if colorable is not None:
            dataset.append((graph, colorable))
            i += 1  # 只在成功添加有效图时增加计数
            
            # 打印进度
            if i % 1 == 0:
                print(f"已生成 {i}/{num_graphs} 个图")
            
            # 定期保存进度（每生成10个图保存一次）
            if i % 100 == 0 and resume_file:
                save_dataset(dataset, f"step{i}_{resume_file}")
        else:
            print(f"当前图求解超时，正在重新生成...")
    
    return dataset

def save_dataset(dataset, filename):
    """
    保存数据集到文件
    :param dataset: 要保存的数据集
    :param filename: 保存文件名
    """
    print("开始保存数据集...")
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"数据集已保存到 {filename}")

def load_dataset(filename):
    """
    从文件加载数据集
    :param filename: 数据集文件名
    :return: 加载的数据集
    """
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)
    print(f"从 {filename} 加载了 {len(dataset)} 个图")
    return dataset

def main():
    try:
        # 配置参数
        num_graphs = 100
        min_nodes = 50
        max_nodes = 500
        edge_frac = 4.1
        chr = 4
        dataset_file = "graph_coloring_dataset.pkl"
        
        # 生成数据集（支持断点续传）
        print("开始生成随机图数据集...")
        dataset = generate_random_graphs(
            num_graphs=num_graphs, 
            min_nodes=min_nodes, 
            max_nodes=max_nodes, 
            edge_frac=edge_frac,
            chr=chr,
            resume_file=dataset_file  # 使用数据集文件作为断点续传文件
        )
        
        # 统计可染色和不可染色的图数量
        colorable_count = sum(1 for _, colorable in dataset if colorable)
        non_colorable_count = len(dataset) - colorable_count
        print(f"数据集统计：")
        print(f"总图数：{len(dataset)}")
        print(f"可{chr}染色图数：{colorable_count} ({colorable_count/len(dataset)*100:.2f}%)")
        print(f"不可{chr}染色图数：{non_colorable_count} ({non_colorable_count/len(dataset)*100:.2f}%)")
        
        # 最终保存数据集
        save_dataset(dataset, dataset_file)
    except Exception as e:
        print(f"发生错误：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
