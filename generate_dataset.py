import networkx as nx
import numpy as np
import z3
import random
import pickle
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from graph_coloring import is_colorable

def generate_single_graph(min_nodes, max_nodes, edge_frac, chr):
    """
    生成单个随机图并判断其可染色性
    :return: (graph, colorable) 元组，如果求解超时返回None
    """
    # 随机生成节点数
    num_nodes = random.randint(min_nodes, max_nodes)
    # 生成随机图
    edge_prob = edge_frac * (num_nodes - 1) / (num_nodes * (num_nodes - 1) / 2)
    graph = nx.erdos_renyi_graph(num_nodes, edge_prob)
    # 判断是否可k染色
    colorable = is_colorable(graph, chr)
    
    # 如果遇到超时（colorable为None），返回None
    if colorable is not None:
        return (graph, colorable)
    return None

def generate_random_graphs(num_graphs, min_nodes=5, max_nodes=50, edge_frac_min=3, edge_frac_max=4, chr=4, resume_file=None, max_workers=None):
    """
    生成随机图数据集，支持断点续传和并行处理
    :param num_graphs: 生成的图数量
    :param min_nodes: 最小节点数
    :param max_nodes: 最大节点数
    :param edge_frac: 边比例系数
    :param chr: 颜色数
    :param resume_file: 用于断点续传的文件路径，如果为None则从头开始
    :param max_workers: 并行工作进程数，如果为None则使用CPU核心数
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
    
    # 计算还需要生成的图数量
    remaining_graphs = num_graphs - start_idx
    
    # 如果没有指定工作进程数，使用CPU核心数
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 8)  # 限制最多使用8个核心以避免系统负载过高
    
    print(f"使用 {max_workers} 个并行进程生成图...")
    
    # 使用进程池并行生成图
    i = start_idx
    futures = set()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 初始提交一批任务
        for _ in range(min(remaining_graphs, max_workers * 2)):  # 提交2倍工作进程数的任务以保持队列饱满
            edge_frac = random.uniform(edge_frac_min, edge_frac_max)
            future = executor.submit(generate_single_graph, min_nodes, max_nodes, edge_frac, chr)
            futures.add(future)
        
        while i < num_graphs:
            # 等待任何一个任务完成
            for future in as_completed(futures):
                futures.remove(future)
                result = future.result()
                
                # 如果任务成功生成了有效图
                if result is not None:
                    graph, colorable = result
                    dataset.append((graph, colorable))
                    i += 1
                    
                    # 打印进度
                    print(f"已生成 {i}/{num_graphs} 个图")
                    
                    # 定期保存进度（每生成10个图保存一次）
                    if i % 100 == 0 and resume_file:
                        save_dataset(dataset, f"step{i}_{resume_file}")
                    
                    # 提交新任务以保持队列中有足够的工作
                    if i < num_graphs:
                        future = executor.submit(generate_single_graph, min_nodes, max_nodes, edge_frac, chr)
                        futures.add(future)
                    break
                else:
                    # 如果任务返回None（超时），重新提交一个任务
                    future = executor.submit(generate_single_graph, min_nodes, max_nodes, edge_frac, chr)
                    futures.add(future)
        # 生成完成后强制终止所有子进程
    for p in multiprocessing.active_children():
        p.terminate()
        p.join(timeout=1)
        if p.is_alive():
            p.kill()
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

if __name__ == "__main__":
    def main():
        try:
            # 配置参数 - 测试用小数据集
            # num_graphs = 10000  # 减少图数量以便快速测试
            num_graphs = 200
            # min_nodes = 50   # 减少最小节点数
            min_nodes = 100   # 减少最小节点数
            # max_nodes = 500   # 减少最大节点数
            max_nodes = 150   # 减少最大节点数
            edge_frac_min = 4.1-0.2
            edge_frac_max = 4.1+0.6
            chr = 4
            dataset_file = "test_graph_dataset.pkl"
            
            # 生成数据集（支持断点续传和并行处理）
            print("开始生成随机图数据集...")
            dataset = generate_random_graphs(
                num_graphs=num_graphs, 
                min_nodes=min_nodes, 
                max_nodes=max_nodes, 
                edge_frac_min=edge_frac_min,
                edge_frac_max=edge_frac_max,
                chr=chr,
                resume_file=dataset_file,  # 使用数据集文件作为断点续传文件
                max_workers=None  # 自动使用CPU核心数
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
    
    main()
