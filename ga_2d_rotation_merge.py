import numpy as np
import networkx as nx
import random
import copy
import matplotlib.pyplot as plt
from typing import List, Optional
from z3 import Solver, Bool, Or, And, Not
import torch
import os

from udg_builder import UDGBuilder 
from graph_coloring import is_colorable
from gnn_model import load_model, predict_graph

class UDGBuilderWrapper:
    """
    对 UDGBuilder 的封装，增加了适合 GA 的深拷贝和特定变异逻辑。
    """
    def __init__(self, builder, model=None, device=None):
        self.builder = builder
        self.fitness = 0.0
        self.graph_cache = None
        self.model = model
        self.device = device

    def copy(self):
        """深拷贝当前个体，用于产生后代"""
        new_builder = copy.deepcopy(self.builder)
        return UDGBuilderWrapper(new_builder, self.model, self.device)

    def update_fitness(self):
        """
        计算适应度：使用平均度数作为适应性分数。
        """
        self.builder.clean_isolated_nodes()
        G = self.builder.get_graph()
        self.graph_cache = G
        
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        
        if num_nodes == 0:
            self.fitness = 0.0
        else:
            self.fitness = 2.0 * num_edges / num_nodes
        
        return self.fitness

def get_rational_angles():
    """
    生成二维有理角度库。
    """
    angles = []
    for a in range(2, 13):
        for b in range(1, a):
            angles.append(np.arccos(b/a))
            angles.append(np.arccos(-b/a))
            angles.append(np.arcsin(b/a))
            angles.append(np.arcsin(-b/a))
        for b in range(2, min(a*a, 13)):
            angles.append(np.arccos(np.sqrt(b)/a))
            angles.append(np.arccos(-np.sqrt(b)/a))
            angles.append(np.arcsin(np.sqrt(b)/a))
            angles.append(np.arcsin(-np.sqrt(b)/a))
    return list(set(angles))

def k_core_prune(G: nx.Graph, k: int) -> nx.Graph:
    """
    不断删除度数 <= k 的节点，直到图中没有节点或所有节点度数均 > k。
    """
    H = G.copy()
    changed = True
    while changed and H.number_of_nodes() > 0:
        changed = False
        to_remove = [n for n, deg in H.degree() if deg <= k]
        if to_remove:
            H.remove_nodes_from(to_remove)
            changed = True
    return H

def calculate_rotation_angle_2d(point_a: np.ndarray, point_b: np.ndarray, pivot: np.ndarray) -> float:
    """
    计算将点B绕原点P旋转到点A位置所需的旋转角度（2D版本）。
    
    Args:
        point_a: 目标位置A (2D坐标)
        point_b: 需要旋转的点B (2D坐标)
        pivot: 旋转中心P (2D坐标)
        
    Returns:
        旋转角度
    """
    # 将坐标转换为以pivot为原点的相对坐标
    a_rel = point_a - pivot
    b_rel = point_b - pivot
    
    # 计算从b_rel到a_rel的旋转角度
    angle_a = np.arctan2(a_rel[1], a_rel[0])
    angle_b = np.arctan2(b_rel[1], b_rel[0])
    
    # 计算相对角度差
    rotation_angle = angle_a - angle_b
    
    # 标准化到[-π, π]范围
    while rotation_angle > np.pi:
        rotation_angle -= 2 * np.pi
    while rotation_angle < -np.pi:
        rotation_angle += 2 * np.pi
    
    return rotation_angle

def rotate_points_2d(points: np.ndarray, angle: float, pivot: np.ndarray) -> np.ndarray:
    """
    绕指定角度旋转点集（2D版本）。
    
    Args:
        points: 需要旋转的点集 (N, 2)
        angle: 旋转角度
        pivot: 旋转中心
        
    Returns:
        旋转后的点集
    """
    # 构建2D旋转矩阵
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    # 2D旋转矩阵
    R = np.array([
        [cos_angle, -sin_angle],
        [sin_angle, cos_angle]
    ])
    
    # 平移到以pivot为原点，旋转，然后平移回去
    centered_points = points - pivot
    rotated_points = (R @ centered_points.T).T + pivot
    
    return rotated_points

RATIONAL_ANGLES = get_rational_angles()

class RotationMergeGeneticSearch2D:
    def __init__(self, 
                 pop_size=20, 
                 max_nodes=2000, 
                 mutation_rate=0.8, 
                 elite_size=3, 
                 model_path="./best_4color_model.pth"):
        """
        使用旋转合并策略的2D进化算法。
        """
        self.pop_size = pop_size
        self.max_nodes = max_nodes
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.population: List[UDGBuilderWrapper] = []
        self.generation = 0
        
        self.best_fitness_history = []
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model = load_model(model_path, self.device)
            print(f"使用设备: {self.device}, 模型已加载")
        except (FileNotFoundError, RuntimeError) as e:
            print(f"无法加载模型 {model_path}: {e}")
            print("将使用平均度数作为适应度评估，不使用GNN模型")
            self.model = None

    def initialize_population(self, base_builder_cls):
        """
        初始化种群。
        """
        self.population = []
        print(f"Initializing population of size {self.pop_size}...")
        
        for _ in range(self.pop_size):
            builder = base_builder_cls(tolerance=1e-5)
            
            # 添加初始结构（Moser Spindle）
            builder.add_moser_spindle()
            
            wrapper = UDGBuilderWrapper(builder, self.model, self.device)
            wrapper.update_fitness()
            self.population.append(wrapper)

    def _rotate_merge_mutation(self, individual: UDGBuilderWrapper):
        """
        核心变异策略：2D旋转合并
        
        1. 从当前图中随机选择三个点：原点P和两个点A, B
        2. 计算将B绕P旋转到A位置的旋转角度
        3. 对整个图进行2D旋转
        4. 合并原图与旋转后的图
        """
        builder = individual.builder
        current_nodes = builder.nodes
        
        if len(current_nodes) < 3:
            return
            
        # 随机选择三个点
        indices = random.sample(range(len(current_nodes)), 3)
        
        if len(indices) == 3:
            pivot = current_nodes[indices[0]]
            point_a = current_nodes[indices[1]] 
            point_b = current_nodes[indices[2]]
        else:
            # 如果节点数不足3个，添加一些随机点
            additional_points = np.random.uniform(-1, 1, (3-len(indices), 2))
            builder.add_points(additional_points)
            current_nodes = builder.nodes
            
            pivot = current_nodes[0]
            point_a = current_nodes[1]
            point_b = current_nodes[2]
        
        # 计算旋转角度
        rotation_angle = calculate_rotation_angle_2d(point_a, point_b, pivot)
        
        # 如果旋转角度很小，则使用随机有理角度以避免无效操作
        if abs(rotation_angle) < 1e-6:
            rotation_angle = random.choice(RATIONAL_ANGLES)
        
        print(f"   Rotate-Merge: P={pivot}, A={point_a}, B={point_b}")
        print(f"   Rotation angle: {np.degrees(rotation_angle):.2f}°")
        
        # 复制当前图并旋转
        rotated_nodes = rotate_points_2d(current_nodes, rotation_angle, pivot)
        
        # 创建新的builder用于旋转后的图
        rotated_builder = copy.deepcopy(builder)
        rotated_builder.nodes = rotated_nodes
        rotated_builder.compute_edges()
        
        # 合并原图与旋转后的图
        builder.merge(rotated_builder)
        
        print(f"   After merge: {len(builder.nodes)} nodes, {builder.get_graph().number_of_edges()} edges")

    def _translation_mutation(self, individual: UDGBuilderWrapper):
        """
        平移变异：添加平移后的图副本
        """
        builder = individual.builder
        
        # 使用有理角度或随机角度生成平移向量
        if random.random() < 0.5:
            theta = random.choice(RATIONAL_ANGLES)
        else:
            theta = random.uniform(0, 2*np.pi)
        
        translation_vector = np.array([np.cos(theta), np.sin(theta)])
        
        # 添加平移后的点
        if len(builder.nodes) > 0:
            new_points = builder.nodes + translation_vector
            builder.add_points(new_points)
            builder.compute_edges()
        
        print(f"   Translation: vector={translation_vector}")

    def _pruning_mutation(self, individual: UDGBuilderWrapper):
        """
        剪枝变异：移除低度数节点
        """
        builder = individual.builder
        G = individual.graph_cache
        
        if G is None:
            G = builder.get_graph()
        
        degrees = dict(G.degree())
        if not degrees:
            return
        
        nodes_before = len(builder.nodes)
        
        avg_deg = sum(degrees.values()) / len(degrees)
        threshold = min(3, int(avg_deg * 0.8))
        
        to_keep_indices = [n for n, d in degrees.items() if d >= threshold]
        
        if len(to_keep_indices) > 0:
            coords = builder.nodes[to_keep_indices]
            builder.nodes = coords
            builder.compute_edges()
            
            print(f"   Pruned from {nodes_before} to {len(builder.nodes)} nodes (threshold: {threshold})")

    def _mutate(self, individual: UDGBuilderWrapper):
        """
        变异算子：主要使用旋转合并策略，辅以其他策略
        """
        current_nodes = len(individual.builder.nodes)
        
        # 策略选择概率
        if current_nodes > self.max_nodes:
            probs = [0.7, 0.2, 0.1]  # Rotate-Merge, Pruning, Translation
        elif current_nodes < 10:
            probs = [0.4, 0.3, 0.3]  # Rotate-Merge, Pruning, Translation
        else:
            probs = [0.8, 0.1, 0.1]  # 主要使用旋转合并
        
        choice = random.choices(
            ['rotate_merge', 'pruning', 'translation'], 
            weights=probs, 
            k=1
        )[0]
        
        if choice == 'rotate_merge':
            self._rotate_merge_mutation(individual)
        elif choice == 'pruning':
            self._pruning_mutation(individual)
        else:
            self._translation_mutation(individual)
        
        # 安全检查：防止节点数归零
        if len(individual.builder.nodes) == 0:
            individual.builder.add_moser_spindle()
        
        # 验证和评估
        G = individual.builder.get_graph()
        coloring = nx.greedy_color(G, strategy='largest_first')
        est_chromatic = max(coloring.values()) + 1
        print(f"   --> Validation: Greedy Coloring uses {est_chromatic} colors.")
        
        if est_chromatic > 4:
            print(f"   --> Checking 4-colorability with SAT-solver...")
            if not is_colorable(G, 4):
                print(f"   --> SAT-solver result: G is not 4-colorable!")
                individual.fitness = 100000
            else:
                print(f"   --> SAT-solver result: G is 4-colorable.")

    def step(self):
        """
        执行一代进化。
        """
        self.generation += 1
        
        # 排序
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        best_ind = self.population[0]
        self.best_fitness_history.append(best_ind.fitness)
        
        print(f"Gen {self.generation}: Best Fitness (Avg Deg) = {best_ind.fitness:.4f} | Nodes: {len(best_ind.builder.nodes)}")
        
        # 精英保留
        next_gen = []
        for i in range(min(self.elite_size, len(self.population))):
            next_gen.append(self.population[i].copy())
            
        # 繁殖与变异
        parents_pool = self.population[:self.pop_size // 2]
        
        while len(next_gen) < self.pop_size:
            parent = random.choice(parents_pool)
            child = parent.copy()
            
            if random.random() < self.mutation_rate:
                self._mutate(child)
            
            # 限制大小
            if len(child.builder.nodes) > self.max_nodes:
                child.builder.k_core_pruning(3)
                if len(child.builder.nodes) > self.max_nodes:
                    child.builder.remove_farthest_points(ratio=0.7)
            
            child.update_fitness()
            next_gen.append(child)
            
        self.population = next_gen
        
        # 定期清理和优化
        if self.generation % 10 == 0:
            for udg in self.population:
                udg.builder.k_core_pruning(3)
                if len(udg.builder.nodes) == 0:
                    udg.builder.add_moser_spindle()
                udg.update_fitness()
            
            # 过滤掉适应度为0的个体
            self.population = [udg for udg in self.population if udg.fitness > 0]

    def get_best_graph(self):
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        return self.population[0].builder.get_graph()

if __name__ == "__main__":
    from udg_builder import UDGBuilder

    # 配置GA
    ga = RotationMergeGeneticSearch2D(
        pop_size=15,
        max_nodes=1500,
        mutation_rate=0.9,
        elite_size=3
    )
    
    # 初始化
    ga.initialize_population(UDGBuilder)
    
    # 运行进化循环
    try:
        for i in range(200):
            os.system("rm tmp/*.cnf")
            ga.step()
            
            best_G = ga.get_best_graph()
            coloring = nx.greedy_color(best_G, strategy='largest_first')
            est_chromatic = max(coloring.values()) + 1
            print(f"   --> Validation: Greedy Coloring uses {est_chromatic} colors.")
            
            if est_chromatic >= 5:
                print(f"   --> Checking 4-colorability with SAT-solver...")
                if not is_colorable(best_G, 4):
                    print(f"   --> SAT-solver result: best_G is not 4-colorable!")
                    break
                else:
                    print(f"   --> SAT-solver result: best_G is 4-colorable.")
                
    except KeyboardInterrupt:
        print("Evolution stopped by user.")

    # 结果展示
    best_G = ga.get_best_graph()
    print(f"Final Best Graph: {best_G.number_of_nodes()} nodes, {best_G.number_of_edges()} edges.")
    
    # 可视化
    plt.figure(figsize=(12, 10))
    pos = nx.get_node_attributes(best_G, 'pos')
    if pos:
        pos_arr = np.array([pos[i] for i in range(len(pos))])
        plt.scatter(pos_arr[:,0], pos_arr[:,1], s=20, c='blue', alpha=0.7)
        
        if best_G.number_of_edges() < 5000:
            nx.draw_networkx_edges(best_G, pos, alpha=0.3, edge_color='red', width=0.5)
    
    plt.title(f"Evolved 2D UDG with Rotation-Merge (Avg Deg: {2*best_G.number_of_edges()/best_G.number_of_nodes():.2f})")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.savefig('ga_2d_rotation_merge_result.png', dpi=150, bbox_inches='tight')
    plt.show()