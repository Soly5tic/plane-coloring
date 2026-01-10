import numpy as np
import networkx as nx
import random
import copy
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from z3 import Solver, Bool, Or, And, Not
import torch
import os

from udg3d_builder import UDG3DBuilder 
from graph_coloring import is_colorable
from gnn_model import load_model, predict_graph

class UDG3DBuilderWrapper:
    """
    对 UDG3DBuilder 的封装，增加了适合 GA 的深拷贝和特定变异逻辑。
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
        return UDG3DBuilderWrapper(new_builder, self.model, self.device)

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

def k_core_prune_3d(G: nx.Graph, k: int) -> nx.Graph:
    """
    三维版本的k-core剪枝，不断删除度数 <= k 的节点。
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

def get_rational_axes_3d() -> List[Tuple[float, float, float]]:
    """
    三维"有理轴"库。
    """
    axes = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
        (1, -1, 0),
        (1, 0, -1),
        (0, 1, -1),
        (1, -1, 1),
        (1, 1, -1),
        (2, 1, 0),
        (2, 0, 1),
        (1, 2, 0),
        (0, 2, 1),
    ]
    uniq = []
    seen = set()
    for a, b, c in axes:
        if (a, b, c) == (0, 0, 0):
            continue
        if (a, b, c) not in seen:
            seen.add((a, b, c))
            uniq.append((a, b, c))
    
    normalized = []
    for a, b, c in uniq:
        norm = np.sqrt(a*a + b*b + c*c)
        normalized.append((a/norm, b/norm, c/norm))
    return normalized

def get_rational_angles_3d() -> List[float]:
    """
    三维用的离散角度集合。
    """
    angles = set()

    base_angles = [
        np.pi / 2,
        np.pi / 3,
        2 * np.pi / 3,
        np.pi / 4,
        3 * np.pi / 4
    ]
    for a in base_angles:
        angles.add(a)
        angles.add(-a)

    for a in range(2, 5):
        for b in range(1, a):
            val = b / a
            try:
                angles.add(np.arccos(val))
                angles.add(-np.arccos(val))
                angles.add(np.arcsin(val))
                angles.add(-np.arcsin(val))
            except ValueError:
                pass
        for b in range(2, min(a * a, 6)):
            val = np.sqrt(b) / a
            if val <= 1:
                try:
                    angles.add(np.arccos(val))
                    angles.add(-np.arccos(val))
                    angles.add(np.arcsin(val))
                    angles.add(-np.arcsin(val))
                except ValueError:
                    pass

    return list(angles)

def calculate_rotation_axis_and_angle(point_a: np.ndarray, point_b: np.ndarray, pivot: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    计算将点B绕原点P旋转到点A位置所需的旋转轴和角度。
    
    Args:
        point_a: 目标位置A
        point_b: 需要旋转的点B
        pivot: 旋转中心P
        
    Returns:
        (旋转轴, 旋转角度)
    """
    # 将坐标转换为以pivot为原点的相对坐标
    a_rel = point_a - pivot
    b_rel = point_b - pivot
    
    # 计算从b_rel到a_rel的旋转
    # 使用四元数方法计算旋转
    
    # 归一化向量
    a_norm = np.linalg.norm(a_rel)
    b_norm = np.linalg.norm(b_rel)
    
    if a_norm < 1e-10 or b_norm < 1e-10:
        # 如果有零向量，返回默认旋转
        return np.array([0, 0, 1]), np.pi / 2
    
    a_unit = a_rel / a_norm
    b_unit = b_rel / b_norm
    
    # 计算旋转轴（叉积）
    rotation_axis = np.cross(b_unit, a_unit)
    rotation_axis_norm = np.linalg.norm(rotation_axis)
    
    if rotation_axis_norm < 1e-10:
        # 如果向量平行或反平行
        if np.dot(b_unit, a_unit) > 0:
            # 同方向，不需要旋转
            return np.array([0, 0, 1]), 0.0
        else:
            # 反方向，绕任意轴旋转180度
            return np.array([1, 0, 0]), np.pi
    
    rotation_axis = rotation_axis / rotation_axis_norm
    
    # 计算旋转角度
    cos_angle = np.clip(np.dot(b_unit, a_unit), -1.0, 1.0)
    rotation_angle = np.arccos(cos_angle)
    
    return rotation_axis, rotation_angle

def rotate_points_around_axis(points: np.ndarray, rotation_axis: np.ndarray, angle: float, pivot: np.ndarray) -> np.ndarray:
    """
    绕指定轴和角度旋转点集。
    
    Args:
        points: 需要旋转的点集 (N, 3)
        rotation_axis: 旋转轴
        angle: 旋转角度
        pivot: 旋转中心
        
    Returns:
        旋转后的点集
    """
    # 罗德里格旋转公式
    axis = rotation_axis / np.linalg.norm(rotation_axis)
    
    # 构建旋转矩阵
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    # 旋转矩阵的各个分量
    R = np.zeros((3, 3))
    R[0, 0] = cos_angle + axis[0]**2 * (1 - cos_angle)
    R[0, 1] = axis[0] * axis[1] * (1 - cos_angle) - axis[2] * sin_angle
    R[0, 2] = axis[0] * axis[2] * (1 - cos_angle) + axis[1] * sin_angle
    
    R[1, 0] = axis[1] * axis[0] * (1 - cos_angle) + axis[2] * sin_angle
    R[1, 1] = cos_angle + axis[1]**2 * (1 - cos_angle)
    R[1, 2] = axis[1] * axis[2] * (1 - cos_angle) - axis[0] * sin_angle
    
    R[2, 0] = axis[2] * axis[0] * (1 - cos_angle) - axis[1] * sin_angle
    R[2, 1] = axis[2] * axis[1] * (1 - cos_angle) + axis[0] * sin_angle
    R[2, 2] = cos_angle + axis[2]**2 * (1 - cos_angle)
    
    # 平移到以pivot为原点，旋转，然后平移回去
    centered_points = points - pivot
    rotated_points = (R @ centered_points.T).T + pivot
    
    return rotated_points

RATIONAL_AXES_3D: List[Tuple[int, int, int]] = get_rational_axes_3d()
RATIONAL_ANGLES_3D: List[float] = get_rational_angles_3d()

class RotationMergeGeneticSearch3D:
    def __init__(self, 
                 pop_size=20, 
                 max_nodes=2000, 
                 mutation_rate=0.8, 
                 elite_size=3, 
                 model_path="./best_4color_model.pth"):
        """
        使用旋转合并策略的进化算法。
        """
        self.pop_size = pop_size
        self.max_nodes = max_nodes
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.population: List[UDG3DBuilderWrapper] = []
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
            
            # 添加初始结构
            initial_points = np.array([
                [0, 0, 0],
                [1, 0, 0],
                [0.5, np.sqrt(3)/2, 0],
                [0.5, np.sqrt(3)/6, np.sqrt(6)/3]
            ])
            
            builder.add_points(initial_points)
            
            wrapper = UDG3DBuilderWrapper(builder, self.model, self.device)
            wrapper.update_fitness()
            self.population.append(wrapper)

    def _rotate_merge_mutation(self, individual: UDG3DBuilderWrapper):
        """
        核心变异策略：旋转合并
        
        1. 从当前图中随机选择三个点：原点P和两个点A, B
        2. 计算将B绕P旋转到A位置的旋转轴和角度
        3. 对整个图进行旋转
        4. 合并原图与旋转后的图
        """
        builder = individual.builder
        current_nodes = builder.nodes
        
        if len(current_nodes) < 3:
            return
            
        # 随机选择三个点
        indices = random.sample(range(len(current_nodes)), min(3, len(current_nodes)))
        
        if len(indices) == 3:
            # 选择原点P和两个点A, B
            pivot_idx = indices[0]
            point_a_idx = indices[1] 
            point_b_idx = indices[2]
            
            pivot = current_nodes[pivot_idx]
            point_a = current_nodes[point_a_idx]
            point_b = current_nodes[point_b_idx]
        else:
            # 如果节点数不足3个，添加一些随机点
            additional_points = np.random.uniform(-1, 1, (3-len(indices), 3))
            builder.add_points(additional_points)
            current_nodes = builder.nodes
            
            pivot_idx = 0
            point_a_idx = 1
            point_b_idx = 2
            
            pivot = current_nodes[pivot_idx]
            point_a = current_nodes[point_a_idx]
            point_b = current_nodes[point_b_idx]
        
        # 计算旋转参数
        rotation_axis, rotation_angle = calculate_rotation_axis_and_angle(
            point_a, point_b, pivot
        )
        
        # 如果旋转角度很小，则添加一些噪声以避免无效操作
        if rotation_angle < 1e-6:
            # 随机选择一个有理角度
            rotation_angle = random.choice(RATIONAL_ANGLES_3D)
            # 随机选择一个有理轴
            axis_tuple = random.choice(RATIONAL_AXES_3D)
            rotation_axis = np.array(axis_tuple)
        
        print(f"   Rotate-Merge: P={pivot}, A={point_a}, B={point_b}")
        print(f"   Rotation axis: {rotation_axis}, angle: {rotation_angle:.4f} rad ({np.degrees(rotation_angle):.2f}°)")
        
        # 复制当前图并旋转
        rotated_nodes = rotate_points_around_axis(
            current_nodes, rotation_axis, rotation_angle, pivot
        )
        
        # 创建新的builder用于旋转后的图
        rotated_builder = copy.deepcopy(builder)
        rotated_builder.nodes = rotated_nodes
        rotated_builder.compute_edges()
        
        # 合并原图与旋转后的图
        builder.merge(rotated_builder)
        
        print(f"   After merge: {len(builder.nodes)} nodes, {builder.get_graph().number_of_edges()} edges")

    def _selective_addition_mutation(self, individual: UDG3DBuilderWrapper):
        """
        基于边方向的Minkowski变异：从图中随机选择一条边，以该边的方向作为平移方向
        """
        builder = individual.builder
        G = individual.graph_cache
        
        if G is None:
            G = builder.get_graph()
        
        # 如果图中没有边，回退到随机加点
        if G.number_of_edges() == 0:
            num_new_points = random.randint(5, 20)
            new_points = np.random.uniform(-2, 2, (num_new_points, 3))
            builder.add_points(new_points)
            print(f"   No edges found, added {num_new_points} random points")
            return
        
        # 随机选择一条边
        edges = list(G.edges())
        if not edges:
            num_new_points = random.randint(5, 20)
            new_points = np.random.uniform(-2, 2, (num_new_points, 3))
            builder.add_points(new_points)
            print(f"   No edges found, added {num_new_points} random points")
            return
        
        edge = random.choice(edges)
        node1, node2 = edge
        
        # 获取节点位置
        pos = nx.get_node_attributes(G, 'pos')
        if node1 not in pos or node2 not in pos:
            num_new_points = random.randint(5, 20)
            new_points = np.random.uniform(-2, 2, (num_new_points, 3))
            builder.add_points(new_points)
            print(f"   Node positions not found, added {num_new_points} random points")
            return
        
        point1 = np.array(pos[node1])
        point2 = np.array(pos[node2])
        
        # 计算边的方向向量并归一化
        edge_direction = point2 - point1
        edge_direction_norm = np.linalg.norm(edge_direction)
        
        if edge_direction_norm < 1e-10:
            # 如果边长度为0，使用随机方向
            if random.random() < 0.5:
                theta = random.choice(RATIONAL_ANGLES_3D)
                phi = random.uniform(0, 2*np.pi)
            else:
                theta = random.uniform(0, np.pi)
                phi = random.uniform(0, 2*np.pi)
            
            # 球坐标系转换为笛卡尔坐标
            translation_vector = np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ])
        else:
            # 使用边的方向作为平移方向
            edge_direction_unit = edge_direction / edge_direction_norm
            
            # 随机选择平移距离
            translation_distance = random.uniform(0.1, 2.0)
            translation_vector = edge_direction_unit * translation_distance
        
        print(f"   Minkowski mutation: edge ({node1}, {node2}), direction: {translation_vector}")
        
        # 对现有所有点进行平移，然后添加回图中
        if len(builder.nodes) > 0:
            new_points = builder.nodes + translation_vector
            builder.add_points(new_points)
            
            print(f"   Added {len(builder.nodes)} translated points in edge direction")

    def _pruning_mutation(self, individual: UDG3DBuilderWrapper):
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
        threshold = min(4, int(avg_deg * 0.8))
        
        to_keep_indices = [n for n, d in degrees.items() if d >= threshold]
        
        if len(to_keep_indices) > 0:
            coords = builder.nodes[to_keep_indices]
            builder.nodes = coords
            builder.compute_edges()
            
            print(f"   Pruned from {nodes_before} to {len(builder.nodes)} nodes (threshold: {threshold})")

    def _mutate(self, individual: UDG3DBuilderWrapper):
        """
        变异算子：主要使用旋转合并策略，辅以其他策略
        """
        current_nodes = len(individual.builder.nodes)
        
        # 策略选择概率
        if current_nodes > self.max_nodes:
            probs = [0.7, 0.2, 0.1]  # Rotate-Merge, Pruning, Selective Addition
        elif current_nodes < 10:
            probs = [0.4, 0.3, 0.3]  # Rotate-Merge, Pruning, Selective Addition
        else:
            probs = [0.8, 0.1, 0.1]  # 主要使用旋转合并
        
        choice = random.choices(
            ['rotate_merge', 'pruning', 'selective_addition'], 
            weights=probs, 
            k=1
        )[0]
        
        if choice == 'rotate_merge':
            self._rotate_merge_mutation(individual)
        elif choice == 'pruning':
            self._pruning_mutation(individual)
        else:
            self._selective_addition_mutation(individual)
        
        # 安全检查：防止节点数归零
        if len(individual.builder.nodes) == 0:
            initial_points = np.array([
                [0, 0, 0],
                [1, 0, 0],
                [0.5, np.sqrt(3)/2, 0],
                [0.5, np.sqrt(3)/6, np.sqrt(6)/3]
            ])
            individual.builder.add_points(initial_points)
        
        # 验证和评估
        G = individual.builder.get_graph()
        coloring = nx.greedy_color(G, strategy='largest_first')
        est_chromatic = max(coloring.values()) + 1
        print(f"   --> Validation: Greedy Coloring uses {est_chromatic} colors.")
        
        if est_chromatic >= 6:
            print(f"   --> Checking 5-colorability with SAT-solver...")
            if not is_colorable(G, 5):
                print(f"   --> SAT-solver result: G is not 5-colorable!")
                individual.fitness = 100000
            else:
                print(f"   --> SAT-solver result: G is 5-colorable.")

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
                G = child.builder.get_graph()
                pruned_G = k_core_prune_3d(G, 4)
                if pruned_G.number_of_nodes() > 0:
                    positions = [pruned_G.nodes[node]['pos'] for node in pruned_G.nodes()]
                    child.builder.nodes = np.array(positions)
                    child.builder.compute_edges()
                    if len(child.builder.nodes) > self.max_nodes:
                        child.builder.remove_farthest_points(ratio=0.5)
                else:
                    child.builder.nodes = []
                    child.builder.edges = []
                    child.builder.graph = nx.Graph()
                    initial_points = np.array([
                        [0, 0, 0],
                        [1, 0, 0],
                        [0.5, np.sqrt(3)/2, 0],
                        [0.5, np.sqrt(3)/6, np.sqrt(6)/3]
                    ])
                    child.builder.add_points(initial_points)
            
            child.update_fitness()
            next_gen.append(child)
            
        self.population = next_gen
        
        # 定期清理和优化
        if self.generation % 10 == 0:
            for udg in self.population:
                G = udg.builder.get_graph()
                pruned_G = k_core_prune_3d(G, 4)
                if pruned_G.number_of_nodes() > 0:
                    positions = [pruned_G.nodes[node]['pos'] for node in pruned_G.nodes()]
                    udg.builder.nodes = np.array(positions)
                    udg.builder.compute_edges()
                    udg.builder.remove_farthest_points(ratio=0.2)
                if len(udg.builder.nodes) == 0:
                    initial_points = np.array([
                        [0, 0, 0],
                        [1, 0, 0],
                        [0.5, np.sqrt(3)/2, 0],
                        [0.5, np.sqrt(3)/6, np.sqrt(6)/3]
                    ])
                    udg.builder.add_points(initial_points)
                udg.update_fitness()
            
            # 过滤掉适应度为0的个体
            self.population = [udg for udg in self.population if udg.fitness > 0]

    def get_best_graph(self):
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        return self.population[0].builder.get_graph()

if __name__ == "__main__":
    from udg3d_builder import UDG3DBuilder

    # 配置GA
    ga = RotationMergeGeneticSearch3D(
        pop_size=15,
        max_nodes=1500,
        mutation_rate=0.9,
        elite_size=3
    )
    
    # 初始化
    ga.initialize_population(UDG3DBuilder)
    
    # 运行进化循环
    try:
        for i in range(300):
            os.system("rm tmp/*.cnf")
            ga.step()
            
            best_G = ga.get_best_graph()
            coloring = nx.greedy_color(best_G, strategy='largest_first')
            est_chromatic = max(coloring.values()) + 1
            print(f"   --> Validation: Greedy Coloring uses {est_chromatic} colors.")
            
            if est_chromatic >= 6:
                print(f"   --> Checking 5-colorability with SAT-solver...")
                if not is_colorable(best_G, 5):
                    print(f"   --> SAT-solver result: best_G is not 5-colorable!")
                    break
                else:
                    print(f"   --> SAT-solver result: best_G is 5-colorable.")
                
    except KeyboardInterrupt:
        print("Evolution stopped by user.")

    # 结果展示
    best_G = ga.get_best_graph()
    print(f"Final Best Graph: {best_G.number_of_nodes()} nodes, {best_G.number_of_edges()} edges.")
    
    # 可视化
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    pos = nx.get_node_attributes(best_G, 'pos')
    if pos:
        pos_arr = np.array([pos[i] for i in range(len(pos))])
        ax.scatter(pos_arr[:,0], pos_arr[:,1], pos_arr[:,2], s=20, c='blue', alpha=0.6)
        
        if best_G.number_of_edges() < 2000:
            for edge in list(best_G.edges())[:1000]:
                node1, node2 = edge
                if node1 in pos and node2 in pos:
                    x_coords = [pos[node1][0], pos[node2][0]]
                    y_coords = [pos[node1][1], pos[node2][1]]
                    z_coords = [pos[node1][2], pos[node2][2]]
                    ax.plot(x_coords, y_coords, z_coords, 'r-', alpha=0.8, linewidth=0.5)
    
    ax.set_title(f"Evolved 3D UDG with Rotation-Merge (Avg Deg: {2*best_G.number_of_edges()/best_G.number_of_nodes():.2f})")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig("best_3d_udg.png")