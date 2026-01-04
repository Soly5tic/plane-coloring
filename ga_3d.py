import numpy as np
import networkx as nx
import random
import copy
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from z3 import Solver, Bool, Or, And, Not
import torch
import os

# 假设之前的代码保存在 udg_builder.py 中
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
        self.graph_cache = None # 缓存 NetworkX 对象
        self.model = model
        self.device = device

    def copy(self):
        """深拷贝当前个体，用于产生后代"""
        new_builder = copy.deepcopy(self.builder)
        return UDG3DBuilderWrapper(new_builder, self.model, self.device)

    def update_fitness(self):
        """
        计算适应度：使用平均度数作为适应性分数。
        对于三维情况，我们寻找6色和更高的图，所以适应性分数越高越好。
        """
        # 只有在变异后才重新计算
        self.builder.clean_isolated_nodes() # 总是先清理孤立点
        G = self.builder.get_graph()
        self.graph_cache = G
        
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        
        if num_nodes == 0:
            self.fitness = 0.0
        else:
            # 使用平均度数作为适应性分数
            self.fitness = 2.0 * num_edges / num_nodes
        
        return self.fitness

def k_core_prune_3d(G: nx.Graph, k: int) -> nx.Graph:
    """
    三维版本的k-core剪枝，不断删除度数 <= k 的节点。
    """
    # 深拷贝，避免修改原图
    H = G.copy()
    changed = True
    while changed and H.number_of_nodes() > 0:
        changed = False
        # 找出所有度数 <= k 的节点
        to_remove = [n for n, deg in H.degree() if deg <= k]
        if to_remove:
            H.remove_nodes_from(to_remove)
            changed = True
    return H

def get_rational_axes_3d() -> List[Tuple[float, float, float]]:
    """
    三维“有理轴”库。
    这里返回的是若干整数向量，真正用时会归一化成单位向量。
    可以根据需要扩充/剪裁。
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
    # 去重并去掉零向量
    uniq = []
    seen = set()
    for a, b, c in axes:
        if (a, b, c) == (0, 0, 0):
            continue
        if (a, b, c) not in seen:
            seen.add((a, b, c))
            uniq.append((a, b, c))
    
    # 对 uniq 中的所有三维向量归一化
    normalized = []
    for a, b, c in uniq:
        norm = np.sqrt(a*a + b*b + c*c)
        normalized.append((a/norm, b/norm, c/norm))
    return normalized

def get_rational_angles_3d() -> List[float]:
    """
    三维用的离散角度集合。
    思路与 2D 中 get_rational_angles 类似：枚举一些 cos/sin 有简单表达的角度。[web:34]
    这里可以直接复用那套逻辑，或者先用一小批“手选角度”。
    """
    angles = set()

    # 1) 一批经典“规则多面体对称”角度
    base_angles = [
        np.pi / 2,     # 90°
        np.pi / 3,     # 60°
        2 * np.pi / 3, # 120°
        np.pi / 4,     # 45°
        3 * np.pi / 4  # 135°
    ]
    for a in base_angles:
        angles.add(a)
        angles.add(-a)

    # 2) 类似 2D 的“有理 cos/sin” 枚举（可选）
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

RATIONAL_AXES_3D: List[Tuple[int, int, int]] = get_rational_axes_3d()
RATIONAL_ANGLES_3D: List[float] = get_rational_angles_3d()

class GeneticUDGSearch3D:
    def __init__(self, 
                 pop_size=20, 
                 max_nodes=2000, 
                 mutation_rate=0.8, 
                 elite_size=3, 
                 model_path="./best_4color_model.pth"):
        """
        Args:
            pop_size: 种群大小
            max_nodes: 为了防止内存爆炸，限制最大节点数
            mutation_rate: 变异概率
            elite_size: 每代保留的最优个体数
            model_path: GNN模型路径
        """
        self.pop_size = pop_size
        self.max_nodes = max_nodes
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.population: List[UDG3DBuilderWrapper] = []
        self.generation = 0
        
        # 初始化数据记录
        self.best_fitness_history = []
        
        # GNN模型相关
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
        每个个体都从一个标准的 Moser Spindle 开始，但带有随机的初始旋转。
        """
        self.population = []
        print(f"Initializing population of size {self.pop_size}...")
        
        for _ in range(self.pop_size):
            # 创建基础构建器
            builder = base_builder_cls(tolerance=1e-5)
            
            # 初始种子：Moser Spindle
            # 为了增加多样性，给每个初始个体一个随机的整体旋转
            initial_axis = random.choice(RATIONAL_AXES_3D)
            initial_angle = random.choice(RATIONAL_ANGLES_3D)
            initial_pivot = (0, 0, 0)
            
            # 添加初始结构（这里需要根据实际的三维基础结构来调整）
            # 暂时使用简单的初始点集
            initial_points = np.array([
                [0, 0, 0],
                [1, 0, 0],
                [0.5, np.sqrt(3)/2, 0],
                [0.5, np.sqrt(3)/6, np.sqrt(6)/3]
            ])
            
            builder.add_points(initial_points)
            builder.rotate(initial_axis, initial_angle, initial_pivot)
            
            wrapper = UDG3DBuilderWrapper(builder, self.model, self.device)
            wrapper.update_fitness()
            self.population.append(wrapper)

    def _select_axis_angle_pivot(self):
        """
        选择三维旋转参数：高概率从有理轴和角度中选择，低概率随机选择。
        """
        # 以高概率从有理轴中选择
        if random.random() < 0.9:
            axis = random.choice(RATIONAL_AXES_3D)
        else:
            # 随机生成一个轴
            axis = (random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))
            # 归一化
            norm = np.sqrt(sum(x*x for x in axis))
            if norm > 0:
                axis = tuple(x/norm for x in axis)
        
        # 以高概率从有理角度中选择
        if random.random() < 0.9:
            angle = random.choice(RATIONAL_ANGLES_3D)
        else:
            angle = random.uniform(0.01, np.pi)
        
        # pivot的选取策略与二维情况相同
        if random.random() < 0.3 and len(self.current_builder_nodes) > 0:
            pivot_idx = random.randint(0, len(self.current_builder_nodes)-1)
            pivot = tuple(self.current_builder_nodes[pivot_idx])
        else:
            pivot = (0, 0, 0)
        
        return axis, angle, pivot

    def _mutate(self, individual1: UDG3DBuilderWrapper, individual2: UDG3DBuilderWrapper):
        """
        变异算子核心逻辑 - 三维版本。
        """
        builder1 = individual1.builder
        builder2 = individual2.builder
        current_nodes = len(builder1.nodes)
        
        # 策略选择概率
        if current_nodes > self.max_nodes:
            probs = [0, 0, 1] # Rotate, Minkowski, Prune
        else:
            probs = [0.6, 0.3, 0.1] # Rotate, Minkowski, Prune
            
        choice = random.choices(['rotate_merge', 'minkowski', 'prune'], weights=probs, k=1)[0]
        
        if choice == 'rotate_merge':
            # --- 变异 1: Rotate & Merge (三维版本) ---
            # 将第二个 UDG3DBuilder 绕随机轴旋转随机角度与第一个合并
            
            # 高概率从有理轴和角度中选择，低概率任意随机选择
            if random.random() < 0.9:
                axis = random.choice(RATIONAL_AXES_3D)
            else:
                axis = (random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))
                norm = np.sqrt(sum(x*x for x in axis))
                if norm > 0:
                    axis = tuple(x/norm for x in axis)
            
            if random.random() < 0.9:
                angle = random.choice(RATIONAL_ANGLES_3D)
            else:
                angle = random.uniform(0.01, np.pi)
                
            # 随机选择旋转中心 (原点或随机现有点)
            if len(builder2.nodes) > 0 and random.random() < 0.3:
                pivot_idx = random.randint(0, len(builder2.nodes)-1)
                pivot = tuple(builder2.nodes[pivot_idx])
            else:
                pivot = (0, 0, 0)
                
            # 旋转第二个 UDG3DBuilder
            temp_builder = copy.deepcopy(builder2)
            temp_builder.rotate(axis, angle, pivot)
            
            # 合并到第一个 UDG3DBuilder
            builder1.merge(temp_builder)
            
        elif choice == 'minkowski':
            # --- 变异 2: Minkowski Sum (三维版本) ---
            # 仅对第一个 UDG3DBuilder 进行操作
            
            # 生成三维随机平移向量
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
            
            # 获取当前所有点，平移，然后添加回图中
            if len(builder1.nodes) > 0:
                new_points = builder1.nodes + translation_vector
                builder1.add_points(new_points)
                
        elif choice == 'prune':
            # --- 变异 3: Pruning (三维版本) ---
            # 仅对第一个 UDG3DBuilder 进行操作
            G = individual1.graph_cache
            if G is None: 
                G = builder1.get_graph()
            
            # 计算度数
            degrees = dict(G.degree())
            if not degrees: return
            
            nodes_before = len(builder1.nodes)
            
            avg_deg = sum(degrees.values()) / len(degrees)
            # 移除度数低于平均值的点（或者低于固定阈值如 2 或 3）
            threshold = min(4, int(avg_deg * 0.8))  # 三维情况可能需要更高的阈值
            
            to_keep_indices = [n for n, d in degrees.items() if d >= threshold]
            
            if len(to_keep_indices) > 0:
                # 更新 builder 的 nodes
                coords = builder1.nodes[to_keep_indices]
                builder1.nodes = coords # Numpy array slicing
                builder1.compute_edges() # 重新计算边
            
            if len(builder1.nodes) >= nodes_before:
                builder1.remove_farthest_points(ratio=0.5)
            
        # 安全检查：防止节点数归零
        if len(builder1.nodes) == 0:
            # 重新添加初始结构
            initial_points = np.array([
                [0, 0, 0],
                [1, 0, 0],
                [0.5, np.sqrt(3)/2, 0],
                [0.5, np.sqrt(3)/6, np.sqrt(6)/3]
            ])
            builder1.add_points(initial_points)
        
        # 验证和评估
        G = builder1.get_graph()
        coloring = nx.greedy_color(G, strategy='largest_first')
        est_chromatic = max(coloring.values()) + 1
        print(f"   --> Validation: Greedy Coloring uses {est_chromatic} colors.")
        
        # 三维情况寻找6色和更高的图
        if est_chromatic >= 6:
            print(f"   --> Checking 5-colorability with SAT-solver...")
            if not is_colorable(G, 5):
                print(f"   --> SAT-solver result: G is not 5-colorable!")
                individual1.fitness = 100000
            else:
                print(f"   --> SAT-solver result: G is 5-colorable.")
        
    def step(self):
        """
        执行一代进化。
        """
        self.generation += 1
        
        # 1. 排序 (根据 Fitness 降序)
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        best_ind = self.population[0]
        self.best_fitness_history.append(best_ind.fitness)
        
        print(f"Gen {self.generation}: Best Fitness (Avg Deg) = {best_ind.fitness:.4f} | Nodes: {len(best_ind.builder.nodes)}")
        
        # 2. 精英保留 (Elitism)
        next_gen = []
        for i in range(self.elite_size):
            # 直接保留，深拷贝以防意外修改
            next_gen.append(self.population[i].copy())
            
        # 3. 繁殖与变异
        # 简单的轮盘赌或锦标赛选择，这里用简单的 Top-K 随机选择
        parents_pool = self.population[:self.pop_size // 2] # 选前 50% 做父母
        
        while len(next_gen) < self.pop_size:
            # 选择父代
            parent = random.choice(parents_pool)
            # 复制产生子代
            child = parent.copy()
            
            # 变异
            if random.random() < self.mutation_rate:
                # 从 parent 列表中随机选择第二个 UDG3DBuilder
                second_parent = random.choice(parents_pool)
                self._mutate(child, second_parent)
            
            # 限制大小 (硬约束)
            if len(child.builder.nodes) > self.max_nodes:
                # 强制修剪
                G = child.builder.get_graph()
                pruned_G = k_core_prune_3d(G, 4)  # 三维情况使用更高的k值
                if pruned_G.number_of_nodes() > 0:
                    # 更新builder
                    positions = [pruned_G.nodes[node]['pos'] for node in pruned_G.nodes()]
                    child.builder.nodes = np.array(positions)
                    child.builder.compute_edges()
                    # 额外使用 remove_farthest_points 进行进一步修剪
                    if len(child.builder.nodes) > self.max_nodes:
                        child.builder.remove_farthest_points(ratio=0.5)
                else:
                    # 如果修剪后为空，重新添加初始结构
                    initial_points = np.array([
                        [0, 0, 0],
                        [1, 0, 0],
                        [0.5, np.sqrt(3)/2, 0],
                        [0.5, np.sqrt(3)/6, np.sqrt(6)/3]
                    ])
                    child.builder.add_points(initial_points)
            
            # 计算子代适应度
            child.update_fitness()
            next_gen.append(child)
            
        self.population = next_gen
        
        # 定期清理和优化
        if self.generation % 8 == 0:
            for udg in self.population:
                G = udg.builder.get_graph()
                pruned_G = k_core_prune_3d(G, 4)
                if pruned_G.number_of_nodes() > 0:
                    positions = [pruned_G.nodes[node]['pos'] for node in pruned_G.nodes()]
                    udg.builder.nodes = np.array(positions)
                    udg.builder.compute_edges()
                    # 使用 remove_farthest_points 进行进一步优化
                    udg.builder.remove_farthest_points(ratio=0.2)
                if len(udg.builder.nodes) == 0:
                    # 重新添加初始结构
                    initial_points = np.array([
                        [0, 0, 0],
                        [1, 0, 0],
                        [0.5, np.sqrt(3)/2, 0],
                        [0.5, np.sqrt(3)/6, np.sqrt(6)/3]
                    ])
                    udg.builder.add_points(initial_points)
                udg.update_fitness()
            
            # 过滤掉适应度为0的个体
            next_gen = [udg for udg in next_gen if udg.fitness > 0]
            self.population = next_gen

    def get_best_graph(self):
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        return self.population[0].builder.get_graph()

if __name__ == "__main__":
    from udg3d_builder import UDG3DBuilder # 导入你保存的类

    # 1. 配置 GA
    ga = GeneticUDGSearch3D(
        pop_size=20,
        max_nodes=1500,  # 限制图规模，防止变慢
        mutation_rate=0.9, # 高变异率，因为探索空间很大
        elite_size=2
    )
    
    # 2. 初始化
    ga.initialize_population(UDG3DBuilder)
    
    # 3. 运行进化循环
    try:
        for i in range(500): # 运行 20 代试试
            os.system("rm tmp/*.cnf")
            ga.step()
            
            best_G = ga.get_best_graph()

            print(best_G)
            # 使用贪心算法快速估算色数
            
            coloring = nx.greedy_color(best_G, strategy='largest_first')
            est_chromatic = max(coloring.values()) + 1
            print(f"   --> Validation: Greedy Coloring uses {est_chromatic} colors.")
            
            if est_chromatic >= 5:  # 三维情况寻找5色和更高的图
                print(f"   --> Checking 5-colorability with SAT-solver...")
                if not is_colorable(best_G, 5):
                    print(f"   --> SAT-solver result: best_G is not 5-colorable!")
                    break
                else:
                    print(f"   --> SAT-solver result: best_G is 5-colorable.")
                
    except KeyboardInterrupt:
        print("Evolution stopped by user.")

    # 4. 结果展示
    best_G = ga.get_best_graph()
    print(f"Final Best Graph: {best_G.number_of_nodes()} nodes, {best_G.number_of_edges()} edges.")
    
    # 可视化（需要安装mayavi或使用matplotlib的3D功能）
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    pos = nx.get_node_attributes(best_G, 'pos')
    if pos:
        pos_arr = np.array([pos[i] for i in range(len(pos))])
        ax.scatter(pos_arr[:,0], pos_arr[:,1], pos_arr[:,2], s=20, c='red', alpha=0.6)
        
        # 绘制部分边（太多边会画不开）
        if best_G.number_of_edges() < 2000:
            for edge in list(best_G.edges())[:1000]:  # 限制边的数量
                node1, node2 = edge
                if node1 in pos and node2 in pos:
                    x_coords = [pos[node1][0], pos[node2][0]]
                    y_coords = [pos[node1][1], pos[node2][1]]
                    z_coords = [pos[node1][2], pos[node2][2]]
                    ax.plot(x_coords, y_coords, z_coords, 'b-', alpha=0.1, linewidth=0.5)
    
    ax.set_title(f"Evolved 3D UDG (Avg Deg: {2*best_G.number_of_edges()/best_G.number_of_nodes():.2f})")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()