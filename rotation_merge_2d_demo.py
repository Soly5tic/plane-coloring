import numpy as np
import networkx as nx
import random
import copy
import matplotlib.pyplot as plt
from typing import List
import os

from udg_builder import UDGBuilder 

class UDGBuilderWrapper:
    """对 UDGBuilder 的封装，增加了适合 GA 的深拷贝和特定变异逻辑。"""
    def __init__(self, builder):
        self.builder = builder
        self.fitness = 0.0
        self.graph_cache = None

    def copy(self):
        """深拷贝当前个体，用于产生后代"""
        new_builder = copy.deepcopy(self.builder)
        return UDGBuilderWrapper(new_builder)

    def update_fitness(self):
        """计算适应度：使用平均度数作为适应性分数。"""
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
    """生成二维有理角度库。"""
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

def calculate_rotation_angle_2d(point_a: np.ndarray, point_b: np.ndarray, pivot: np.ndarray) -> float:
    """
    计算将点B绕原点P旋转到点A位置所需的旋转角度（2D版本）。
    """
    a_rel = point_a - pivot
    b_rel = point_b - pivot
    
    angle_a = np.arctan2(a_rel[1], a_rel[0])
    angle_b = np.arctan2(b_rel[1], b_rel[0])
    
    rotation_angle = angle_a - angle_b
    
    while rotation_angle > np.pi:
        rotation_angle -= 2 * np.pi
    while rotation_angle < -np.pi:
        rotation_angle += 2 * np.pi
    
    return rotation_angle

def rotate_points_2d(points: np.ndarray, angle: float, pivot: np.ndarray) -> np.ndarray:
    """绕指定角度旋转点集（2D版本）。"""
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    R = np.array([
        [cos_angle, -sin_angle],
        [sin_angle, cos_angle]
    ])
    
    centered_points = points - pivot
    rotated_points = (R @ centered_points.T).T + pivot
    
    return rotated_points

RATIONAL_ANGLES = get_rational_angles()

class RotationMergeGeneticSearch2D:
    def __init__(self, pop_size=10, max_nodes=500, mutation_rate=0.9, elite_size=2):
        """使用旋转合并策略的2D进化算法。"""
        self.pop_size = pop_size
        self.max_nodes = max_nodes
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.population: List[UDGBuilderWrapper] = []
        self.generation = 0
        
        self.best_fitness_history = []

    def initialize_population(self, base_builder_cls):
        """初始化种群。"""
        self.population = []
        print(f"Initializing population of size {self.pop_size}...")
        
        for _ in range(self.pop_size):
            builder = base_builder_cls(tolerance=1e-5)
            
            # 添加初始结构（Moser Spindle）
            builder.add_moser_spindle()
            
            wrapper = UDGBuilderWrapper(builder)
            wrapper.update_fitness()
            self.population.append(wrapper)

    def _rotate_merge_mutation(self, individual: UDGBuilderWrapper):
        """核心变异策略：2D旋转合并"""
        builder = individual.builder
        current_nodes = builder.nodes
        
        if len(current_nodes) < 3:
            return
            
        # 随机选择三个点
        indices = random.sample(range(len(current_nodes)), 3)
        
        pivot = current_nodes[indices[0]]
        point_a = current_nodes[indices[1]] 
        point_b = current_nodes[indices[2]]
        
        # 计算旋转角度
        rotation_angle = calculate_rotation_angle_2d(point_a, point_b, pivot)
        
        # 如果旋转角度很小，则使用随机有理角度
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

    def _add_points_mutation(self, individual: UDGBuilderWrapper):
        """添加随机点的变异"""
        builder = individual.builder
        
        # 添加一些随机点
        num_new_points = random.randint(3, 8)
        new_points = np.random.uniform(-1, 1, (num_new_points, 2))
        
        builder.add_points(new_points)
        
        print(f"   Added {num_new_points} random points")

    def _mutate(self, individual: UDGBuilderWrapper):
        """变异算子：主要使用旋转合并策略"""
        current_nodes = len(individual.builder.nodes)
        
        # 策略选择
        if current_nodes < 10:
            choice = random.choices(['rotate_merge', 'add_points'], weights=[0.6, 0.4], k=1)[0]
        else:
            choice = random.choices(['rotate_merge', 'add_points'], weights=[0.8, 0.2], k=1)[0]
        
        if choice == 'rotate_merge':
            self._rotate_merge_mutation(individual)
        else:
            self._add_points_mutation(individual)
        
        # 安全检查：防止节点数归零
        if len(individual.builder.nodes) == 0:
            individual.builder.add_moser_spindle()
        
        # 验证和评估
        G = individual.builder.get_graph()
        coloring = nx.greedy_color(G, strategy='largest_first')
        est_chromatic = max(coloring.values()) + 1
        print(f"   --> Validation: Greedy Coloring uses {est_chromatic} colors.")

    def step(self):
        """执行一代进化。"""
        self.generation += 1
        
        # 排序
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        best_ind = self.population[0]
        self.best_fitness_history.append(best_ind.fitness)
        
        print(f"Gen {self.generation}: Best Fitness = {best_ind.fitness:.4f} | Nodes: {len(best_ind.builder.nodes)}")
        
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
                child.builder.remove_farthest_points(ratio=0.7)
            
            child.update_fitness()
            next_gen.append(child)
            
        self.population = next_gen
        
        # 过滤掉适应度为0的个体
        self.population = [udg for udg in self.population if udg.fitness > 0]

    def get_best_graph(self):
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        return self.population[0].builder.get_graph()

def main():
    """演示2D旋转合并策略的进化算法"""
    print("=== 2D旋转合并策略进化算法演示 ===")
    
    # 配置GA
    ga = RotationMergeGeneticSearch2D(
        pop_size=8,
        max_nodes=300,
        mutation_rate=0.9,
        elite_size=2
    )
    
    # 初始化
    ga.initialize_population(UDGBuilder)
    
    print("\n开始进化过程...")
    
    # 运行几代看看效果
    for i in range(12):
        ga.step()
        
        best_G = ga.get_best_graph()
        if i % 3 == 0:  # 每3代展示一次结果
            print(f"   当前最优图: {best_G.number_of_nodes()} 节点, {best_G.number_of_edges()} 边")
    
    # 最终结果
    best_G = ga.get_best_graph()
    print(f"\n=== 最终结果 ===")
    print(f"节点数: {best_G.number_of_nodes()}")
    print(f"边数: {best_G.number_of_edges()}")
    print(f"平均度数: {2*best_G.number_of_edges()/best_G.number_of_nodes():.4f}")
    
    # 简单的可视化
    pos = nx.get_node_attributes(best_G, 'pos')
    if pos:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左图：显示节点和边
        pos_arr = np.array([pos[i] for i in range(len(pos))])
        ax1.scatter(pos_arr[:,0], pos_arr[:,1], s=30, c='blue', alpha=0.7)
        
        # 绘制一些边
        edges_to_draw = min(500, best_G.number_of_edges())
        for i, edge in enumerate(list(best_G.edges())[:edges_to_draw]):
            node1, node2 = edge
            if node1 in pos and node2 in pos:
                x_coords = [pos[node1][0], pos[node2][0]]
                y_coords = [pos[node1][1], pos[node2][1]]
                ax1.plot(x_coords, y_coords, 'r-', alpha=0.3, linewidth=0.5)
        
        ax1.set_title(f"2D旋转合并策略演化的图\n{best_G.number_of_nodes()} 节点, {best_G.number_of_edges()} 边")
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # 右图：适应度历史
        ax2.plot(ga.best_fitness_history, 'b-', linewidth=2)
        ax2.set_title('适应度历史')
        ax2.set_xlabel('代数')
        ax2.set_ylabel('最佳适应度')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('rotation_merge_2d_result.png', dpi=150, bbox_inches='tight')
        print(f"可视化结果已保存到 rotation_merge_2d_result.png")
        
        # plt.show()  # 在服务器环境中可能无法显示
    
    return best_G

if __name__ == "__main__":
    result_graph = main()