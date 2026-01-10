import numpy as np
import networkx as nx
import random
import copy
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
import os

from udg3d_builder import UDG3DBuilder 

class UDG3DBuilderWrapper:
    """对 UDG3DBuilder 的封装，增加了适合 GA 的深拷贝和特定变异逻辑。"""
    def __init__(self, builder):
        self.builder = builder
        self.fitness = 0.0
        self.graph_cache = None

    def copy(self):
        """深拷贝当前个体，用于产生后代"""
        new_builder = copy.deepcopy(self.builder)
        return UDG3DBuilderWrapper(new_builder)

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

def calculate_rotation_axis_and_angle(point_a: np.ndarray, point_b: np.ndarray, pivot: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    计算将点B绕原点P旋转到点A位置所需的旋转轴和角度。
    """
    # 将坐标转换为以pivot为原点的相对坐标
    a_rel = point_a - pivot
    b_rel = point_b - pivot
    
    # 归一化向量
    a_norm = np.linalg.norm(a_rel)
    b_norm = np.linalg.norm(b_rel)
    
    if a_norm < 1e-10 or b_norm < 1e-10:
        return np.array([0, 0, 1]), np.pi / 2
    
    a_unit = a_rel / a_norm
    b_unit = b_rel / b_norm
    
    # 计算旋转轴（叉积）
    rotation_axis = np.cross(b_unit, a_unit)
    rotation_axis_norm = np.linalg.norm(rotation_axis)
    
    if rotation_axis_norm < 1e-10:
        # 如果向量平行或反平行
        if np.dot(b_unit, a_unit) > 0:
            return np.array([0, 0, 1]), 0.0
        else:
            return np.array([1, 0, 0]), np.pi
    
    rotation_axis = rotation_axis / rotation_axis_norm
    
    # 计算旋转角度
    cos_angle = np.clip(np.dot(b_unit, a_unit), -1.0, 1.0)
    rotation_angle = np.arccos(cos_angle)
    
    return rotation_axis, rotation_angle

def rotate_points_around_axis(points: np.ndarray, rotation_axis: np.ndarray, angle: float, pivot: np.ndarray) -> np.ndarray:
    """
    绕指定轴和角度旋转点集。
    """
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

class RotationMergeGeneticSearch3D:
    def __init__(self, pop_size=10, max_nodes=500, mutation_rate=0.9, elite_size=2):
        """使用旋转合并策略的进化算法。"""
        self.pop_size = pop_size
        self.max_nodes = max_nodes
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.population: List[UDG3DBuilderWrapper] = []
        self.generation = 0
        
        self.best_fitness_history = []

    def initialize_population(self, base_builder_cls):
        """初始化种群。"""
        self.population = []
        print(f"Initializing population of size {self.pop_size}...")
        
        for _ in range(self.pop_size):
            builder = base_builder_cls(tolerance=1e-5)
            
            # 添加初始结构（Moser Spindle的3D版本）
            initial_points = np.array([
                [0, 0, 0],
                [1, 0, 0],
                [0.5, np.sqrt(3)/2, 0],
                [0.5, np.sqrt(3)/6, np.sqrt(6)/3],
                [0.5, np.sqrt(3)/6, -np.sqrt(6)/3]
            ])
            
            builder.add_points(initial_points)
            
            wrapper = UDG3DBuilderWrapper(builder)
            wrapper.update_fitness()
            self.population.append(wrapper)

    def _rotate_merge_mutation(self, individual: UDG3DBuilderWrapper):
        """核心变异策略：旋转合并"""
        builder = individual.builder
        current_nodes = builder.nodes
        
        if len(current_nodes) < 3:
            return
            
        # 随机选择三个点
        indices = random.sample(range(len(current_nodes)), 3)
        
        pivot = current_nodes[indices[0]]
        point_a = current_nodes[indices[1]] 
        point_b = current_nodes[indices[2]]
        
        # 计算旋转参数
        rotation_axis, rotation_angle = calculate_rotation_axis_and_angle(
            point_a, point_b, pivot
        )
        
        # 如果旋转角度很小，则使用随机角度
        if rotation_angle < 1e-6:
            rotation_angle = np.pi / 3  # 60度
            rotation_axis = np.array([1, 1, 1]) / np.sqrt(3)
        
        print(f"   Rotate-Merge: P={pivot}, A={point_a}, B={point_b}")
        print(f"   Rotation: axis={rotation_axis}, angle={np.degrees(rotation_angle):.2f}°")
        
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

    def _add_points_mutation(self, individual: UDG3DBuilderWrapper):
        """添加随机点的变异"""
        builder = individual.builder
        
        # 添加一些随机点
        num_new_points = random.randint(3, 8)
        new_points = np.random.uniform(-1, 1, (num_new_points, 3))
        
        builder.add_points(new_points)
        
        print(f"   Added {num_new_points} random points")

    def _mutate(self, individual: UDG3DBuilderWrapper):
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
            initial_points = np.array([
                [0, 0, 0],
                [1, 0, 0],
                [0.5, np.sqrt(3)/2, 0]
            ])
            individual.builder.add_points(initial_points)
        
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
    """演示旋转合并策略的进化算法"""
    print("=== 旋转合并策略进化算法演示 ===")
    
    # 配置GA
    ga = RotationMergeGeneticSearch3D(
        pop_size=8,
        max_nodes=300,
        mutation_rate=0.9,
        elite_size=2
    )
    
    # 初始化
    ga.initialize_population(UDG3DBuilder)
    
    print("\n开始进化过程...")
    
    # 运行几代看看效果
    for i in range(15):
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
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        pos_arr = np.array([pos[i] for i in range(len(pos))])
        ax.scatter(pos_arr[:,0], pos_arr[:,1], pos_arr[:,2], s=30, c='blue', alpha=0.7)
        
        # 绘制一些边
        edges_to_draw = min(200, best_G.number_of_edges())
        for i, edge in enumerate(list(best_G.edges())[:edges_to_draw]):
            node1, node2 = edge
            if node1 in pos and node2 in pos:
                x_coords = [pos[node1][0], pos[node2][0]]
                y_coords = [pos[node1][1], pos[node2][1]]
                z_coords = [pos[node1][2], pos[node2][2]]
                ax.plot(x_coords, y_coords, z_coords, 'r-', alpha=0.3, linewidth=0.5)
        
        ax.set_title(f"旋转合并策略演化的3D图\n{best_G.number_of_nodes()} 节点, {best_G.number_of_edges()} 边")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # 保存图像
        plt.savefig('rotation_merge_result.png', dpi=150, bbox_inches='tight')
        print(f"可视化结果已保存到 rotation_merge_result.png")
        
        # plt.show()  # 在服务器环境中可能无法显示
    
    return best_G

if __name__ == "__main__":
    result_graph = main()