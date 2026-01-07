import numpy as np
import math
import networkx as nx
import random
import copy
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from fractions import Fraction

# 导入 AlgebraicUDGBuilder
from integer_udg_builder import AlgebraicUDGBuilder, AlgebraicField, AlgebraicComplex
from graph_coloring import is_colorable

# 假设之前的代码保存在 integer_udg_builder.py 中

class IntegerUDGWrapper:
    """
    对 AlgebraicUDGBuilder 的封装，增加了适合随机搜索的深拷贝和特定变异逻辑。
    """
    def __init__(self, builder):
        self.builder = builder
        self.fitness = 0.0
        self.graph_cache = None # 缓存 NetworkX 对象

    def copy(self):
        """深拷贝当前个体，用于产生后代"""
        new_builder = copy.deepcopy(self.builder)
        return IntegerUDGWrapper(new_builder)

    def update_fitness(self):
        """
        计算适应度：结合图的平均度数和大小。
        适应度越高，表示图越难4染色且大小越小。
        """
        # 总是先清理孤立点
        self.clean_isolated_nodes()
        
        G = self.builder.get_graph()
        self.graph_cache = G
        
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        
        if num_nodes == 0:
            self.fitness = 0.0
        else:
            # 使用平均度数作为适应度指标
            self.fitness = 2.0 * num_edges / num_nodes
        
        return self.fitness
    
    def clean_isolated_nodes(self):
        """
        清理孤立点
        """
        G = self.builder.get_graph()
        isolated_nodes = [n for n, d in G.degree() if d == 0]
        
        if isolated_nodes:
            # 重新构建节点列表，排除孤立点
            new_points = []
            for i, point in enumerate(self.builder.points):
                if i not in isolated_nodes:
                    new_points.append(point)
            
            self.builder.points = new_points
            self.builder.compute_edges()

# --- 旋转库生成器 --- 
def generate_rotation_library(field: AlgebraicField) -> List[AlgebraicComplex]:
    """
    生成一个旋转库，包含 unit-norm 的 AlgebraicComplex 元素
    这里简单生成一些基本的旋转，后续可以扩展
    """
    rotations = [AlgebraicComplex(field, field.one(), field.zero())]
    
    for a in range(2, 5):
        for b in range(1, 3):
            if math.sqrt(a) / b >= math.sqrt(2) / 2:
                cosv_val = 1 - 1 / ((2 * a) / (b * b))
                cosv = field.sub(field.one(), field.scalar_mul(Fraction(b * b, 2 * a), field.one()))
                try:
                    rtv = field.get_root(4 * a - (b * b))
                    rtv = field.scalar_mul(Fraction(1, b), rtv)
                    sinv = field.scalar_mul(Fraction(b * b, 2 * a), rtv)
                    rotations.append(AlgebraicComplex(field, cosv, sinv))
                    rotations.append(AlgebraicComplex(field, cosv, field.scalar_mul(Fraction(-1, 1), sinv)))
                except ValueError:
                    pass
    for rot in rotations:
        #print(rot.to_float_pair())
        assert field.equal(field.one(), rot.abs2())
    return rotations

class IntegerRandomSearch:
    """
    基于整数运算的随机搜索算法，使用 AlgebraicUDGBuilder
    """
    def __init__(self, 
                 pop_size=20, 
                 max_nodes=2000, 
                 mutation_rate=0.8, 
                 elite_size=3):
        """
        Args:
            pop_size: 种群大小
            max_nodes: 为了防止内存爆炸，限制最大节点数
            mutation_rate: 变异概率
            elite_size: 每代保留的最优个体数
        """
        self.pop_size = pop_size
        self.max_nodes = max_nodes
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.population: List[IntegerUDGWrapper] = []
        self.generation = 0
        
        # 初始化数据记录
        self.best_fitness_history = []
        
        # 扩域设置：包含 sqrt(3)、sqrt(5) 和 sqrt(11)
        self.square_roots = [3, 5, 11]

    def initialize_population(self):
        """
        初始化种群。
        每个个体都从一个标准的 Moser Spindle 开始，但带有随机的初始旋转。
        """
        self.population = []
        print(f"Initializing population of size {self.pop_size}...")
        
        for _ in range(self.pop_size):
            # 创建基础构建器，使用包含 sqrt(3)、sqrt(5) 和 sqrt(11) 的扩域
            builder = AlgebraicUDGBuilder(self.square_roots, tolerance=1e-9)
            
            # 初始种子：Moser Spindle
            # 为了增加多样性，给每个初始个体一个随机的整体旋转
            builder.add_moser_spindle()
            
            wrapper = IntegerUDGWrapper(builder)
            wrapper.update_fitness()
            self.population.append(wrapper)

    def _mutate(self, individual: IntegerUDGWrapper, other_individual: IntegerUDGWrapper):
        """
        变异算子核心逻辑。
        """
        builder = individual.builder
        other_builder = other_individual.builder
        current_nodes = len(builder.points)
        
        # 策略选择概率
        # 如果节点数过多，强制增加 Pruning 的概率
        if current_nodes > self.max_nodes:
            probs = [0.1, 0.1, 0.8] # Rotate, Minkowski, Prune
        else:
            probs = [0.5, 0.5, 0] # Rotate, Minkowski, Prune
            
        choice = random.choices(['rotate_merge', 'minkowski', 'prune'], weights=probs, k=1)[0]
        
        if choice == 'rotate_merge':
            # --- 变异 1: Rotate & Merge ---
            # 随机旋转并复制当前图形（仅对第一个builder操作）
            
            # 从旋转库中随机选择一个旋转
            rotation_library = generate_rotation_library(builder.field)
            rot = random.choice(rotation_library)
            
            # 随机选择旋转中心 (原点或随机现有点)
            if len(builder.points) > 0 and random.random() < 0.3:
                pivot_idx = random.randint(0, len(builder.points)-1)
                pivot = builder.points[pivot_idx]  # 直接使用 AlgebraicComplex 点
            else:
                pivot = None  # 使用默认原点
            
            # 旋转并复制
            builder.rotate_and_copy(rot, pivot=pivot)
            
        elif choice == 'minkowski':
            # --- 变异 2: Minkowski Sum ---
            # 将两个图剪枝到 sqrt(self.max_nodes) 大小
            prune_size = 100
            
            # 对第一个图剪枝
            temp_builder1 = copy.deepcopy(builder)
            temp_builder1.prune_to_size(prune_size)
            
            # 对第二个图剪枝
            temp_builder2 = copy.deepcopy(other_builder)
            temp_builder2.prune_to_size(prune_size)
            
            # 计算点集的闵可夫斯基和
            new_points = []
            for p1 in temp_builder1.points:
                for p2 in temp_builder2.points:
                    sum_point = p1.add(p2)
                    new_points.append(sum_point)
            
            # 添加到第一个builder
            builder.add_algebraic_points(new_points)
            builder.compute_edges()
            
        elif choice == 'prune':
            # --- 变异 3: Pruning (Low Degree Removal) ---
            # 移除度数最小的点（仅对第一个builder操作）
            builder.prune_to_size(len(builder.points) // 2)
        
        # 安全检查：防止节点数归零
        if len(builder.points) == 0:
            builder.add_moser_spindle()

        print(f"{len(builder.points)} nodes, {len(builder.edges)} edges")
        # 验证图的4染色性
        G = builder.get_graph()
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
        执行一代随机搜索。
        """
        self.generation += 1
        
        # 1. 排序 (根据 Fitness 降序)
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        best_ind = self.population[0]
        self.best_fitness_history.append(best_ind.fitness)
        
        print(f"Gen {self.generation}: Best Fitness (Avg Deg) = {best_ind.fitness:.4f} | Nodes: {len(best_ind.builder.points)}")
        
        # 2. 精英保留 (Elitism)
        next_gen = []
        for i in range(self.elite_size):
            # 直接保留，深拷贝以防意外修改
            next_gen.append(self.population[i].copy())
            
        # 3. 繁殖与变异
        for parent in self.population:
            # 随机选择一个父代
            #parent = random.choice(self.population)
            # 复制产生子代
            for k in range(5):
                child = parent.copy()
                
                # 变异
                if random.random() < self.mutation_rate:
                    # 从父母列表中随机选取第二个参数
                    other_parent = random.choice(self.population[:self.pop_size // 2])
                    self._mutate(child, other_parent)
    
                if child.fitness > 10000:
                    next_gen.append(child)
                    next_gen.sort(key=lambda x: x.fitness, reverse=True)
                    self.population = next_gen[:self.pop_size]
                    return
                # 限制大小 (硬约束)
                if len(child.builder.points) > self.max_nodes:
                    # 强制修剪
                    child.builder.prune_to_size(self.max_nodes)
                
                # 计算子代适应度
                child.update_fitness()
                next_gen.append(child)
            
        next_gen.sort(key=lambda x: x.fitness, reverse=True)
        self.population = next_gen[:self.pop_size]

    def get_best_graph(self):
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        return self.population[0].builder.get_graph()

if __name__ == "__main__":
    # 1. 配置随机搜索
    rs = IntegerRandomSearch(
        pop_size=50,
        max_nodes=2500,  # 限制图规模，防止变慢
        mutation_rate=0.9, # 高变异率，因为探索空间很大
        elite_size=5
    )
    
    # 2. 初始化
    rs.initialize_population()
    
    # 3. 运行进化循环
    try:
        for i in range(50): # 运行 50 代试试
            rs.step()
            
            best_G = rs.get_best_graph()

            print(best_G)
            # 使用贪心算法快速估算色数
            
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
        print("Random search stopped by user.")

    # 4. 结果展示
    best_G = rs.get_best_graph()
    print(f"Final Best Graph: {best_G.number_of_nodes()} nodes, {best_G.number_of_edges()} edges.")
    
    # 可视化
    plt.figure(figsize=(10, 10))
    pos = nx.get_node_attributes(best_G, 'pos')
    # 转换为 numpy 数组用于绘图
    if pos:
        pos_arr = np.array([pos[i] for i in range(len(pos))])
        plt.scatter(pos_arr[:,0], pos_arr[:,1], s=10, c='red', alpha=0.6)
        # 绘制部分边（太多边会画不开）
        if best_G.number_of_edges() < 5000:
            nx.draw_networkx_edges(best_G, pos, alpha=0.1)
    plt.title(f"Evolved Integer UDG (Avg Deg: {2*best_G.number_of_edges()/best_G.number_of_nodes():.2f})")
    plt.show()
    
    # 保存图中所有点的坐标
    coordinates_filename = "best_graph_coordinates.txt"
    pos = nx.get_node_attributes(best_G, 'pos')
    with open(coordinates_filename, 'w') as f:
        # 写入点的数量和边的数量
        f.write(f"{best_G.number_of_nodes()} {best_G.number_of_edges()}\n")
        # 写入每个点的坐标
        for node_id in sorted(pos.keys()):
            x, y = pos[node_id]
            f.write(f"{node_id} {x} {y}\n")
        # 写入每条边
        for u, v in best_G.edges():
            f.write(f"{u} {v}\n")
    print(f"Best graph coordinates saved to {coordinates_filename}")
