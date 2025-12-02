import numpy as np
import networkx as nx
import random
import copy
import matplotlib.pyplot as plt
from typing import List, Optional
from z3 import Solver, Bool, Or, And, Not

# 假设之前的代码保存在 udg_builder.py 中
from udg_builder import UDGBuilder 
from graph_coloring import is_colorable

# 为了代码独立运行，这里简略重新定义必要的 UDGBuilder 接口，
# 实际使用时请直接引用你之前保存的文件。
class UDGBuilderWrapper:
    """
    对 UDGBuilder 的封装，增加了适合 GA 的深拷贝和特定变异逻辑。
    """
    def __init__(self, builder):
        self.builder = builder
        self.fitness = 0.0
        self.graph_cache = None # 缓存 NetworkX 对象

    def copy(self):
        """深拷贝当前个体，用于产生后代"""
        new_builder = copy.deepcopy(self.builder)
        return UDGBuilderWrapper(new_builder)

    def update_fitness(self):
        """
        计算适应度：平均度数 (Average Degree)。
        Avg Degree = 2 * |E| / |V|
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
            self.fitness = 2.0 * num_edges / num_nodes
        
        return self.fitness

# --- 有理角度库生成器 ---
def get_rational_angles():
    angles = [np.arccos(1/3), np.arccos(-1/3)] # de Grey 的核心角度
    # 添加勾股数角度 (3,4,5), (5,12,13) 等
    triples = [(3,4,5), (5,12,13), (8,15,17), (7,24,25)]
    for a, b, c in triples:
        angles.append(np.arccos(a/c))
        angles.append(np.arccos(-a/c))
    return list(set(angles))

RATIONAL_ANGLES = get_rational_angles()

class GeneticUDGSearch:
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
        self.population: List[UDGBuilderWrapper] = []
        self.generation = 0
        
        # 初始化数据记录
        self.best_fitness_history = []

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
            initial_rotation = random.uniform(0, 2*np.pi)
            builder.add_moser_spindle(angle=initial_rotation)
            
            wrapper = UDGBuilderWrapper(builder)
            wrapper.update_fitness()
            self.population.append(wrapper)

    def _mutate(self, individual: UDGBuilderWrapper):
        """
        变异算子核心逻辑。
        """
        builder = individual.builder
        current_nodes = len(builder.nodes)
        
        # 策略选择概率
        # 如果节点数过多，强制增加 Pruning 的概率
        if current_nodes > self.max_nodes:
            probs = [0, 0, 1] # Rotate, Minkowski, Prune
        else:
            probs = [0.6, 0.3, 0.1] # Rotate, Minkowski, Prune
            
        choice = random.choices(['rotate_merge', 'minkowski', 'prune'], weights=probs, k=1)[0]
        
        if choice == 'rotate_merge':
            # --- 变异 1: Rotate & Merge ---
            # 从有理角度库中随机选一个，或者微小概率选随机角度
            if random.random() < 0.9:
                angle = random.choice(RATIONAL_ANGLES)
            else:
                angle = random.uniform(0.01, 1.0) # 小幅随机扰动
                
            # 随机选择旋转中心 (原点或随机现有点)
            if len(builder.nodes) > 0 and random.random() < 0.3:
                pivot_idx = random.randint(0, len(builder.nodes)-1)
                pivot = tuple(builder.nodes[pivot_idx])
            else:
                pivot = (0, 0)
                
            builder.rotate_and_copy(angle, pivot=pivot)
            
        elif choice == 'minkowski':
            # --- 变异 2: Minkowski Sum (Translation Copy) ---
            # 实际上是 G U (G + v)，其中 |v|=1。
            # 这相当于把图往某个方向平移一个单位距离并合并。
            # 这会产生大量新的单位距离边（连接原图和影子图的对应点）。
            
            if random.random() < 0.5:
                theta = random.choice(RATIONAL_ANGLES)
            else:
                theta = random.uniform(0, 2*np.pi) # 小幅随机扰动
            translation_vector = np.array([np.cos(theta), np.sin(theta)])
            
            # 获取当前所有点，平移，然后添加回图中
            if len(builder.nodes) > 0:
                new_points = builder.nodes + translation_vector
                builder.add_points(new_points)
                builder.compute_edges()
                
        elif choice == 'prune':
            # --- 变异 3: Pruning (K-Core Style) ---
            # 移除低度数的点，保留核心冲突结构
            G = individual.graph_cache
            if G is None: 
                G = builder.get_graph()
            
            # 计算度数
            degrees = dict(G.degree())
            if not degrees: return
            
            avg_deg = sum(degrees.values()) / len(degrees)
            # 移除度数低于平均值的点（或者低于固定阈值如 2 或 3）
            threshold = min(3, int(avg_deg * 0.8)) 
            
            to_keep_indices = [n for n, d in degrees.items() if d >= threshold]
            
            if len(to_keep_indices) > 0:
                # 更新 builder 的 nodes
                # 注意：这里需要直接操作 builder 内部数据，比较 tricky
                # 简单做法：提取保留点的坐标，清空 builder，重新添加
                # (性能较低但实现安全)
                coords = builder.nodes[to_keep_indices]
                builder.nodes = coords # Numpy array slicing
                builder.compute_edges() # 重新计算边
        
        # 安全检查：防止节点数归零
        if len(builder.nodes) == 0:
            builder.add_moser_spindle()

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
                self._mutate(child)
            
            # 限制大小 (硬约束)
            if len(child.builder.nodes) > self.max_nodes:
                # 强制修剪
                # 这里的逻辑简化处理：如果太大，就重置回较小的状态或强力修剪
                # 这里演示简单的强力 Pruning
                child.builder.clean_isolated_nodes()
                # 如果还大，可能需要更激进的随机采样保留
            
            # 计算子代适应度
            child.update_fitness()
            next_gen.append(child)
            
        self.population = next_gen

    def get_best_graph(self):
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        return self.population[0].builder.get_graph()

if __name__ == "__main__":
    from udg_builder import UDGBuilder # 导入你保存的类

    # 1. 配置 GA
    ga = GeneticUDGSearch(
        pop_size=100,
        max_nodes=500,  # 限制图规模，防止变慢
        mutation_rate=0.9, # 高变异率，因为探索空间很大
        elite_size=2
    )
    
    # 2. 初始化
    ga.initialize_population(UDGBuilder)
    
    # 3. 运行进化循环
    try:
        for i in range(50): # 运行 20 代试试
            ga.step()
            
            best_G = ga.get_best_graph()

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
        print("Evolution stopped by user.")

    # 4. 结果展示
    best_G = ga.get_best_graph()
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
    plt.title(f"Evolved UDG (Avg Deg: {2*best_G.number_of_edges()/best_G.number_of_nodes():.2f})")
    plt.show()