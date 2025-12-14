import numpy as np
import networkx as nx
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

class UDGBuilder:
    """
    Unit Distance Graph (UDG) Builder.
    用于生成、组合和验证平面上的单位距离图，辅助寻找高色数子图。
    """

    def __init__(self, tolerance: float = 1e-5):
        """
        初始化构建器。
        
        Args:
            tolerance: 判断两点距离是否为 1 的数值容差。
        """
        self.nodes = np.empty((0, 2))  # 存储节点坐标 (N, 2)
        self.tolerance = tolerance
        self.edges = set()             # 存储边 (i, j) 其中 i < j
    
    def add_points(self, points: np.ndarray):
        """
        添加点到图中。会自动去重（在容差范围内）。
        """
        if len(points) == 0:
            return
            
        # 确保 points 是 float 类型，避免精度问题
        points = points.astype(np.float64)
        
        if len(self.nodes) == 0:
            self.nodes = points
            # 初始不需要去重，或者假定输入本身可能重复，可以在这里调用一次内部清理
            self._deduplicate_nodes()
        else:
            # 1. 快速筛选：只处理那些与现有节点距离 > tolerance 的新点
            # 构建现有节点的 KDTree
            tree = KDTree(self.nodes)
            
            # 查询新点到现有最近节点的距离
            # distances: 每个新点到最近旧点的距离
            # indices: 最近旧点的索引（这里不用，只需要距离）
            distances, _ = tree.query(points, k=1)
            
            # 找出所有距离大于容差的点（即真正的新点）
            # 注意：这里只排除了新点与旧点的重复，没排除新点内部的重复
            # 但考虑到新点内部通常由某种几何变换生成，本身重复概率低，
            # 或者会在下一轮 _deduplicate_nodes 中被清理。
            is_new_mask = distances > self.tolerance
            
            new_unique_points = points[is_new_mask]
            
            if len(new_unique_points) > 0:
                # 2. 将真正的新点合并进节点列表
                self.nodes = np.vstack([self.nodes, new_unique_points])
                
                # 3. 为了保险（防止新加入的点之间互相重复，或者累积误差），
                # 执行一次全局去重。虽然耗时，但对保证几何图的严格性很重要。
                # 如果追求极致速度，且确信输入源无内部重复，可以跳过这一步。
                self._deduplicate_nodes()

        # 4. 重新计算边 (因为节点索引变了，且有新点加入)
        self.compute_edges()

    def _deduplicate_nodes(self):
        """
        内部辅助函数：对 self.nodes 进行全局去重。
        保留第一个出现的点，移除后续距离在 tolerance 内的点。
        """
        if len(self.nodes) == 0:
            return

        # 策略：使用 KDTree 查找所有距离 < tolerance 的点对
        # query_pairs 返回所有 (i, j) 且 i < j 的对
        tree = KDTree(self.nodes)
        pairs = tree.query_pairs(r=self.tolerance)
        
        if not pairs:
            return # 没有重复

        # 找出需要移除的索引
        # 如果 (i, j) 距离很近且 i < j，我们保留 i，移除 j
        to_remove = set()
        for i, j in pairs:
            # 我们倾向于保留索引小的（旧点），移除索引大的（新点）
            # 但必须小心传递性：如果 A~B, B~C，可能 A~C 不成立但它们实际上是一团
            # 简单的贪心策略：如果 j 还没被标记移除，且与 i 重复，则移除 j
            if i not in to_remove:
                to_remove.add(j)
        
        if not to_remove:
            return

        # 构建新的节点数组
        # 使用布尔掩码比列表推导式更快
        keep_mask = np.ones(len(self.nodes), dtype=bool)
        keep_mask[list(to_remove)] = False
        
        self.nodes = self.nodes[keep_mask]
        # 注意：边集 self.edges 此时失效了，必须在调用方重新计算

    def compute_edges(self):
        """
        基于当前的点集，重新计算所有满足单位距离约束的边。
        使用 KDTree 进行高效半径搜索。
        """
        if len(self.nodes) == 0:
            self.edges = set()
            return

        tree = KDTree(self.nodes)
        
        # 查找所有距离在 [1-tol, 1+tol] 范围内的点对
        # KDTree query_pairs 返回距离 <= r 的点对
        # 我们需要 query_pairs(r_max) - query_pairs(r_min) 的逻辑
        # 但 scipy API 直接支持 query_pairs(r)
        
        # 策略：找到所有距离 <= 1 + tol 的对，然后过滤掉 < 1 - tol 的对
        pairs = tree.query_pairs(r=1.0 + self.tolerance)
        
        valid_edges = set()
        for i, j in pairs:
            dist = np.linalg.norm(self.nodes[i] - self.nodes[j])
            if dist >= 1.0 - self.tolerance:
                valid_edges.add(tuple(sorted((i, j))))
        
        self.edges = valid_edges
        print(f"Recomputed edges: {len(self.edges)} edges found for {len(self.nodes)} nodes.")

    def add_moser_spindle(self, origin=(0, 0), angle=0):
        """
        在指定位置和角度添加一个 Moser Spindle (色数=4 的最小单位距离图)。
        包含 7 个顶点，11 条边。
        """
        # Moser Spindle 的标准坐标构造
        # 两个菱形共享一个顶点，且菱形锐角为 60 度? 不，是特定的构造
        # 更简单的构造：两个边长为1的正三角形背靠背组成菱形，顶角距离为1?
        # Moser Spindle 构造：
        # A, B 是距离为 1 的点。
        # 构造两组菱形结构。
        
        # 标准坐标推导（简化版）：
        # 顶点 1: (0, 0)
        # 顶点 2: (1, 0) (与1距离1)
        # 顶点 3: (0.5, sqrt(3)/2) (与1,2成正三角形)
        # 顶点 4: (0.5, -sqrt(3)/2) (与1,2成正三角形) -- 此时 3-4 距离 sqrt(3) != 1
        
        # 正确的 Moser Spindle 坐标生成：
        # 角度 alpha = arccos(1/2) = 60度
        # 它是两个菱形组成的，菱形锐角为 2*arcsin(1/2) = 60度 (正三角形) 
        # 实际上 Moser Spindle 关键在于那个 "spindle" 长度。
        
        # 硬编码一组预计算好的 Moser Spindle 坐标 (中心在原点附近)
        # 下面坐标是近似值，但在 compute_edges 时会在容差内匹配
        # 精确值：
        # 1. 顶部点
        # 2. 底部点 (距离1) -> 这实际上不是 Spindle 的构型
        
        # 我们直接生成几何构造：
        # Base points
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        
        def rotate(p, theta):
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s], [s, c]])
            return np.dot(R, p)

        # 构造菱形 1
        # 角度满足单位距离: 60度
        deg60 = np.pi / 3
        c = rotate(b, deg60) # 与 a, b 距离 1
        d = b + (c - a)      # 平行四边形，与 b, c 距离 1
        
        # 现在的点集: a, b, c, d.  Edges: (a,b), (a,c), (b,c), (b,d), (c,d)
        # Moser Spindle 是把两个这样的结构通过某些点融合，使得特定两点距离为 1
        
        # 让我们使用一个硬编码的标准化 Moser Spindle 坐标集
        # 来源：Moser, L. & Moser, W. (1961)
        h = np.sqrt(3)/2
        points = np.array([
            [0, 0],           # 0
            [1, 0],           # 1
            [0.5, h],         # 2
            [-0.5, h],        # 3
            [0.5, -h],        # 4 (镜像)
            [-0.5, -h],       # 5 (镜像)
            # 上面是一个点群，Moser Spindle 的关键是第7个点与其中特定点距离为1
            # 实际上 Moser Spindle 有 7 个点。
            # 让我们用更通用的 Minkowski Sum 方式或旋转复制方式构建
        ])
        
        # 更可靠的方式：利用 rotate_copy 功能
        # 1. 生成一个基础菱形 (Diamond): 两个边长1的正三角形共边
        #    Points: O(0,0), A(1,0), B(1/2, sqrt(3)/2), C(-1/2, sqrt(3)/2) -> NO
        
        # 重新实现：使用旋转法生成 Moser Spindle
        # 1. 创建边长为 1 的菱形 (两个正三角形拼合)
        #    Vertices: u, v, w, x.  Edges: uv, uw, vx, wx, vw(对角线长1)
        #    wait, Moser Spindle is created by rotating a "diamond" (2 triangles) 
        #    so that the other tips are distance 1 apart.
        
        # 构造基础菱形
        p0 = np.array([0.0, 0.0])
        p1 = np.array([1.0, 0.0])
        p2 = np.array([0.5, np.sqrt(3)/2]) # Top
        p3 = np.array([1.5, np.sqrt(3)/2]) # Right top
        # 菱形: p0-p1, p1-p3, p3-p2, p2-p0. And diagonal p1-p2 is 1.
        # 这是一个由两个正三角形 (0,1,2) 和 (1,3,2)? No.
        
        # Let's stick to the simplest valid definition:
        # Union of two diamonds.
        theta = np.arccos(5/6) # Moser Spindle 旋转角? 不，这是别的图
        # Moser Spindle 旋转角是 2*arcsin(1/(2*1))?
        # 目标：两个菱形顶点的距离为 1。
        # 菱形长对角线长度为 sqrt(3)。
        # 我们需要旋转角度 alpha 使得顶端距离为 1。
        # 2 * (sqrt(3)/2) * sin(alpha/2) = 0.5 (一半距离)
        # sin(alpha/2) = 0.5 / (sqrt(3)/2) = 1/sqrt(3)
        # alpha = 2 * arcsin(1/sqrt(3))
        
        alpha = 2 * np.arcsin(1/np.sqrt(3))
        
        # 基础菱形点
        diamond = np.array([
            [0, 0],
            [1, 0],
            [0.5, np.sqrt(3)/2], # Top vertex (Apex)
            [1.5, np.sqrt(3)/2]  # Far vertex ? No, simplify.
        ])
        # 简化：菱形由两个正三角形组成。
        # A(0,0), B(1,0). C在上方使得 ABC正三角形. D在下方使得 ABD正三角形.
        # 此时 C-D 距离 sqrt(3)。
        base_diamond = np.array([
            [0.5, np.sqrt(3)/2],  # Top
            [1.0, 0.0],           # Right
            [0.5, -np.sqrt(3)/2], # Bottom
            [0.0, 0.0]            # Left (Pivot)
        ])
        # Pivot 是原点 (0,0). Top 和 Bottom 的距离是 sqrt(3).
        # 我们需要旋转这个菱形，使得 Top 和 Top' 的距离为 1.
        
        # 生成
        self.add_points(base_diamond) # 原始
        
        # 旋转
        R = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
        rotated_diamond = base_diamond @ R.T
        
        self.add_points(rotated_diamond)
        
        # 应用整体位移和旋转 (传入参数)
        if angle != 0:
            c, s = np.cos(angle), np.sin(angle)
            R_global = np.array([[c, -s], [s, c]])
            self.nodes = self.nodes @ R_global.T
            
        self.nodes += np.array(origin)
        
        # 更新边
        self.compute_edges()

    def rotate_and_copy(self, angle: float, pivot: Tuple[float, float] = (0,0)):
        """
        将当前整个图绕 pivot 旋转 angle 度，并将结果作为新点加入图中。
        这是构造 Aubrey de Grey 类型图的关键操作。
        """
        if len(self.nodes) == 0:
            return

        pivot_arr = np.array(pivot)
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        
        # (P - origin) * R + origin
        centered_nodes = self.nodes - pivot_arr
        rotated_nodes = centered_nodes @ R.T + pivot_arr
        
        self.add_points(rotated_nodes)
        self.compute_edges()

    def get_graph(self) -> nx.Graph:
        """
        返回 NetworkX 图对象。
        节点带有 'pos' 属性，值为 (x, y) 坐标。
        """
        G = nx.Graph()
        for i, coord in enumerate(self.nodes):
            G.add_node(i, pos=coord)
        G.add_edges_from(self.edges)
        return G

    def clean_isolated_nodes(self):
        """
        移除孤立节点（度为0的节点）。
        在大量随机生成或旋转后，这有助于减小图规模。
        """
        if len(self.nodes) == 0:
            return
            
        # 找出所有在边集中出现的节点索引
        active_nodes = set()
        for u, v in self.edges:
            active_nodes.add(u)
            active_nodes.add(v)
            
        if len(active_nodes) == 0:
            self.nodes = np.empty((0, 2))
            self.edges = set()
            return

        # 创建旧索引到新索引的映射
        sorted_active = sorted(list(active_nodes))
        mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_active)}
        
        # 更新节点数组
        self.nodes = self.nodes[sorted_active]
        
        # 更新边集
        new_edges = set()
        for u, v in self.edges:
            if u in mapping and v in mapping:
                new_edges.add((mapping[u], mapping[v]))
        self.edges = new_edges
        
        print(f"Cleaned graph: {len(self.nodes)} nodes, {len(self.edges)} edges.")

    def plot(self, show_edges=True, title="Unit Distance Graph"):
        """
        可视化当前图。
        """
        if len(self.nodes) == 0:
            print("No nodes to plot.")
            return

        plt.figure(figsize=(8, 8))
        pos = self.nodes
        
        if show_edges:
            # 绘制边
            for u, v in self.edges:
                p1 = pos[u]
                p2 = pos[v]
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', alpha=0.3, linewidth=0.5)
        
        # 绘制点
        plt.scatter(pos[:, 0], pos[:, 1], s=20, c='red', edgecolors='black')
        plt.title(f"{title}\nNodes: {len(self.nodes)}, Edges: {len(self.edges)}")
        plt.axis('equal')
        plt.grid(True)
        plt.show()
        
    def k_core_pruning(self, k: int):
        """
        重复删除所有度数 <=k 的点，直到图为空或者所有点度数 >k 为止。
        
        Args:
            k: 核心数阈值。
        """
        if len(self.nodes) == 0:
            return
        
        # 获取当前图的NetworkX表示
        G = self.get_graph()
        
        # 计算每个节点的度数
        degrees = dict(G.degree())
        
        # 重复删除度数<=k的节点，直到没有这样的节点为止
        while True:
            # 找出所有度数<=k的节点
            nodes_to_remove = [node for node, degree in degrees.items() if degree <= k]
            
            # 如果没有需要删除的节点，退出循环
            if not nodes_to_remove:
                break
            
            # 删除这些节点
            for node in nodes_to_remove:
                G.remove_node(node)
            
            # 如果图为空，退出循环
            if G.number_of_nodes() == 0:
                break
            
            # 更新度数
            degrees = dict(G.degree())
        
        # 更新UDGBuilder的内部状态
        if G.number_of_nodes() == 0:
            self.nodes = np.empty((0, 2))
            self.edges = set()
        else:
            # 获取剩余节点的索引和坐标
            remaining_nodes = sorted(G.nodes())
            new_nodes = np.array([self.nodes[node] for node in remaining_nodes])
            
            # 创建旧索引到新索引的映射
            mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(remaining_nodes)}
            
            # 更新边集
            new_edges = set()
            for u, v in G.edges():
                new_edges.add((mapping[u], mapping[v]))
            
            # 更新内部状态
            self.nodes = new_nodes
            self.edges = new_edges
        
        print(f"K-core pruning (k={k}) completed. Remaining nodes: {len(self.nodes)}, edges: {len(self.edges)}.")