# udg3d_builder.py
import numpy as np
import networkx as nx
from scipy.spatial import KDTree
from typing import Tuple, Optional, List, Iterable

class UDG3DBuilder:
    """
    3D Unit Distance Graph (UDG) Builder.
    用于生成、组合和验证三维空间中的单位距离图。
    接口尽量模仿二维的 UDGBuilder：nodes, edges, compute_edges, merge 等。
    """

    def __init__(self, tolerance: float = 1e-5):
        """
        Args:
            tolerance: 判断两点距离是否为 1 的数值容差。
        """
        self.nodes = np.empty((0, 3))   # (N, 3)
        self.edges = set()              # {(i, j)}
        self.tolerance = tolerance

    # ---------- 节点与边的维护 ----------

    def _deduplicate_nodes(self):
        """全局去重：距离在 tolerance 内的点视为同一点，保留索引较小者。"""
        if len(self.nodes) == 0:
            return
        tree = KDTree(self.nodes)
        pairs = tree.query_pairs(r=self.tolerance)
        if not pairs:
            return
        to_remove = set()
        for i, j in pairs:
            if i not in to_remove:
                to_remove.add(j)
        if not to_remove:
            return
        keep_mask = np.ones(len(self.nodes), dtype=bool)
        keep_mask[list(to_remove)] = False
        self.nodes = self.nodes[keep_mask]

    def add_points(self, points: np.ndarray):
        """
        添加一批 3D 点，自动去重并重算单位距离边。
        points: shape (M, 3)
        """
        if len(points) == 0:
            return
        points = points.astype(np.float64)
        if len(self.nodes) == 0:
            self.nodes = points
            self._deduplicate_nodes()
        else:
            tree = KDTree(self.nodes)
            dists, _ = tree.query(points, k=1)
            is_new = dists > self.tolerance
            new_pts = points[is_new]
            if len(new_pts) > 0:
                self.nodes = np.vstack([self.nodes, new_pts])
                self._deduplicate_nodes()
        self.compute_edges()

    def compute_edges(self):
        """基于当前点集重新计算所有单位距离边。"""
        n = len(self.nodes)
        if n == 0:
            self.edges = set()
            return
        tree = KDTree(self.nodes)
        pairs = tree.query_pairs(r=1.0 + self.tolerance)
        valid_edges = set()
        for i, j in pairs:
            dist = np.linalg.norm(self.nodes[i] - self.nodes[j])
            if dist >= 1.0 - self.tolerance:
                valid_edges.add(tuple(sorted((i, j))))
        self.edges = valid_edges
        print(f"[UDG3D] Recomputed edges: {len(self.edges)} edges for {len(self.nodes)} nodes.")

    def merge(self, other: "UDG3DBuilder"):
        """将另一个三维 UDG 的点加入当前图中。"""
        if len(other.nodes) == 0:
            return
        self.add_points(other.nodes)

    def get_graph(self) -> nx.Graph:
        """返回 NetworkX 图对象，节点带 'pos'=(x,y,z) 属性。"""
        G = nx.Graph()
        for i, coord in enumerate(self.nodes):
            G.add_node(i, pos=coord)
        G.add_edges_from(self.edges)
        return G

    # ---------- 三维旋转与“旋转-复制/旋转-合并” ----------

    @staticmethod
    def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
        """
        轴角 -> 3x3 旋转矩阵 (Rodrigues 公式)。
        axis: shape (3,), 不要求归一化，这里内部归一化。[web:34][web:38]
        """
        axis = np.asarray(axis, dtype=np.float64)
        norm = np.linalg.norm(axis)
        if norm == 0:
            return np.eye(3)
        n = axis / norm
        x, y, z = n
        c = np.cos(angle)
        s = np.sin(angle)
        C = 1 - c
        R = np.array([
            [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
            [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
            [z*x*C - y*s, z*y*C + x*s, z*z*C + c  ]
        ], dtype=np.float64)
        return R

    def rotate(self, axis: Tuple[float, float, float], angle: float,
               pivot: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        """
        将当前图绕三维轴 axis 通过 pivot 旋转 angle。
        axis: 三维向量，不必是单位向量。
        angle: 弧度。[web:34]
        pivot: 旋转中心。
        """
        if len(self.nodes) == 0:
            return
        axis = np.asarray(axis, dtype=np.float64)
        R = self.axis_angle_to_matrix(axis, angle)
        pivot_arr = np.asarray(pivot, dtype=np.float64)
        centered = self.nodes - pivot_arr
        rotated = centered @ R.T + pivot_arr
        self.nodes = rotated
        self.compute_edges()

    def rotate_and_copy(self, axis: Tuple[float, float, float], angle: float,
                        pivot: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        """
        将当前图绕轴 axis 通过 pivot 旋转 angle，并把旋转后的点加入图中（旋转-合并）。
        和 2D 里的 rotate_and_copy 对应，只是旋转推广到 3D。
        """
        if len(self.nodes) == 0:
            return
        axis = np.asarray(axis, dtype=np.float64)
        R = self.axis_angle_to_matrix(axis, angle)
        pivot_arr = np.asarray(pivot, dtype=np.float64)
        centered = self.nodes - pivot_arr
        rotated = centered @ R.T + pivot_arr
        self.add_points(rotated)  # 内部会 dedup + compute_edges()

    # ---------- 一些实用操作 ----------

    def clean_isolated_nodes(self):
        """移除所有度为 0 的点。"""
        if len(self.nodes) == 0:
            return
        G = self.get_graph()
        active_nodes = [n for n, d in G.degree() if d > 0]
        if not active_nodes:
            self.nodes = np.empty((0, 3))
            self.edges = set()
            return
        active_nodes = sorted(active_nodes)
        mapping = {old: new for new, old in enumerate(active_nodes)}
        new_nodes = self.nodes[active_nodes]
        new_edges = set()
        for u, v in self.edges:
            if u in mapping and v in mapping:
                new_edges.add((mapping[u], mapping[v]))
        self.nodes = new_nodes
        self.edges = new_edges
        print(f"[UDG3D] Cleaned: {len(self.nodes)} nodes, {len(self.edges)} edges.")

    def remove_farthest_points(self, ratio: float):
        """
        移除度数最小的一定比例的点。
        
        Args:
            ratio: 移除点的比例，范围 [0, 1]。0 表示不移除任何点，1 表示移除所有点。
        """
        if len(self.nodes) == 0:
            return
            
        # 确保比例在有效范围内
        ratio = max(0.0, min(1.0, ratio))
        
        if ratio == 0.0:
            return
        elif ratio == 1.0:
            self.nodes = np.empty((0, 3))
            self.edges = set()
            print(f"[UDG3D] Removed all {len(self.nodes)} nodes.")
            return
        
        # 获取图的度数信息
        G = self.get_graph()
        degrees = dict(G.degree())
        
        if not degrees:
            return
            
        # 按度数从小到大排序节点
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1])
        
        # 确定需要移除的点的数量
        num_to_remove = int(len(self.nodes) * ratio)
        num_to_remove = max(1, num_to_remove)  # 至少移除1个点
        num_to_remove = min(len(self.nodes) - 1, num_to_remove)  # 至少保留1个点
        
        # 获取要移除的节点索引
        nodes_to_remove = [node for node, degree in sorted_nodes[:num_to_remove]]
        indices_to_remove = sorted(nodes_to_remove)
        
        # 创建保留节点的掩码
        keep_mask = np.ones(len(self.nodes), dtype=bool)
        keep_mask[indices_to_remove] = False
        
        # 更新节点数组
        new_nodes = self.nodes[keep_mask]
        
        # 创建旧索引到新索引的映射
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(np.where(keep_mask)[0])}
        
        # 更新边集
        new_edges = set()
        for u, v in self.edges:
            if u in old_to_new and v in old_to_new:
                new_edges.add((old_to_new[u], old_to_new[v]))
        
        # 更新内部状态
        self.nodes = new_nodes
        self.edges = new_edges
        
        print(f"[UDG3D] Removed {len(indices_to_remove)} lowest degree points. Remaining nodes: {len(self.nodes)}, edges: {len(self.edges)}.")
