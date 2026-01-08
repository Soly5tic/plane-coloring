# algebraic_udg_builder.py
import itertools
from fractions import Fraction
from typing import List, Tuple, Dict, Iterable, Optional, Union

import networkx as nx
import numpy as np
from scipy.spatial import KDTree


class AlgebraicField:
    """
    一个简单的多平方根扩域 Q(√d1, √d2, ...) 实现。
    - 给定若干整数 d_i，假定它们两两不同且非零非平方数。
    - 采用标准基: 所有 sqrt(d1)^e1 * ... * sqrt(dm)^em, ei ∈ {0,1} 共 2^m 个基。
    - 每个数表示为该基上的有理系数向量，内部用 Fraction 保证精度。
    """

    def __init__(self, square_roots: List[int]):
        """
        square_roots: 例如 [2, 3] 表示构造 Q(√2, √3)
        """
        self.ds = list(square_roots)
        self.m = len(self.ds)
        # 基：所有 {0,1}^m 的指数向量
        self.exponents = list(itertools.product([0, 1], repeat=self.m))
        self.dim = len(self.exponents)  # = 2^m
        # 预先构造乘法表：基_i * 基_j = sum_k c_ijk * 基_k, c_ijk ∈ Q
        # 这里用“合并指数并把偶数次幂下沉为有理数”的方式手算乘法
        self._mul_cache: Dict[Tuple[int, int], List[Fraction]] = {}

    def zero(self) -> List[Fraction]:
        return [Fraction(0) for _ in range(self.dim)]

    def one(self) -> List[Fraction]:
        v = self.zero()
        v[0] = Fraction(1)  # 对应所有指数为0的基(1)
        return v

    def _get_square_part_and_remainder(self, n: int) -> Tuple[int, int]:
        """
        提取整数n的最大平方因子和剩余部分。
        返回：(square_part, remainder)，其中 square_part * square_part * remainder = n，且 remainder 无平方因子。
        """
        if n == 0:
            return 0, 0
        if n < 0:
            raise ValueError(f"Negative number: {n}")
        
        square_part = 1
        remainder = n
        
        # 提取平方因子
        i = 2
        while i*i <= remainder:
            if remainder % i == 0:
                count = 0
                while remainder % i == 0:
                    count += 1
                    remainder //= i
                # 偶数个因子加入平方部分
                if count >= 2:
                    square_part *= i ** (count // 2)
                # 奇数个因子，剩余1个在remainder中
                if count % 2 == 1:
                    remainder *= i
            i += 1
        
        return square_part, remainder

    def get_root(self, n : int) -> List[Fraction]:
        """
        获取 n 的平方根在标准基上的坐标向量。
        先提取 n 的所有平方因子，并尝试将剩余的部分分解为扩域的平方根的乘积。
        例如：
        - n=2 时，返回 [0, 1]，表示 sqrt(2) = 1 * sqrt(2)
        - n=8=4*2 时，返回 [0, 2]，表示 sqrt(8) = 2 * sqrt(2)
        - n=6=2*3 时，返回 [0, 0, 0, 1]，表示 sqrt(6) = 1 * sqrt(2)*sqrt(3)
        """
        if n < 0:
            raise ValueError(f"Negative number: {n}")
        
        if n == 0:
            return self.zero()
        
        if n == 1:
            return self.one()
        
        # 提取平方因子和剩余部分：n = square_part² * remainder
        square_part, remainder = self._get_square_part_and_remainder(n)
        
        # 如果剩余部分为1，直接返回平方因子的平方根（即整数）
        if remainder == 1:
            return self.from_rational(Fraction(square_part))
        
        # 检查剩余部分是否可以分解为self.ds中元素的乘积
        # 这里使用回溯法尝试所有可能的组合
        from itertools import combinations_with_replacement
        
        # 过滤出可能的因子（小于等于remainder且能整除remainder）
        possible_factors = [d for d in self.ds if d <= remainder and remainder % d == 0]
        
        if not possible_factors:
            raise ValueError(f"Cannot decompose remainder {remainder} into product of square roots in the field")
        
        # 尝试找到乘积为remainder的组合
        found = False
        factors_used = []
        
        # 检查单个因子的情况
        if remainder in self.ds:
            factors_used = [remainder]
            found = True
        else:
            # 检查多个因子的组合
            max_factors = len(self.ds)  # 最多检查所有因子的组合
            for k in range(2, max_factors + 1):
                for combo in combinations_with_replacement(possible_factors, k):
                    product = 1
                    for d in combo:
                        product *= d
                        if product > remainder:
                            break
                    if product == remainder:
                        factors_used = list(combo)
                        found = True
                        break
                if found:
                    break
        
        if not found:
            raise ValueError(f"Cannot decompose remainder {remainder} into product of square roots in the field")
        
        # 构建结果向量：square_part * product(sqrt(d) for d in factors_used)
        result = self.from_rational(Fraction(square_part))
        
        for d in factors_used:
            # 找到该平方因子在self.ds中的索引
            ds_idx = self.ds.index(d)
            # 创建对应的指数向量（只有该位置为1，其他为0）
            exp = [0] * self.m
            exp[ds_idx] = 1
            exp_tuple = tuple(exp)
            # 找到该指数向量在self.exponents中的索引
            idx = self.exponents.index(exp_tuple)
            # 创建该平方根对应的向量
            sqrt_d = self.zero()
            sqrt_d[idx] = Fraction(1)
            # 乘以当前结果
            result = self.mul(result, sqrt_d)
        
        return result

    def basis_vector(self, idx: int) -> List[Fraction]:
        v = self.zero()
        v[idx] = Fraction(1)
        return v

    def _mul_basis(self, i: int, j: int) -> List[Fraction]:
        """
        计算基_i * 基_j 的结果，在基上的坐标向量。
        基_i 对应 exponents[i] = (e1,...,em) 表示 prod sqrt(dk)^ek
        两者相乘：指数相加，如果某个指数 >=2，则拆成 (指数 mod 2) + (指数//2)*2
        而 sqrt(dk)^2 = dk ∈ Q，会累积到一个整体的有理数因子里。
        """
        key = (i, j)
        if key in self._mul_cache:
            return self._mul_cache[key]

        ei = self.exponents[i]
        ej = self.exponents[j]
        new_exp = []
        rational_factor = Fraction(1)
        for e_i, e_j, d in zip(ei, ej, self.ds):
            s = e_i + e_j
            # s = 0 or 1 or 2
            if s >= 2:
                # sqrt(d)^2 = d
                rational_factor *= d
                s -= 2
            new_exp.append(s)
        new_exp = tuple(new_exp)
        # 找到新基的索引
        k = self.exponents.index(new_exp)
        res = self.zero()
        res[k] = rational_factor
        self._mul_cache[key] = res
        return res

    def add(self, a: List[Fraction], b: List[Fraction]) -> List[Fraction]:
        return [ai + bi for ai, bi in zip(a, b)]

    def sub(self, a: List[Fraction], b: List[Fraction]) -> List[Fraction]:
        return [ai - bi for ai, bi in zip(a, b)]

    def scalar_mul(self, c: Fraction, a: List[Fraction]) -> List[Fraction]:
        return [c * ai for ai in a]

    def mul(self, a: List[Fraction], b: List[Fraction]) -> List[Fraction]:
        """
        一般向量乘法：利用基乘法展开
        """
        res = self.zero()
        for i, ai in enumerate(a):
            if ai == 0:
                continue
            for j, bj in enumerate(b):
                if bj == 0:
                    continue
                part = self._mul_basis(i, j)
                # res += ai * bj * part
                for k in range(self.dim):
                    res[k] += ai * bj * part[k]
        return res

    def equal(self, a: List[Fraction], b: List[Fraction]) -> bool:
        return all(ai == bi for ai, bi in zip(a, b))

    def from_rational(self, q: Fraction) -> List[Fraction]:
        v = self.zero()
        v[0] = q
        return v

    def to_float(self, a: List[Fraction]) -> float:
        """
        仅用于可视化/输出（不是几何判定）。
        这里用浮点近似：把 sqrt(di) 视为标准实值。
        """
        val = 0.0
        for coeff, exp in zip(a, self.exponents):
            factor = float(coeff)
            for e, d in zip(exp, self.ds):
                if e == 1:
                    factor *= float(np.sqrt(d))
            val += factor
        return val


class AlgebraicComplex:
    """
    把 (x,y) 看作 x + i y，其中 x,y 均在同一 AlgebraicField 中。
    用一对向量表示实部/虚部。
    """

    def __init__(self, field: AlgebraicField, real=None, imag=None):
        self.K = field
        self.real = real if real is not None else self.K.zero()
        self.imag = imag if imag is not None else self.K.zero()

    def __eq__(self, other: "AlgebraicComplex") -> bool:
        return self.K.equal(self.real, other.real) and self.K.equal(self.imag, other.imag)

    def __hash__(self) -> int:
        # 将 real 和 imag 转换为不可变的元组形式以用于哈希
        real_tuple = tuple(self.real)
        imag_tuple = tuple(self.imag)
        return hash((real_tuple, imag_tuple))

    @classmethod
    def from_rationals(cls, field: AlgebraicField, x: Fraction, y: Fraction):
        return cls(field, field.from_rational(x), field.from_rational(y))

    def add(self, other: "AlgebraicComplex") -> "AlgebraicComplex":
        return AlgebraicComplex(
            self.K,
            self.K.add(self.real, other.real),
            self.K.add(self.imag, other.imag),
        )

    def sub(self, other: "AlgebraicComplex") -> "AlgebraicComplex":
        return AlgebraicComplex(
            self.K,
            self.K.sub(self.real, other.real),
            self.K.sub(self.imag, other.imag),
        )

    def mul(self, other: "AlgebraicComplex") -> "AlgebraicComplex":
        # (x+iy)(u+iv) = (xu - yv) + i(xv + yu)
        xu = self.K.mul(self.real, other.real)
        yv = self.K.mul(self.imag, other.imag)
        xv = self.K.mul(self.real, other.imag)
        yu = self.K.mul(self.imag, other.real)
        real = self.K.sub(xu, yv)
        imag = self.K.add(xv, yu)
        return AlgebraicComplex(self.K, real, imag)

    def conj(self) -> "AlgebraicComplex":
        return AlgebraicComplex(self.K, self.real, self.K.scalar_mul(Fraction(-1), self.imag))

    def abs2(self) -> List[Fraction]:
        """
        返回 |z|^2 在 K 中的表示。
        |z|^2 = z * conj(z) 的实部。
        """
        zc = self.conj()
        prod = self.mul(zc)
        return prod.real

    def to_float_pair(self) -> Tuple[float, float]:
        return self.K.to_float(self.real), self.K.to_float(self.imag)


class AlgebraicUDGBuilder:
    """
    基于 AlgebraicField 的精确 UDGBuilder。
    对外接口尽量与原 UDGBuilder 类似：
        - add_points: 接受实数坐标，并嵌入到代数域中（只用有理数时就是 Q）
        - rotate(angle_element): angle_element 是 AlgebraicComplex，代表 unit-norm 旋转元
        - rotate_and_copy(...)
        - get_graph / compute_edges 等
    """

    def __init__(self, square_roots: List[int], tolerance: float = 1e-9):
        """
        square_roots: 例如 [] -> Q, [3] -> Q(√3), [3, 5] -> Q(√3, √5)
        tolerance: 仅用于 KDTree 近似筛选候选边对，判定是否距离为1仍用精确代数。
        """
        self.field = AlgebraicField(square_roots)
        self.tolerance = tolerance
        self.points: List[AlgebraicComplex] = []
        self.edges = set()
        # 常量 1 在 K 中的表示
        self.one = self.field.one()

    # -------- 内部辅助 ------------

    def _embed_float_point(self, x: float, y: float) -> AlgebraicComplex:
        """
        简单做法：把 x,y 都当作有理数近似嵌入（这里先用 Fraction(x).limit_denominator）。
        如果你希望 100% 精确，可以只在几何构造中使用有理数和 sqrt(di) 的线性组合，
        而不要从任意浮点反推。
        """
        fx = Fraction(x).limit_denominator()
        fy = Fraction(y).limit_denominator()
        return AlgebraicComplex.from_rationals(self.field, fx, fy)

    def _distance_is_one(self, i: int, j: int) -> bool:
        """
        用代数方式判定两点是否单位距离：
            |z_i - z_j|^2 == 1
        """
        zi = self.points[i]
        zj = self.points[j]
        diff = zi.sub(zj)
        abs2 = diff.abs2()  # in K
        return self.field.equal(abs2, self.one)

    # -------- 对外接口 ------------

    def add_points(self, points: np.ndarray):
        """
        与原 UDGBuilder 接口兼容：接受一个 (N,2) 的 float 数组。
        这里会把这些点嵌入到代数域中（近似为有理数），
        然后做去重（完全代数判定）。
        """
        if len(points) == 0:
            return
        new_pts = [self._embed_float_point(x, y) for x, y in points]

        # 使用哈希集合高效去重
        point_set = set(self.points)
        for p in new_pts:
            if p not in point_set:
                point_set.add(p)
        # 更新点列表
        self.points = list(point_set)
        # 更新边
        self.compute_edges()

    def add_algebraic_point(self, real: List[Fraction], imag: List[Fraction]):
        """
        直接添加一个代数域坐标的点。
        real: 点的实部在代数域中的表示（系数向量）
        imag: 点的虚部在代数域中的表示（系数向量）
        """
        new_point = AlgebraicComplex(self.field, real, imag)
        
        # 使用哈希集合高效去重
        point_set = set(self.points)
        if new_point not in point_set:
            point_set.add(new_point)
            # 更新点列表
            self.points = list(point_set)
            # 更新边
            self.compute_edges()

    def add_algebraic_points(self, points: List[AlgebraicComplex]):
        """
        直接添加多个代数域坐标的点。
        points: AlgebraicComplex 点的列表
        """
        if not points:
            return
            
        # 使用哈希集合高效去重
        point_set = set(self.points)
        added_new = False
        for new_point in points:
            if new_point not in point_set:
                point_set.add(new_point)
                added_new = True
        
        if added_new:
            # 更新点列表
            self.points = list(point_set)
            # 更新边
            self.compute_edges()

    def compute_edges(self):
        """
        先用 float 近似坐标建 KDTree 找候选点对，再用代数测试剔除非 1 距离。
        """
        if not self.points:
            self.edges = set()
            return
        coords = np.array([p.to_float_pair() for p in self.points])
        tree = KDTree(coords)
        # 先找距离 <= 1 + tol 的对
        candidate_pairs = tree.query_pairs(r=1+self.tolerance)
        edges = set()
        for i, j in candidate_pairs:
            # 用代数精确判定
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist >= 1.0 - self.tolerance:
                if self._distance_is_one(i, j):
                    edges.add(tuple(sorted((i, j))))
        self.edges = edges

    def get_graph(self) -> nx.Graph:
        G = nx.Graph()
        coords = [p.to_float_pair() for p in self.points]
        for i, (x, y) in enumerate(coords):
            G.add_node(i, pos=(x, y))
        G.add_edges_from(self.edges)
        return G

    def prune_to_size(self, n: int):
        """
        通过重复删去度数最小的点，并在图不连通时删去平均度数更小的连通块，将图的点数降到 n 以下。
        
        Args:
            n: 目标最大点数
        """
        if len(self.points) <= n:
            return
        
        # 获取当前的图结构
        G = self.get_graph()
        
        if G.number_of_nodes() == 0:
            return
        
        # 标记要删除的节点
        to_remove = set()
        current_size = len(self.points)
        
        # 复制图进行操作
        working_graph = G.copy()
        
        while current_size - len(to_remove) > n:
            # 检查图是否连通
            if nx.is_connected(working_graph):
                # 图连通，找到度数最小的点
                degrees = dict(working_graph.degree())
                if not degrees:
                    break
                
                # 找到度数最小的点
                min_degree = min(degrees.values())
                min_degree_nodes = [node for node, d in degrees.items() if d == min_degree]
                
                # 标记要删除的点
                for node in min_degree_nodes:
                    to_remove.add(node)
                
                # 从工作图中删除这些点
                working_graph.remove_nodes_from(min_degree_nodes)
            else:
                # 图不连通，找到平均度数更小的连通块
                components = list(nx.connected_components(working_graph))
                
                # 计算每个连通块的平均度数
                component_stats = []
                for component in components:
                    subgraph = working_graph.subgraph(component)
                    num_nodes = subgraph.number_of_nodes()
                    num_edges = subgraph.number_of_edges()
                    avg_degree = 2 * num_edges / num_nodes if num_nodes > 0 else 0
                    component_stats.append((avg_degree, num_nodes, component))
                
                # 找到平均度数最小的连通块（如果有多个，选择节点数最少的）
                component_stats.sort(key=lambda x: (x[0], x[1]))
                component_to_remove = component_stats[0][2]
                
                # 标记要删除的点
                for node in component_to_remove:
                    to_remove.add(node)
                
                # 从工作图中删除这些点
                working_graph.remove_nodes_from(component_to_remove)
            
            # 如果工作图为空，停止
            if working_graph.number_of_nodes() == 0:
                break
        
        # 如果没有要删除的点，直接返回
        if not to_remove:
            return
        
        # 更新点列表，排除要删除的点
        new_points = []
        for i, point in enumerate(self.points):
            if i not in to_remove:
                new_points.append(point)
        
        # 更新 points 并调用一次 compute_edges
        self.points = new_points
        self.compute_edges()

    # -------- 旋转相关 ------------

    def rotate(self, rot: AlgebraicComplex, pivot: Tuple[float, float] = (0.0, 0.0)):
        """
        与原 UDGBuilder.rotate 类似，但旋转参数不再是浮点角度，而是一个 unit-norm 的
        AlgebraicComplex 元 rot (|rot|=1)，代表复平面中的旋转。
        pivot 用普通浮点表示，会被嵌入代数域。
        """
        if not self.points:
            return
        pv = self._embed_float_point(*pivot)
        new_points = []
        for z in self.points:
            # z' = rot * (z - pivot) + pivot
            z_shift = z.sub(pv)
            z_rot = rot.mul(z_shift)
            z_new = z_rot.add(pv)
            new_points.append(z_new)
        self.points = new_points
        self.compute_edges()

    def rotate_and_copy(self, rot: AlgebraicComplex, pivot: Optional[Union[Tuple[float, float], AlgebraicComplex]] = None):
        """
        与原 UDGBuilder.rotate_and_copy 类似：在原图基础上添加一个旋转后的拷贝。
        
        pivot: 旋转中心，可以是浮点坐标元组或直接是 AlgebraicComplex 点
               - 如果为 None，默认使用原点 (0, 0)
               - 如果为浮点坐标元组，会被嵌入到代数域中
               - 如果为 AlgebraicComplex，直接使用该点作为旋转中心
        """
        if not self.points:
            return
            
        # 处理旋转中心
        if pivot is None:
            # 默认使用原点
            pv = AlgebraicComplex.from_rationals(self.field, Fraction(0), Fraction(0))
        elif isinstance(pivot, AlgebraicComplex):
            # 直接使用代数域坐标点
            pv = pivot
        else:
            # 浮点坐标需要嵌入到代数域
            pv = self._embed_float_point(*pivot)
            
        new_points = []
        for z in self.points:
            z_shift = z.sub(pv)
            z_rot = rot.mul(z_shift)
            z_new = z_rot.add(pv)
            new_points.append(z_new)

        # 合并到现有点集中（带去重）
        self.add_algebraic_points(new_points)  # 直接使用代数点，避免float转换损失

    # -------- 示例构造：Moser Spindle（用 √3） --------

    def add_moser_spindle(self, origin=(0.0, 0.0)):
        """
        一个使用 √3 的简单 Moser Spindle 示例（这里依然用标准正三角形坐标，
        然后嵌入 Q(√3)，保证所有距离如 √3 等都是精确代数表示）。
        如果你在构造时完全用 Fraction 和 sqrt(3) 的线性组合，就可以避免
        从 float 反推。
        """
        def halve(a):
            return self.field.scalar_mul(Fraction(1, 2), a)
        def mhalve(a):
            return self.field.scalar_mul(Fraction(-1, 2), a)
        s3 = self.field.get_root(3)
        self.add_algebraic_points([
            AlgebraicComplex.from_rationals(self.field, Fraction(0), Fraction(0)),
            AlgebraicComplex(self.field, real=halve(self.field.one), imag=halve(s3)),
            AlgebraicComplex(self.field, real=mhalve(self.field.one), imag=halve(s3)),
            AlgebraicComplex(self.field, real=self.field.zero, imag=s3),
        ])
        rot = AlgebraicComplex(self.field, 
                               real=self.field.scalar_mul(Fraction(5, 6), self.field.one),
                               imag=self.field.scalar_mul(Fraction(1, 6), self.field.get_root(11)))
        self.rotate_and_copy(rot, pivot=None)
        
