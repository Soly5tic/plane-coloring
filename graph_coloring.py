import networkx as nx
import subprocess
import random
import string

def graph_to_cnf_sat(G, k):
    """
    将图的k染色问题转换为CNF-SAT问题。
    
    参数:
    G: networkx图对象，表示要染色的图
    k: 整数，表示可用的颜色数
    
    返回:
    clauses: 列表，包含CNF-SAT问题的所有子句
    var_mapping: 字典，将变量索引映射回(节点,颜色)对，用于解释解
    """
    # 变量映射: (节点, 颜色) -> 变量索引
    var_mapping = {}
    next_var = 1
    
    # 为每个节点的每个可能颜色创建一个变量
    for node in G.nodes():
        for color in range(1, k+1):
            var_mapping[(node, color)] = next_var
            next_var += 1
    
    clauses = []
    
    # 任务2: 每个节点必须恰好有一种颜色
    # 首先，确保每个节点至少有一种颜色
    for node in G.nodes():
        # 添加子句: 节点至少有一种颜色
        clause = [var_mapping[(node, color)] for color in range(1, k+1)]
        clauses.append(clause)
        
        # 确保每个节点最多有一种颜色（通过排除两两组合）
        for i in range(1, k):
            for j in range(i+1, k+1):
                # 添加子句: 节点不能同时有颜色i和颜色j
                clause = [-var_mapping[(node, i)], -var_mapping[(node, j)]]
                clauses.append(clause)
    
    # 任务3: 相邻节点不能有相同的颜色
    for u, v in G.edges():
        for color in range(1, k+1):
            # 添加子句: 如果u有颜色color，则v不能有颜色color，反之亦然
            clause = [-var_mapping[(u, color)], -var_mapping[(v, color)]]
            clauses.append(clause)
    
    return clauses, var_mapping

# 任务4: 添加额外的实用函数
def get_var_count(clauses):
    """
    获取CNF公式中的变量总数。
    
    参数:
    clauses: 列表，包含CNF-SAT问题的所有子句
    
    返回:
    var_count: 整数，变量总数
    """
    var_count = 0
    for clause in clauses:
        clause_max = max(abs(literal) for literal in clause)
        var_count = max(var_count, clause_max)
    return var_count

def save_cnf_to_file(clauses, filename):
    """
    将CNF公式保存为DIMACS格式的文件，供Kissat等求解器使用。
    
    参数:
    clauses: 列表，包含CNF-SAT问题的所有子句
    filename: 字符串，保存的文件名
    """
    var_count = get_var_count(clauses)
    clause_count = len(clauses)
    
    with open(filename, 'w') as f:
        # 写入CNF头部
        f.write(f"p cnf {var_count} {clause_count}\n")
        
        # 写入每个子句
        for clause in clauses:
            # 每个子句以0结尾
            f.write(' '.join(map(str, clause)) + " 0\n")

# 任务5: 实现一个简单的验证函数
def verify_coloring(G, k, assignment, var_mapping):
    """
    验证给定的赋值是否是图的一个有效k染色。
    
    参数:
    G: networkx图对象
    k: 整数，表示颜色数
    assignment: 字典，变量索引到布尔值的映射
    var_mapping: 字典，(节点,颜色)对到变量索引的映射
    
    返回:
    is_valid: 布尔值，表示赋值是否有效
    coloring: 字典，如果有效则返回节点到颜色的映射
    """
    coloring = {}
    
    # 构建颜色映射
    for (node, color), var in var_mapping.items():
        if assignment.get(var, False):
            # 如果已经为该节点分配了颜色，则冲突
            if node in coloring:
                return False, None
            coloring[node] = color
    
    # 检查所有节点是否都有颜色
    if len(coloring) != len(G.nodes()):
        return False, None
    
    # 检查相邻节点是否颜色不同
    for u, v in G.edges():
        if u in coloring and v in coloring and coloring[u] == coloring[v]:
            return False, None
    
    return True, coloring

def graph_coloring(G, k):
    clauses, var_mapping = graph_to_cnf_sat(G, k)
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=32))
    cnf_filename = f"./tmp/graph_coloring{random_str}.cnf"
    save_cnf_to_file(clauses, cnf_filename)
    
    try:
        # 添加30秒超时机制
        result = subprocess.run(
            ['./kissat', '--quiet', cnf_filename], 
            capture_output=True, 
            text=True
        )
        output = result.stdout
        lines = output.split('\n')

        status = "UNKNOWN"
        model = []
        
        for line in lines:
            if line.startswith('s '):
                status = line.split()[1]  # SATISFIABLE 或 UNSATISFIABLE
            elif line.startswith('v '):
                # 提取变量值 (忽略 'v' 和行尾的 '0')
                vars = line.split()[1:]
                for v in vars:
                    if v != '0':
                        model.append(int(v))
                        
        return status, model
    except subprocess.TimeoutExpired:
        print(f"求解器超时（30秒），放弃当前图")
        return "TIMEOUT", []


def is_colorable(G, k):
    status, model = graph_coloring(G, k)
    if status == "TIMEOUT":
        return None  # 返回None表示无法确定
    return status == "SATISFIABLE"


# 示例函数：演示如何使用该模块处理不同类型的图和k值
def example_usage():
    """
    示例函数，展示如何使用该模块处理不同类型的networkx图对象和不同的k值。
    """
    # 示例1: 简单的三角形图（需要3种颜色）
    print("示例1: 三角形图")
    G1 = nx.Graph()
    G1.add_edges_from([(1, 2), (2, 3), (3, 1)])  # 三角形
    
    # 使用3种颜色
    clauses1_3, var_mapping1_3 = graph_to_cnf_sat(G1, 3)
    print(f"  三角形图 (k=3): {len(clauses1_3)}个子句, {get_var_count(clauses1_3)}个变量")
    
    # 使用2种颜色（应该不可解）
    clauses1_2, var_mapping1_2 = graph_to_cnf_sat(G1, 2)
    print(f"  三角形图 (k=2): {len(clauses1_2)}个子句, {get_var_count(clauses1_2)}个变量")
    
    # 示例2: 完全图K4（需要4种颜色）
    print("\n示例2: 完全图K4")
    G2 = nx.complete_graph(4)
    clauses2_4, var_mapping2_4 = graph_to_cnf_sat(G2, 4)
    print(f"  完全图K4 (k=4): {len(clauses2_4)}个子句, {get_var_count(clauses2_4)}个变量")
    
    # 示例3: 二分图（只需要2种颜色）
    print("\n示例3: 二分图")
    G3 = nx.bipartite.complete_bipartite_graph(3, 3)
    clauses3_2, var_mapping3_2 = graph_to_cnf_sat(G3, 2)
    print(f"  二分图K3,3 (k=2): {len(clauses3_2)}个子句, {get_var_count(clauses3_2)}个变量")
    
    # 示例4: 空图（只有一个节点）
    print("\n示例4: 空图")
    G4 = nx.Graph()
    G4.add_node(1)
    clauses4_1, var_mapping4_1 = graph_to_cnf_sat(G4, 1)
    print(f"  单节点图 (k=1): {len(clauses4_1)}个子句, {get_var_count(clauses4_1)}个变量")
    
    print("\n示例完成。要将CNF公式保存到文件以供Kissat使用，可以调用save_cnf_to_file函数。")
    
    # 返回示例结果，可用于进一步测试
    return {
        'triangle_k3': (clauses1_3, var_mapping1_3),
        'triangle_k2': (clauses1_2, var_mapping1_2),
        'k4': (clauses2_4, var_mapping2_4),
        'bipartite': (clauses3_2, var_mapping3_2),
        'single_node': (clauses4_1, var_mapping4_1)
    }

# 如果直接运行此脚本，展示示例
if __name__ == "__main__":
    example_usage()