import numpy as np
import math

def generate_rational_angles(limit=10):
    """
    生成基于勾股数的有理旋转角度库 (cos, sin 均为有理数)。
    limit: 搜索勾股数的范围 c <= limit
    """
    angles = []
    
    # 1. 勾股数角度
    for c in range(1, limit + 1):
        for a in range(1, c):
            theta = np.arccos(a / c)
            angles.append(theta)
            angles.append(-theta)
    
    # 去重并排序
    unique_angles = sorted(list(set(angles)))
    
    # 移除 0
    if 0 in unique_angles:
        unique_angles.remove(0)
        
    return unique_angles



