#!/usr/bin/env python3
# Test script for the modified get_root function

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from integer_udg_builder import AlgebraicField
from fractions import Fraction


def test_get_root():
    print("Testing AlgebraicField.get_root function...")
    
    # 测试1: Q(√2)
    print("\n1. Testing Q(√2):")
    field = AlgebraicField([2])
    print(f"   Exponents: {field.exponents}")
    
    # 测试 n=2
    try:
        result = field.get_root(2)
        print(f"   √2 = {result}")
        # 对于 Q(√2)，exponents 是 [(0,), (1,)]，所以 √2 对应索引1
        assert result[1] == 1, f"Expected component 1 to be 1, got {result}"
        assert sum(1 for x in result if x != 0) == 1, f"Expected only one non-zero component, got {result}"
    except Exception as e:
        print(f"   Error testing √2: {e}")
    
    # 测试 n=8 (4*2)
    try:
        result = field.get_root(8)
        print(f"   √8 = {result}")
        # √8 = 2√2，所以应该是 [0, 2]
        assert result[1] == 2, f"Expected component 1 to be 2, got {result}"
    except Exception as e:
        print(f"   Error testing √8: {e}")
    
    # 测试2: Q(√2, √3)
    print("\n2. Testing Q(√2, √3):")
    field = AlgebraicField([2, 3])
    print(f"   Exponents: {field.exponents}")
    
    # 测试 n=2
    try:
        result = field.get_root(2)
        print(f"   √2 = {result}")
        # 对于 Q(√2, √3)，exponents 是 [(0,0), (0,1), (1,0), (1,1)]
        # √2 对应索引2 (指数 (1,0))
        assert result[2] == 1, f"Expected component 2 to be 1, got {result}"
    except Exception as e:
        print(f"   Error testing √2: {e}")
    
    # 测试 n=3
    try:
        result = field.get_root(3)
        print(f"   √3 = {result}")
        # √3 对应索引1 (指数 (0,1))
        assert result[1] == 1, f"Expected component 1 to be 1, got {result}"
    except Exception as e:
        print(f"   Error testing √3: {e}")
    
    # 测试 n=6 (2*3)
    try:
        result = field.get_root(6)
        print(f"   √6 = {result}")
        # √6 = √2*√3，对应索引3 (指数 (1,1))
        assert result[3] == 1, f"Expected component 3 to be 1, got {result}"
    except Exception as e:
        print(f"   Error testing √6: {e}")
    
    # 测试 n=24 (4*6)
    try:
        result = field.get_root(24)
        print(f"   √24 = {result}")
        # √24 = 2√6，对应索引3
        assert result[3] == 2, f"Expected component 3 to be 2, got {result}"
    except Exception as e:
        print(f"   Error testing √24: {e}")
    
    # 测试 n=18 (9*2)
    try:
        result = field.get_root(18)
        print(f"   √18 = {result}")
        # √18 = 3√2，对应索引2
        assert result[2] == 3, f"Expected component 2 to be 3, got {result}"
    except Exception as e:
        print(f"   Error testing √18: {e}")
    
    # 测试3: Q(√2, √3, √5)
    print("\n3. Testing Q(√2, √3, √5):")
    field = AlgebraicField([2, 3, 5])
    print(f"   Exponents: {field.exponents}")
    
    # 测试 n=30 (2*3*5)
    try:
        result = field.get_root(30)
        print(f"   √30 = {result}")
        # 在 Q(√2, √3, √5) 中，√30 = √2*√3*√5，对应指数 (1,1,1)，索引为7
        assert result[7] == 1, f"Expected component 7 to be 1, got {result}"
        assert sum(1 for x in result if x != 0) == 1, f"Expected only one non-zero component, got {result}"
    except Exception as e:
        print(f"   Error testing √30: {e}")
    
    print("\nAll tests completed!")


if __name__ == "__main__":
    test_get_root()
