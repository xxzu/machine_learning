a = [[3,45,5,6,7,778,87,6,4,3,2,4],
     [4, 2, 3, 4, 6, 87, 778, 7, 6, 5, 45, 3]
     ]
#切片学习
a_all = a[:]
# a_all[2] = 999
print(f'原始的参数{a},切片为a[:]的结果{a_all},修改了以后的{a_all}')
a_sub = a[1:5]
print(f'1到五的数据{a_sub}')
a_subsub = a[::3]
print(f"a[::3]的结果为{a_subsub}")
a_reverse = a[::-1] 
print(f"a[::-1]的结果为{a_reverse}")

import numpy as np

# 定义二维数组
a = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
# 选择第 1 列（索引从 0 开始）
column = a[:, 1]
print("第 1 列:", column)  # [2, 5, 8]

# 选择第 0 列到第 1 列
columns = a[:, 0:2]
print("第 0 到 1 列:\n", columns)
Q = np.random.randn(5, 5, 4)
a = Q[3,4,:]
print(Q)
print("q:",a)