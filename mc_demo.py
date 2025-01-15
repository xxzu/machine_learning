import numpy as np
"""蒙特卡洛计算方法"""
values = {  }
visit_count = {}
gamma = 0.9

episodes = [
    ("A",3),
    ("B",2),
    ("C",1)
]


# 反向计算回报 G
# 减少重复计算的次数
returns = []
G = 0 
for state,value in reversed(episodes):
    G  = G + gamma * value
    returns.insert(0,(state,G))
    
#计算价值V(s)←V(s)+ 1/n(G−V(s))
for state,G in returns:
    if state not in values:
        values[state] = 0
        visit_count[state] =0
        
    visit_count[state] += 1
    values[state] += (G- values[state] ) / visit_count[state]

print("最后结果",values)