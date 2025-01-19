#td时序差分算法
#V(s)←V(s)+α⋅δ
#δt​=Rt+1 +γV(s′)−V(s) 
#从状态 𝑠 到下一状态 𝑠′获得的即时奖励。
# V(s′) 下一状态s'的价值估计


# 初始化状态价值
values = {"A": 0, "B": 0, "C": 0}  # 状态的初始价值
alpha = 0.1  # 学习率
gamma = 0.9  # 折扣因子

# 一个回合的轨迹
trajectory = [("A", 1, "B"), ("B", 2, "C"), ("C", 0, None)]
# 轨迹格式：(当前状态, 奖励, 下一状态)

# 更新状态价值
for state, reward, next_state in trajectory:
    if next_state is None:
        td_target = reward  # 终止状态的目标只包含即时奖励
    else:
        td_target = reward + gamma * values[next_state]  # TD目标  Rt+1 +γV(s′)

    td_error = td_target - values[state]  # 计算TD误差 δt​=Rt+1 +γV(s′)−V(s) 
    values[state] += alpha * td_error  # 更新价值函数 #V(s)←V(s)+α⋅δ

# 输出最终的状态价值
print("状态价值估计：", values)
