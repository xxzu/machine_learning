'''sarsa算法 同策略的，当前动作和价值评估没适用于下一步的动作和价值
公式为  Q[s, a] = Q[s, a] + alpha * (R + gamma * Q[s_next, a_next] - Q[s, a])

'''

import numpy as np
import random

# 初始化参数
alpha = 0.1  # 学习率
gamma = 0.9  # 折扣因子
epsilon = 0.1  # 探索概率
num_episodes = 500  # 回合数
grid_size = (5, 5)  # 网格大小

# 初始化 Q 值表
Q = np.zeros((grid_size[0], grid_size[1], 4))  # 每个状态的四个动作值

# 定义动作 [上, 下, 左, 右]
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# 选择动作的 epsilon-greedy 策略
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(4))  # 随机选择动作
    else:
        return np.argmax(Q[state[0], state[1], :])  # 选择最大 Q 值的动作

# 更新状态
def take_action(state, action):
    next_state = (state[0] + actions[action][0], state[1] + actions[action][1])
    next_state = (max(0, min(next_state[0], grid_size[0] - 1)),  # 限制范围
                  max(0, min(next_state[1], grid_size[1] - 1)))
    reward = -1 if next_state != (grid_size[0] - 1, grid_size[1] - 1) else 0  # 到达目标状态奖励为 0
    done = next_state == (grid_size[0] - 1, grid_size[1] - 1)
    return next_state, reward, done

# SARSA 学习过程
for episode in range(num_episodes):
    state = (0, 0)  # 初始状态
    action = choose_action(state)  # 初始动作
    
    while True:
        next_state, reward, done = take_action(state, action)  # 执行动作
        next_action = choose_action(next_state)  # 下一动作
        # 更新 Q 值
        # Q[s, a] = Q[s, a] + alpha * (R + gamma * Q[s_next, a_next] - Q[s, a])
        Q[state[0], state[1], action] += alpha * (
            reward + gamma * Q[next_state[0], next_state[1], next_action] - Q[state[0], state[1], action]
        )
        state, action = next_state, next_action  # 更新状态和动作
        
        if done:  # 到达终止状态，结束回合
            break

# 打印最终 Q 值
print("最终的 Q 值表：")
print(Q)
