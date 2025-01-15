import numpy as np
import random

# 定义迷宫环境
maze = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [0, 0, 0, 0, 0]
]

# 定义迷宫大小
maze_size = (5, 5)

# 起点和终点
start = (0, 0)
goal = (4, 4)

# 定义动作 [上, 下, 左, 右]
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
action_space = len(actions)
# 检查是否为有效位置
def is_valid(state):
    x, y = state
    return 0 <= x < maze_size[0] and 0 <= y < maze_size[1] and maze[x][y] == 0

# 根据动作获取下一个状态
def take_action(state, action):
    next_state = (state[0] + actions[action][0], state[1] + actions[action][1])
    if not is_valid(next_state):
        return state, -1, False  # 碰墙惩罚 -1，返回原状态
    reward = 0 if next_state != goal else 1  # 到达目标奖励 1，其余为 0
    done = next_state == goal
    return next_state, reward, done


# 初始化参数
alpha = 0.1  # 学习率
gamma = 0.9  # 折扣因子
epsilon = 0.1  # 探索概率
num_episodes = 50000  # 总训练回合

# 初始化 Q 表
Q = np.zeros((*maze_size, action_space)) # *(a,b) =a,b 解包
# 选择动作的 epsilon-greedy 策略
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(action_space))  # 随机选择动作
    return np.argmax(Q[state[0], state[1], :])  # 选择最大 Q 值的动作

# SARSA 主循环
for episode in range(num_episodes):
    state = start  # 初始化起点
    action = choose_action(state)  # 初始动作

    while True:
        next_state, reward, done = take_action(state, action)  # 执行动作
        next_action = choose_action(next_state)  # 根据策略选择下一动作

        # 更新 Q 值
        Q[state[0], state[1], action] += alpha * (
            reward + gamma * Q[next_state[0], next_state[1], next_action] - Q[state[0], state[1], action]
        )

        state, action = next_state, next_action  # 更新状态和动作

        if done:  # 到达目标状态，结束回合
            break
# 提取策略
policy = np.full(maze_size, ' ')
for i in range(maze_size[0]):
    for j in range(maze_size[1]):
        if maze[i][j] == 1:
            policy[i][j] = '#'  # 墙
        elif (i, j) == goal:
            policy[i][j] = 'G'  # 目标
        else:
            best_action = np.argmax(Q[i, j, :])
            if best_action == 0:
                policy[i][j] = '↑'
            elif best_action == 1:
                policy[i][j] = '↓'
            elif best_action == 2:
                policy[i][j] = '←'
            elif best_action == 3:
                policy[i][j] = '→'

# 打印策略
for row in policy:
    print(' '.join(row))
