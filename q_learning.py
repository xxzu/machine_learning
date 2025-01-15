#异步策略的td算法
#公式 Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])

import numpy as np
import random

# 定义迷宫环境
maze = [
    ['S', '.', '.', 'W'],
    ['.', 'W', 'W', 'G'],
    ['.', '.', 'W', '.'],
    ['W', '.', '.', '.']
]

maze_size = (len(maze), len(maze[0]))  # 迷宫大小
action_space = 4  # 动作空间 [上, 下, 左, 右]
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 动作对应的坐标变化
start = (0, 0)  # 起点
goal = (1, 3)   # 终点

# 初始化 Q 表
Q = np.zeros((*maze_size, action_space))

# 参数设置
alpha = 0.1      # 学习率
gamma = 0.9      # 折扣因子
epsilon = 0.2    # 探索率
episodes = 500   # 总回合数

# 判断是否越界或撞墙
def is_valid(state):
    x, y = state
    return 0 <= x < maze_size[0] and 0 <= y < maze_size[1] and maze[x][y] != 'W'

# 随机选择动作
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(action_space))  # 随机选择动作
    return np.argmax(Q[state[0], state[1], :])     # 选择当前最优动作

# 环境交互
def step(state, action):
    next_state = (state[0] + actions[action][0], state[1] + actions[action][1])
    if not is_valid(next_state):  # 如果撞墙或越界，停留在原地
        next_state = state
    reward = 1 if next_state == goal else -0.1  # 到达终点得 1 分，否则惩罚 -0.1
    done = next_state == goal
    return next_state, reward, done

# Q-Learning 训练过程
for episode in range(episodes):
    state = start  # 每次从起点开始
    done = False

    while not done:
        action = choose_action(state)  # 选择动作
        next_state, reward, done = step(state, action)  # 执行动作
        # 更新 Q 值
        Q[state[0], state[1], action] += alpha * (
            reward + gamma * np.max(Q[next_state[0], next_state[1], :]) - Q[state[0], state[1], action]
        )
        state = next_state  # 更新状态

#通过训练好的 Q 表找到最优路径
def find_path(start):
    path = [start]
    state = start
    while state != goal:
        action = np.argmax(Q[state[0], state[1], :])
        state = (state[0] + actions[action][0], state[1] + actions[action][1])
        path.append(state)
        if len(path) > 50:  # 防止陷入死循环
            break
    return path
def visualize_path(maze, path):
    # 复制迷宫用于标记路径
    visual_maze = [row[:] for row in maze]

    # 用箭头标记路径
    directions = {
        (-1, 0): '↑',  # 上
        (1, 0): '↓',   # 下
        (0, -1): '←',  # 左
        (0, 1): '→'    # 右
    }

    for i in range(len(path) - 1):
        current = path[i]
        next_ = path[i + 1]
        direction = (next_[0] - current[0], next_[1] - current[1])
        visual_maze[current[0]][current[1]] = directions[direction]

    # 保留起点和终点标记
    visual_maze[start[0]][start[1]] = 'S'
    visual_maze[goal[0]][goal[1]] = 'G'

    # 打印迷宫
    for row in visual_maze:
        print(' '.join(row))


# 调用函数进行路径可视化
print("\nVisualized Maze with Optimal Path:")



# 输出结果
optimal_path = find_path(start)

print("Optimal Path:", optimal_path)
visualize_path(maze, optimal_path)
