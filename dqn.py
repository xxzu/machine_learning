import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gym

# 定义 Q 网络
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义 DQN 智能体
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # 经验回放缓冲区
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_decay = 0.995  # 探索率衰减
        self.epsilon_min = 0.01  # 最小探索率
        self.learning_rate = 0.001  # 学习率
        self.batch_size = 32  # 批量大小
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU/CPU
        
        # 初始化主网络和目标网络
        self.q_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.update_target_network()

    def update_target_network(self):
        """将主网络的参数复制到目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """使用 ε-贪婪策略选择动作"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # 随机选择动作（探索）
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)
        return torch.argmax(q_values).item()  # 选择 Q 值最大的动作

    def replay(self):
        """从经验缓冲区中采样并训练"""
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 当前 Q 值
        q_values = self.q_network(states).gather(1, actions).squeeze(1)

        # 目标 Q 值
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        # 计算损失
        loss = nn.MSELoss()(q_values, target_q_values)

        # 更新网络参数
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 减少探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 主程序
if __name__ == "__main__":
    env = gym.make('CartPole-v1')  # 加载 CartPole 环境
    state_size = env.observation_space.shape[0]  # 状态空间维度
    action_size = env.action_space.n  # 动作空间维度
    agent = DQNAgent(state_size, action_size)

    episodes = 500  # 最大训练回合数
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])  # 调整状态形状
        total_reward = 0
        for time in range(500):  # 每回合最多运行 500 步
            env.render()  # 可视化环境
            action = agent.act(state)  # 使用 ε-贪婪策略选择动作
            next_state, reward, done, _ = env.step(action)  # 执行动作
            reward = reward if not done else -10  # 如果失败，给予惩罚
            next_state = np.reshape(next_state, [1, state_size])  # 调整下一个状态形状
            agent.remember(state, action, reward, next_state, done)  # 存储经验
            state = next_state
            total_reward += reward
            if done:
                agent.update_target_network()  # 每回合结束后更新目标网络
                print(f"Episode: {e+1}/{episodes}, Score: {time}, Epsilon: {agent.epsilon:.2}")
                break
            agent.replay()  # 训练模型

    env.close()
