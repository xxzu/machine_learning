'''
∇θ J(θ) = E[∇θ logπ_θ(a|s) ⋅ Qπ(s, a)]
'''
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 定义策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)  # 输出动作的概率分布
        return x

# REINFORCE算法
class REINFORCEAgent:
    def __init__(self, state_size, action_size, learning_rate=0.01):
        self.policy_network = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.gamma = 0.99  # 折扣因子

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)  # 添加批量维度
        probs = self.policy_network(state)
        # 采样动作:用于从给定的概率分布中进行采样。它返回从概率分布中根据给定的样本数选择的索引。
        action = torch.multinomial(probs, num_samples=1).item()  
        return action, probs[0, action]

    def compute_discounted_rewards(self, rewards):
        discounted_rewards = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            discounted_rewards.insert(0, G)
        return discounted_rewards

    def update_policy(self, log_probs, rewards):
        # Step 1: 计算 Q(s, a) 的近似值 (累计折扣奖励 G_t)
        discounted_rewards = self.compute_discounted_rewards(rewards)
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        
        # 标准化奖励，减小方差，提升稳定性
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        
        # Step 2: 计算策略梯度的损失 (对应公式 ∇θ J(θ) = E[∇θ logπ_θ(a|s) ⋅ Qπ(s, a)])
        loss = 0
        for log_prob, reward in zip(log_probs, discounted_rewards):
            loss -= log_prob * reward  # REINFORCE公式  # 对应 ∇θ logπ_θ(a|s) ⋅ Qπ(s, a)
        
        # Step 3: 通过反向传播更新参数
        self.optimizer.zero_grad()# 梯度清零
        loss.backward() # 计算梯度
        self.optimizer.step()# 更新策略参数 θ

# 主程序
if __name__ == "__main__":
    env = gym.make("CartPole-v1",render_mode='human')
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = REINFORCEAgent(state_size, action_size)

    episodes = 500
    for episode in range(episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        total_reward = 0

        while True:
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            log_probs.append(torch.log(log_prob))  # 保存对数概率
            rewards.append(reward)
            total_reward += reward
            state = next_state

            if terminated or truncated:
                break
        
        # 更新策略
        agent.update_policy(log_probs, rewards)
        print(f"Episode: {episode+1}, Total Reward: {total_reward}")
    torch.save(agent.policy_network.state_dict(),"/home/ubuntu/machine_learning/policy_base/pg.pth")
    env.close()

