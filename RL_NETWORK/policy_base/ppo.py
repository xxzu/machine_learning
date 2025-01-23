import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 超参数
LEARNING_RATE = 0.0003
GAMMA = 0.99
LAMBDA = 0.95
EPS_CLIP = 0.2
K_EPOCH = 3
T_HORIZON = 20

# 定义 Actor 网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim),
            nn.Softmax(dim=-1) # 输出动作概率
        )

    def forward(self, state):
        return self.fc(state)

# 定义 Critic 网络
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1) # 输出状态价值
        )

    def forward(self, state):
        return self.fc(state)

def train_ppo(env_name="CartPole-v1", n_episodes=500):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim)
    actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
    critic_optimizer = optim.Adam(critic.parameters(), lr=LEARNING_RATE)

    for episode in range(n_episodes):
        state,info = env.reset()
        states, actions, rewards = [], [], []
        for t in range(T_HORIZON):
            state = torch.tensor(state, dtype=torch.float32)
            action_prob = actor(state)
            action = np.random.choice(action_dim, p=action_prob.detach().numpy())
            next_state, reward, done, _ ,_= env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            if done:
                break

        # 计算 GAE 和 TD(λ) 目标值
        values = critic(torch.stack(states)).squeeze()
        returns = []
        advantage = 0
        return_ = 0
        for i in reversed(range(len(rewards))):
            return_ = rewards[i] + GAMMA * return_ if i < len(rewards) - 1 else rewards[i]
            delta = rewards[i] + GAMMA * values[i+1] - values[i] if i < len(rewards) -1 else rewards[i] - values[i]
            advantage = delta + GAMMA * LAMBDA * advantage
            returns.insert(0, return_)

        returns = torch.tensor(returns, dtype=torch.float32)
        advantage = torch.tensor(advantage, dtype=torch.float32)

        # PPO 更新
        for _ in range(K_EPOCH):
            action_probs = actor(torch.stack(states))
            old_action_probs = actor(torch.stack(states)).detach()
            ratios = action_probs[torch.arange(len(actions)), actions] / old_action_probs[torch.arange(len(actions)), actions]
            clipped_ratios = torch.clamp(ratios, 1 - EPS_CLIP, 1 + EPS_CLIP)
            actor_loss = -torch.min(ratios * advantage, clipped_ratios * advantage).mean()
            critic_loss = (returns - critic(torch.stack(states)).squeeze()).pow(2).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
        print(f"Episode: {episode+1}, Reward: {sum(rewards)}")

    env.close()

if __name__ == "__main__":
    train_ppo()