import torch
import torch.nn as nn
import torch.optim as optim
import gym

# 定义 Actor 和 Critic 网络
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)

class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# 定义 Actor-Critic 智能体
class ActorCriticAgent:
    def __init__(self, state_size, action_size, lr=0.001):
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.actor(state)
        action = torch.distributions.Categorical(probs).sample()
        return action.item(), probs[:, action.item()]
    
    def train(self, state, action_prob, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        reward = torch.FloatTensor([reward])
        done = torch.FloatTensor([done])

        # 计算 TD 误差
        value = self.critic(state)
        next_value = self.critic(next_state)
        target = reward + (1 - done) * 0.99 * next_value
        td_error = target - value

        # 更新 Critic
        critic_loss = td_error.pow(2).mean()
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # 更新 Actor
        actor_loss = -torch.log(action_prob) * td_error.detach()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

# 主程序
if __name__ == "__main__":
    # env = gym.make("CartPole-v1",render_mode='human')
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = ActorCriticAgent(state_size, action_size)

    for episode in range(500):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action, action_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.train(state, action_prob, reward, next_state, done)

            state = next_state
            total_reward += reward

        print(f"Episode {episode+1}, Total Reward: {total_reward}")
    
    torch.save(agent.actor.state_dict(),'/home/ubuntu/machine_learning/policy_base/ac.pth')
    env.close()
