'''DPG 的核心思想
策略网络（Deterministic Policy）：确定性策略意味着，给定一个状态 
𝑠，策略网络 𝜋𝜃(𝑠)  会直接输出一个 确定的动作a =𝜋𝜃(𝑠)而不是动作的概率分布。

确定性策略梯度：DPG计算策略梯度，但是与传统的概率策略梯度不同，DPG 计算的是确定性策略梯度。它通过Q网络 估计策略的动作值，
并且通过计算确定性策略梯度来更新网络。

目标：DPG 的目标是最大化长期回报。它使用确定性策略，通过 Q 网络 来计算梯度，并且通过 确定性策略梯度 来更新参数。
'''
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym

# Q网络
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 策略网络
class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x  # Deterministic action

# DPG Agent
class DPGAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_network = QNetwork(state_size, action_size)
        self.actor = ActorNetwork(state_size, action_size)
        self.target_q_network = QNetwork(state_size, action_size)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(list(self.q_network.parameters()) + list(self.actor.parameters()), lr=0.001)
        
    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            action_values = self.actor(state)  # Deterministic action
        action = torch.argmax(action_values, dim=1).item() 
        return action

    # def update(self, state, action, reward, next_state, done):
    #     state = torch.FloatTensor(state).unsqueeze(0)
    #     next_state = torch.FloatTensor(next_state).unsqueeze(0)
        
    #     # Compute Q value using Q network
    #     q_value = self.q_network(state)[0, action]
    #     # Compute target Q value using target Q network
    #     target_q_value = reward + (1 - done) * self.target_q_network(next_state).max()

    #     # Update Q network (Bellman error)
    #     q_loss = nn.MSELoss()(q_value, target_q_value)
    #     self.optimizer.zero_grad()
    #     q_loss.backward()
    #     self.optimizer.step()

    #     # Update Actor network based on Q network's gradient
    #     # This part typically requires applying the policy gradient
    #     # Since here the actor network is trained through deterministic gradient descent
    #     self.optimizer.zero_grad()
    #     action_grad = torch.autograd.grad(q_value, self.actor.parameters())
    #     for param, grad in zip(self.actor.parameters(), action_grad):
    #         param.grad = grad
    #     self.optimizer.step()
    def update(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        
        # Compute Q value using Q network
        q_value = self.q_network(state)[0, action]
        # Compute target Q value using target Q network
        target_q_value = reward + (1 - done) * self.target_q_network(next_state).max()

        # Update Q network (Bellman error)
        q_loss = nn.MSELoss()(q_value, target_q_value)
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

        # Update Actor network based on Q network's gradient
        # Now we compute the gradient of Q with respect to the action, and backpropagate it to the Actor network.
        action_value = self.actor(state)
        action_grad = torch.autograd.grad(action_value, self.actor.parameters(), grad_outputs=torch.ones_like(action_value))
        
        # Apply the policy gradient step
        for param, grad in zip(self.actor.parameters(), action_grad):
            param.grad = grad
        self.optimizer.step()

if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode='human')
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DPGAgent(state_size, action_size)

    episodes = 500
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        while True:
            # Get action from agent
            action = agent.act(state)
            # Execute action in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            # Update the agent (Q network and actor network)
            agent.update(state, action, reward, next_state, terminated or truncated)
            
            state = next_state

            if terminated or truncated:
                break
        
        print(f"Episode: {episode+1}, Total Reward: {total_reward}")
    torch.save(agent.actor.state_dict(), "/home/ubuntu/machine_learning/policy_base/dpg_actor.pth")
    env.close()
