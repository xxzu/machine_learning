import gym
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from policy_base.policygradient import PolicyNetwork,REINFORCEAgent

import torch

import gym

if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode='human')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = REINFORCEAgent(state_size,action_size)
    
    agent.policy_network.load_state_dict(torch.load('/home/ubuntu/machine_learning/policy_base/pg.pth'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = agent.policy_network.to(device=device)
    
    episodes = 5 # 玩几局游戏
    for e in range(episodes):
        state, _ = env.reset()
        
        # state = np.reshape(state, [state_size])
        
        total_reward = 0
        done = False
        
        
        while not done:
            env.render()  # 显示游戏画面

            # 智能体选择动作（只依赖模型，不使用 epsilon-greedy 策略）
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)#.unsqueeze(0) 扩张行维度 【1，2】 变为【【1，2】】
            
            with torch.no_grad():
                
                actions = model(state_tensor)
            action = torch.argmax(actions).item()

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            total_reward += reward
            done = terminated or truncated

        print(f"Episode: {e+1}/{episodes}, Total Reward: {total_reward}")
    print('done')

    env.close()
    
    