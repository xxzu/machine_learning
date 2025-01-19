import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



from statistics import StatisticsError
import torch
import numpy as np
import gym
from value_base.dqn_demo import DQNAgent  

if __name__ == "__main__":
    # 创建环境
    # env = gym.make('CartPole-v1')
    env = gym.make('CartPole-v1', render_mode='human')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # 初始化智能体
    agent = DQNAgent(state_size, action_size)

    # 加载模型
    agent.q_network.load_state_dict(torch.load("/home/ubuntu/machine_learning/value_base/dqn_q_network.pth"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = agent.q_network.to(device)  # 将模型转移到指定设备
    # agent.q_network.eval()  # 切换到评估模式，不需要梯度计算
    # model = model.eval()
    print('device' ,device)
    # 玩游戏
    episodes = 100  # 玩几局游戏
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
                q_values = agent.q_network(state_tensor)
                q = model(state_tensor)
            action = torch.argmax(q).item()

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            total_reward += reward
            done = terminated or truncated

        print(f"Episode: {e+1}/{episodes}, Total Reward: {total_reward}")

    env.close()
