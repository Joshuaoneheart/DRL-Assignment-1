from model import DQN
import torch
from tqdm import tqdm
import numpy as np
from simple_custom_taxi_env import SimpleTaxiEnv
policy = DQN(127, 6, 0.99, 512, 0.0003, "cuda")
env_config = {
    "fuel_limit": 5000,
    "grid_size": 10
}
policy.to("cuda")
policy.load()
env = SimpleTaxiEnv(**env_config)
total_reward = 0

rewards = []
su_cnt = 0
for _ in range(100):
    total_reward = 0
    done = False
    obs, _ = env.reset()
    state, goal = policy.get_state_and_goal(obs, False)
    step = 0
    while not done:
        action = policy.get_action(state, 0)

        obs, reward, done, _ = env.step(action)
        # print(state, action, reward, policy.reward_shaping(reward, state, action), policy.has_passenger)
        state, goal = policy.get_state_and_goal(obs, done)
        # env.render_env((obs[0], obs[1]))
        total_reward += reward
        step += 1
        if step > 200:
            break
    su_cnt += done
    rewards.append(total_reward)
print(su_cnt)
print(f"Total Reward: {np.mean(rewards[-100:])}")
