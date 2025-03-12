from model import DQN
import torch
from tqdm import tqdm
import numpy as np
from simple_custom_taxi_env import SimpleTaxiEnv
policy = DQN(11, 6, 0.99, 128, 0.0001)
env_config = {
    "fuel_limit": 5000
}
policy.to("cuda")
env = SimpleTaxiEnv(**env_config)
total_reward = 0
stations = [(0, 0), (0, 4), (4, 0), (4,4)]

rewards = []
epsilon = 1.0
for episode in tqdm(range(5000)):
    total_reward = 0
    done = False
    obs, _ = env.reset()
    state, goal = policy.get_state_and_goal(obs, done)
    epi_dict = {
            "state": [],
            "desired_goal": [],
            "achieved_goal": [],
            "next_achieved_goal": [],
            "next_state": [],
            "action": [],
            "reward": [],
            "done": []
            }
    while not done:
        action = policy.get_action(obs, epsilon)

        obs, reward, done, _ = env.step(action)
        next_state, next_goal = policy.get_state_and_goal(obs, done)
        total_reward += reward
        reward = policy.reward_shaping(reward, state, action)
        epi_dict["state"].append(state)
        epi_dict["action"].append(action)
        epi_dict["reward"].append(reward)
        epi_dict["achieved_goal"].append((state[0], state[1]))
        epi_dict["next_state"].append(next_state)
        epi_dict["next_achieved_goal"].append((next_state[0], next_state[1]))
        epi_dict["desired_goal"].append(goal)
        epi_dict["done"].append(done)
        state = next_state
        goal = next_goal
    losses = []
    for _ in range(10):
        losses.append(policy.update())
    policy.memory.add(epi_dict)
    epsilon = max(epsilon * 0.9995, 0.1)
    rewards.append(total_reward)
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}, Total Reward: {np.mean(rewards[-100:])}, Epsilon: {epsilon}")
        torch.save(policy.policy_net.state_dict(), "model.pkl")
