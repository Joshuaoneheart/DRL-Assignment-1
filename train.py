from model import DQN
import torch
from tqdm import tqdm
import numpy as np
from simple_custom_taxi_env import SimpleTaxiEnv
policy = DQN(27, 6, 0.99, 40960, 1e-4, "cuda")
env_config = {
    "fuel_limit": 5000,
    "grid_size": 10
}
policy.to("cuda")
policy.load()
env = SimpleTaxiEnv(**env_config)
total_reward = 0

rewards = []
shaped_rewards = []
epsilon = 0.2
total_losses = []
for episode in tqdm(range(25000)):
    total_reward = 0
    total_shaped_reward = 0
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
    losses = []
    step = 0
    while not done:
        action = policy.get_action(obs, epsilon)
        
        obs, reward, done, _ = env.step(action)
        next_state, next_goal = policy.get_state_and_goal(obs, done)
        total_reward += reward
        reward = policy.reward_shaping(reward, state, action)
        total_shaped_reward += reward
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
        if done:
            print("Goal!!")
        step += 1
        if step > 200:
            break
    total_losses.append(policy.update())
    policy.memory.add(epi_dict)
    epsilon = max(epsilon * 0.9999, 0.1)
    rewards.append(total_reward)
    shaped_rewards.append(total_shaped_reward)
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}, Total Reward: {np.mean(rewards[-100:])}, Epsilon: {epsilon}, Total Shaped Reward: {np.mean(shaped_rewards[-100:])}, Loss: {np.mean(total_losses[-100:])}")
        torch.save(policy.state_dict(), "model.pkl")
