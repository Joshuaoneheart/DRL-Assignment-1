import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from collections import deque
from memory import Memory
from torch.distributions import Categorical
from utils import RunningMeanStd
class PPO(nn.Module):
    def __init__(self, state_size, action_size, gamma=0.99, batch_size=32, lr=1e-5, device="cpu"):
        super().__init__()
        self.body = nn.Sequential(
                nn.Linear(state_size, 512),
                nn.GELU(),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Linear(256, 256),
                nn.GELU()
                )
        self.actor = nn.Sequential(
                nn.Linear(256, action_size),
                nn.Softmax(dim=-1)
                )
        self.critic = nn.Linear(256, 1)
        self.rms = RunningMeanStd(shape=())
        self.embed = nn.Embedding(101, 20)
        self.device = device
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.batch_size = batch_size
        self.gamma = gamma
        self.memory = Memory(100000000, 4)
        self.criterion = nn.MSELoss()
        self.stations = [[0, 0], [0, 0], [0, 0], [0, 0]]

        # state related
        self.goal_cnt = 0
        self.has_passenger = False
        self.receive_reward = False
        self.prev_has_passenger = False
        self.prev_pos = deque([], 4)
        for _ in range(4):
            self.prev_pos.append([10, 0])
    
    def load(self):
        self.load_state_dict(torch.load("PPO.pkl", map_location=self.device))
        # self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def reward_shaping(self, reward, state, action, next_state):
        # if action == 5 and ((self.has_passenger and not([state[0], state[1]] in self.stations and state[-2])) or not self.has_passenger):
            # reward -= 20
        if action in [0, 1, 2, 3] and (state[0] != next_state[0] or state[1] != next_state[1]):
            if next_state[0] == state[10] and next_state[1] == state[11]:
                print("Get Goal")
                reward += 10 
            elif (state[2] == next_state[0] and state[3] == next_state[1]) or (state[4] == next_state[0] and state[5] == next_state[1]) or (state[6] == next_state[0] and state[7] == next_state[1]) or (state[8] == next_state[0] and state[9] == next_state[1]):
                reward -= 0
        if action == 5:
            self.has_passenger = False
        if action == 4 and state[-3] and state[0] == state[10] and state[1] == state[11] and not self.receive_reward:
            print("Get Passenger")
            reward += 50
            self.receive_reward = True
            # reward += 1000
        # elif action == 4:
            # reward -= 20
        self.prev_has_passenger = self.has_passenger
        return reward

    def reset(self):
        self.has_passenger = False

    def state_to_onehot(self, state):
        idx = state[:, 0] * 10 + state[:, 1]
        prev_idx = state[:, 2] * 10 + state[:, 3]
        prev_idx_2 = state[:, 4] * 10 + state[:, 5]
        prev_idx_3 = state[:, 6] * 10 + state[:, 7]
        prev_idx_4 = state[:, 8] * 10 + state[:, 9]
        idx_goal = state[:, 10] * 10 + state[:, 11]
        return torch.cat([self.embed(idx.long()), self.embed(prev_idx.long()), self.embed(prev_idx_2.long()), self.embed(prev_idx_3.long()), self.embed(prev_idx_4.long()), self.embed(idx_goal.long()), state[:, 12:]], dim=1)

    def get_action(self, state, action=None):
        """
        for i in range(10):
            for j in range(10):
                s = [i, j, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                d = self.policy_net(self.state_to_onehot(torch.as_tensor(s).float().to(self.device).unsqueeze(0)))[:4]
                d = d.argmax().item()
                if d == 0:
                elif d == 1:
                    print("/\\ ", end="")
                elif d == 2:
                    print(">  ", end="")
                elif d == 3:
                    print("<  ", end="")
            print()
        """

        if action == None:
            feat = self.body(self.state_to_onehot(torch.as_tensor(state).unsqueeze(0).to(self.device)))
        else:
            feat = self.body(self.state_to_onehot(state.to(self.device)))
        probs = self.actor(feat)
        done = False
        if action == None:
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            action = action.item()
            if action == 5 and self.has_passenger:
                done = True
            if action in [0, 1, 2, 3]:
                self.prev_pos.append([state[0], state[1]])
            if action == 4 and [state[0], state[1]] in self.stations and state[-3] and not self.has_passenger:
                self.has_passenger = True
        else:
            dist = Categorical(probs)
            log_prob = dist.log_prob(action)
        return action, log_prob, self.critic(feat), dist.entropy(), done

    def get_state_and_goal(self, obs, done):
        stations = [[0, 0], [0, 0], [0, 0], [0, 0]]
        taxi_row, taxi_col, stations[0][0], stations[0][1], stations[1][0], stations[1][1], stations[2][0], stations[2][1], stations[3][0], stations[3][1],obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs
        if not passenger_look or not all([i[0] == j[0] and i[1] == j[1] for (i, j) in zip(self.stations, stations)]):
            self.has_passenger = False
        if not all([i[0] == j[0] and i[1] == j[1] for (i, j) in zip(self.stations, stations)]):
            self.receive_reward = False
            for _ in range(4):
                self.prev_pos.append([10, 0])
        goal = stations[self.goal_cnt]
        self.stations = stations
        if [taxi_row, taxi_col] == goal:
            if not(passenger_look and not self.has_passenger) and not(destination_look and self.has_passenger):
                for _ in range(4):
                    self.prev_pos.append([10, 0])
                self.goal_cnt += 1
                self.goal_cnt %= 4
        if done:
            self.reset()
        prev_pos = list(self.prev_pos)
        return [taxi_row, taxi_col, prev_pos[0][0], prev_pos[0][1],prev_pos[1][0], prev_pos[1][1],prev_pos[2][0], prev_pos[2][1],prev_pos[3][0], prev_pos[3][1],goal[0], goal[1], obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look, self.has_passenger], goal

    def forward(self, x):
        return self.policy_net(x)

    def update(self, dicts):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(dicts["reward"]), reversed(dicts["done"])):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        self.rms.update(np.array(rewards))
        rewards = torch.as_tensor(rewards).to(self.device)
        rewards = rewards / (np.sqrt(self.rms.var) + 1e-8)
        old_states = torch.as_tensor(dicts["state"]).to(self.device)
        old_actions = torch.as_tensor(dicts["action"]).to(self.device)
        old_logprobs = torch.as_tensor(dicts["logprob"]).to(self.device).detach()
        old_state_values = torch.as_tensor(dicts["value"]).to(self.device)
        advantages = rewards.detach() - old_state_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        entropy = []
        value_loss = []
        for _ in range(10):
            _, logprobs, state_values, dist_entropy, _ = self.get_action(old_states, old_actions)

            ratios = torch.exp(logprobs - old_logprobs.detach())
            logratio = logprobs - old_logprobs.detach()
            approx_kl = ((ratios - 1) - logratio).mean()
            if approx_kl > 0.01:
                print(f"Early Stop due to approx_kl {approx_kl}")
                break
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 0.9, 1.1) * advantages
            v_l = self.criterion(state_values.squeeze(1), rewards)
            dist_entropy = dist_entropy.mean()
            loss = -torch.min(surr1, surr2) + v_l - 0.01 * dist_entropy
            entropy.append(dist_entropy.item())
            value_loss.append(v_l.item())
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        return np.mean(entropy), np.mean(value_loss)

