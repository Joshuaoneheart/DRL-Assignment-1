import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
from memory import Memory
import torch.nn.functional as F
class DQN(nn.Module):
    def __init__(self, state_size, action_size, gamma=0.99, batch_size=32, lr=1e-5, device="cpu"):
        super().__init__()
        self.policy_net = nn.Sequential(
                nn.Linear(state_size, 512),
                nn.GELU(),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Linear(256, action_size)
                )
        self.embed = nn.Embedding(100, 10)
        self.target_net = nn.Sequential(
                nn.Linear(state_size, 512),
                nn.GELU(),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Linear(256, action_size)
                )
        self.tau = 0.005
        self.device = device
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = Memory(1500, 4)
        self.batch_size = batch_size
        self.gamma = gamma
        self.criterion = nn.MSELoss()
        self.update_freq = 1000
        self.num_update = 0
        self.stations = [[0, 0], [0, 0], [0, 0], [0, 0]]

        # state related
        self.goal_cnt = 0
        self.has_passenger = False
        self.prev_has_passenger = False
    
    def load(self):
        self.load_state_dict(torch.load("model.pkl", map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def reward_shaping(self, reward, state, action):
        if action == 5 and not([state[0], state[1]] in self.stations and state[-2]):
            reward -= 10
            self.has_passenger = False
        if self.prev_has_passenger != self.has_passenger and self.has_passenger and state[0] == state[2] and state[1] == state[3]:
            print("Get Passenger")
            reward += 10
        self.prev_has_passenger = self.has_passenger
        return reward

    def reset(self):
        self.has_passenger = False

    def state_to_onehot(self, state):
        idx = state[:, 0] * 10 + state[:, 1]
        idx_goal = state[:, 2] * 10 + state[:, 3]
        return torch.cat([self.embed(idx.long()), self.embed(idx_goal.long()), state[:, 4:]], dim=1)

    def get_action(self, obs, epsilon):
        state, goal = self.get_state_and_goal(obs, False)
        """
        for i in range(10):
            for j in range(10):
                s = [i, j, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                d = self.policy_net(self.state_to_onehot(torch.as_tensor(s).float().to(self.device).unsqueeze(0)))[:4]
                d = d.argmax().item()
                if d == 0:
                    print("\\/ ", end="")
                elif d == 1:
                    print("/\\ ", end="")
                elif d == 2:
                    print(">  ", end="")
                elif d == 3:
                    print("<  ", end="")
            print()
        """
        valid_action = []
        if not state[5]:
            valid_action.append(0)
        if not state[4]:
            valid_action.append(1)
        if not state[6]:
            valid_action.append(2)
        if not state[7]:
            valid_action.append(3)
        if [state[0], state[1]] in self.stations and state[-3] and not self.has_passenger:
            valid_action.append(4)
        if [state[0], state[1]] in self.stations and state[-2] and self.has_passenger:
            valid_action.append(5)
        if random.random() > epsilon:
            with torch.no_grad():
                q_table = self.policy_net(self.state_to_onehot(torch.as_tensor(state).float().to(self.device).unsqueeze(0))).argsort(descending=True)
                action = q_table.squeeze(0)[0].item()
                if action not in valid_action:
                    for a in q_table.squeeze(0)[1:]:
                        if a.item() in valid_action:
                            action = a.item()
                            break
        else:
            action = random.choice(valid_action)
        if action == 4 and [state[0], state[1]] in self.stations and state[-3] and not self.has_passenger:
            self.has_passenger = True
        return action

    def get_state_and_goal(self, obs, done):
        stations = [[0, 0], [0, 0], [0, 0], [0, 0]]
        taxi_row, taxi_col, stations[0][0], stations[0][1], stations[1][0], stations[1][1], stations[2][0], stations[2][1], stations[3][0], stations[3][1],obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs
        if not passenger_look or not all([i[0] == j[0] and i[1] == j[1] for (i, j) in zip(self.stations, stations)]):
            self.has_passenger = False
        goal = stations[self.goal_cnt]
        self.stations = stations
        if [taxi_row, taxi_col] == goal:
            if not(passenger_look and not self.has_passenger) and not(destination_look and self.has_passenger):
                self.goal_cnt += 1
                self.goal_cnt %= 4
        if done:
            self.reset()
        return [taxi_row, taxi_col, goal[0], goal[1], obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look, self.has_passenger], goal

    def forward(self, x):
        return self.policy_net(x)

    def update(self):
        if len(self.memory) < self.batch_size:
            return 0
        states, actions, rewards, next_states, desired_goal, dones = self.memory.sample(self.batch_size)
        non_final_mask = 1 - np.array(dones)
        non_final_mask = torch.as_tensor(non_final_mask)
        state_action_values = self.policy_net(self.state_to_onehot(torch.as_tensor(states).float().to("cuda"))).gather(1, torch.as_tensor(actions).to("cuda"))
        with torch.no_grad():
            next_state_values = self.target_net(self.state_to_onehot(torch.as_tensor(next_states).float().to("cuda"))).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + torch.as_tensor(rewards).float().to("cuda").squeeze(1).squeeze(1)
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.num_update += 1
        if self.num_update % self.update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return loss.item()
