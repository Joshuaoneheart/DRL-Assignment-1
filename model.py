import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
from memory import Memory
class DQN(nn.Module):
    def __init__(self, state_size, action_size, gamma=0.99, batch_size=32, lr=1e-5, device="cpu"):
        super().__init__()
        self.policy_net = nn.Sequential(
                nn.Linear(state_size, 128),
                nn.GELU(),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, action_size)
                )
        self.target_net = nn.Sequential(
                nn.Linear(state_size, 128),
                nn.GELU(),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, action_size)
                )
        self.tau = 0.005
        self.device = device
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = Memory(1500, 4)
        self.batch_size = batch_size
        self.gamma = gamma
        self.criterion = nn.MSELoss()

        # state related
        self.goal_cnt = 0
        self.has_passenger = False
    
    def load(self):
        self.policy_net.load_state_dict(torch.load("model.pkl", map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def reward_shaping(self, reward, state, action):
        if action == 5 and not([state[0], state[1]] in self.stations and state[-2]):
            reward -= 100
        return reward

    def reset(self):
        self.has_passenger = False

    def get_action(self, obs, epsilon):
        state, goal = self.get_state_and_goal(obs, False)
        if random.random() > epsilon:
            with torch.no_grad():
                action = self.policy_net(torch.as_tensor(state).float().to(self.device)).argmax().item()
        else:
            action = random.choice(list(range(6)))
        if action == 4 and [state[0], state[1]] in self.stations and state[-2]:
            self.has_passenger = True
        return action

    def get_state_and_goal(self, obs, done):
        stations = [[0, 0], [0, 0], [0, 0], [0, 0]]
        taxi_row, taxi_col, stations[0][0], stations[0][1], stations[1][0], stations[1][1], stations[2][0], stations[2][1], stations[3][0], stations[3][1],obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs
        goal = stations[self.goal_cnt]
        self.stations = stations
        if [taxi_row, taxi_col] == goal:
            if not(passenger_look and not self.has_passenger) or not(destination_look and self.has_passenger):
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
        state_action_values = self.policy_net(torch.as_tensor(states).float().to("cuda")).gather(1, torch.as_tensor(actions).to("cuda"))
        with torch.no_grad():
            next_state_values = self.target_net(torch.as_tensor(next_states).float().to("cuda")).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + torch.as_tensor(rewards).float().to("cuda").squeeze(1).squeeze(1)
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)
        return loss.item()
