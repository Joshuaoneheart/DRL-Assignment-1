import numpy as np
import random


class Memory:
    def __init__(self, capacity, k_future):
        self.capacity = capacity
        self.memory = []
        self.memory_counter = 0
        self.memory_length = 0

        self.future_p = 1 - (1. / (1 + k_future))

    def compute_reward(self, achieved_goal, desired_goal, rewards):
        for i, (a_g, d_g) in enumerate(zip(achieved_goal, desired_goal)):
            rewards[i] += 10 - abs(a_g[0] - d_g[0]) - abs(a_g[1] - d_g[1])
        return rewards

    def sample(self, batch_size):

        ep_indices = np.random.randint(0, len(self.memory), batch_size)
        time_indices = []
        epi_len = []
        for epi in ep_indices:
            epi_len.append(len(self.memory[epi]["next_state"]))
            time_indices.append(np.random.randint(0, len(self.memory[epi]["next_state"])))
        epi_len = np.array(epi_len)
        time_indices = np.array(time_indices)
        states = []
        actions = []
        rewards = []
        desired_goals = []
        next_states = []
        next_achieved_goals = []
        dones = []

        for episode, timestep in zip(ep_indices, time_indices):
            states.append(self.memory[episode]["state"][timestep])
            actions.append(self.memory[episode]["action"][timestep])
            rewards.append(self.memory[episode]["reward"][timestep])
            desired_goals.append(self.memory[episode]["desired_goal"][timestep])
            next_achieved_goals.append(self.memory[episode]["next_achieved_goal"][timestep])
            next_states.append(self.memory[episode]["next_state"][timestep])
            dones.append(self.memory[episode]["done"][timestep])

        states = np.vstack(states)
        actions = np.vstack(actions)
        desired_goals = np.vstack(desired_goals)
        rewards = np.vstack(rewards)
        next_achieved_goals = np.vstack(next_achieved_goals)
        next_states = np.vstack(next_states)

        her_indices = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (epi_len - time_indices)
        future_offset = future_offset.astype(int)
        future_t = (time_indices + future_offset + 1)
        future_ag = []
        for i, (episode, f_offset) in enumerate(zip(ep_indices, future_t)):
            future_ag.append(self.memory[episode]["achieved_goal"][min(f_offset, epi_len[i] - 1)])
        future_ag = np.vstack(future_ag)

        desired_goals[her_indices[0]] = future_ag[her_indices[0]]
        for i, goal in enumerate(desired_goals):
            states[i][2] = goal[0]
            states[i][3] = goal[1]
            next_states[i][2] = goal[0]
            next_states[i][3] = goal[1]

        rewards = np.expand_dims(self.compute_reward(next_achieved_goals, desired_goals, rewards), 1)

        return states, actions, rewards, next_states, desired_goals, dones

    def add(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)
        assert len(self.memory) <= self.capacity

    def __len__(self):
        return len(self.memory)

