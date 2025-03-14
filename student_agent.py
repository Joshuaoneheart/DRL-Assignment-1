# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

from model import DQN
policy = DQN(27, 6, 0.99, 128, 0.0001)
policy.load()
def get_action(obs):
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.

    return policy.get_action(obs, 0.4)

    # You can submit this random agent to evaluate the performance of a purely random strategy.

