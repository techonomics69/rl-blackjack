# License: MIT License
from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from tqdm import tqdm

import gymnasium as gym

# Let's start by creating the blackjack environment.
# Note: We are going to follow the rules from Sutton & Barto.
env = gym.make("Blackjack-v1", sab=True)

# Reset the environment to get the first state (observation)
# The state comprises of a 3-tuple (int, int, bool):
# - The players current sum
# - Value of the dealer's face-up card
# - Boolean whether the player holds a usable ace, i.e. an ace that count as 11 without busting
state, info = env.reset()
done = False

# Sample a random action from all valid actions
action = env.action_space.sample()

# Execute the action in our environment and receive data from the environment
# next_state: This is the observation that the agent will receive after taking the action.
# reward: This is the reward that the agent will receive after taking the action.
# terminated: This is a boolean variable that indicates whether or not the environment has terminated.
# truncated: This is a boolean variable that also indicates whether the episode ended by early truncation, i.e., time limit is reached.
# info: This is a dictionary that might contain additional information about the environment.
next_state, reward, terminated, truncated, info = env.step(action)
# Once terminated or truncated is True, we should stop the current episode and begin a new one with env.reset(). 
# If you continue executing actions without resetting the environment, it still responds but the output won’t be useful 
# for training (it might even be harmful if the agent learns on invalid data).