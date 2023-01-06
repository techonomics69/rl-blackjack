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


# *** EXECUTING AN ACTION ***

# Reset the environment to get the first state (observation)
# The state comprises of a 3-tuple (int, int, bool):
# - The players current sum
# - Value of the dealer's face-up card
# - Boolean whether the player holds a usable ace, i.e. an ace that can count as 11 without busting
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
# If you continue executing actions without resetting the environment, it still respond but the output won’t be useful 
# for training (it might even be harmful if the agent learns on invalid data).


# *** BUILDING AN AGENT ***

class BlackjackAgent:
    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95
    ):
        """
        Initialize a Reinforcement Learning agent with an empty dictionary of state-action values (q_values),
        a learning rate and an epsilon.

        Args:
        - learning_rate: Amount with which to weight newly learned reward vs old reward (1 - lr)
        - initial epsilon: The initial probability w/ with we sample random action (exploration)
        - epsilon_decay: Value by which epsilon value decays through subtraction
        - final_epsilon: Epsilon value at which decay stops
        - discount_factor: The factor by which future rewards are counted, i.e. expected return on next state (recursive)
        """
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        
        self.training_error = []
    
    def get_action(self, state: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon) -> exploitation. 
        Otherwise a random action with probability epsilon to ensure exploration.
        """
        if np.random.random() < self.epsilon:
            return env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[state]))
    
    def update(
        self,
        state: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_state: tuple[int, int, bool]
    ):
        """
        Updates the Q-value of an action.
        The Q-value update is equivalent to the following weighting of old and new information by the learning rate:
        # self.q_values[state][action] = (1 - self.lr) * self.q_values[state][action] +
        #                                self.lr * (reward + self.discount_factor * future_q_value)
        The temporal difference is the difference between the old and new value over one (time) step.
        """
        future_q_value = (not terminated) * np.max(self.q_values[next_state]) 
        temporal_difference = reward + self.discount_factor * future_q_value - self.q_values[state][action]
        self.q_values[state][action] = self.q_values[state][action] + self.lr * temporal_difference
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


# *** WRITING TRAINING LOOP ***

learning_rate = 0.01
n_episodes = 100_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)
final_epsilon = 0.1

agent = BlackjackAgent(
    learning_rate = learning_rate,
    initial_epsilon = start_epsilon,
    epsilon_decay = epsilon_decay,
    final_epsilon = final_epsilon,
)

env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
for episode in tqdm(range(n_episodes)):
    state, info = env.reset()
    done = False
    
    #play one episode
    while not done:
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)

        # update the agent
        agent.update(state, action, reward, terminated, next_state)

        # update done status and state
        done = terminated or truncated
        state = next_state

    # once a game is finished we decay epsilon -> converge towards exploitation
    agent.decay_epsilon()


# *** VISUALISE THE TRAINING ***

rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

# Episode rewards plot
axs[0].set_title("Episode rewards")
reward_moving_average = (
    np.convolve(np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid") / rolling_length
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)

# Episode lengths plot
axs[1].set_title("Episode lengths")
length_moving_average = (
    np.convolve(np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same") / rolling_length
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)

# Training error plot
axs[2].set_title("Training Error")
training_error_moving_average = (
    np.convolve(np.array(agent.training_error), np.ones(rolling_length), mode="same") / rolling_length
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)

plt.tight_layout()
plt.show()