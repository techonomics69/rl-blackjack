# Solving Blackjack with Q-Learning

This repository follows along with the [OpenAI Gymnasium tutorial](https://gymnasium.farama.org/tutorials/blackjack_tutorial/) on how to solve Blackjack with Reinforcement Learning (RL). The tutorial uses a fundamental model-free RL algorithm known as Q-learning.

## Outline

In this tutorial, we’ll explore and solve the *Blackjack-v1* environment. Blackjack is one of the most popular casino card games that is also infamous for being beatable under certain conditions. This version of the game uses an infinite deck (we draw the cards with replacement), so counting cards won’t be a viable strategy in our simulated game. Full documentation can be found [here](https://gymnasium.farama.org/environments/toy_text/blackjack).

**Objective**: To win, your card sum should be greater than the dealers without exceeding 21.
**Actions**: Agents can pick between two actions:
- stand (0): the player takes no more cards
- hit (1): the player will be given another card, however the player could get over 21 and bust

To solve this problem you may pick your favorite discrete RL algorithm. The presented solution uses Q-learning (a model-free RL algorithm). 

## Installing environment

To run the code in this repository, you must first install and activate the conda environment. Simply paste the following commands into your terminal.

```bash
# Create conda environment from YML file
conda env create -f environment.yml

# Activate the conda environment
conda activate drl
```
