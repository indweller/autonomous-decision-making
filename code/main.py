import rooms
import random
import agent as a
import pandas as pd
import sys
from utils import save_agent, load_agent, plot_returns
import numpy as np
import matplotlib.pyplot as plt

def episode(env, agent, nr_episode=0, evaluation_mode=False, verbose=True):
    """
    Function to run a single episode of the reinforcement learning process.

    Args:
        env: The environment instance.
        agent: The agent instance.
        nr_episode: Episode number.
        evaluation_mode: Flag indicating evaluation mode.
        verbose: Flag indicating verbosity.

    Returns:
        discounted_return: Discounted return for the episode.
    """
    state = env.reset() # Reset the environment and get initial state
    discounted_return = 0 # Initialize discounted return
    discount_factor = 0.99 # Discount factor for future rewards
    done = False # Flag indicating episode completion
    time_step = 0  # Initialize time step

    if evaluation_mode: # if evaluating set epsilon and exploration to 0 for consistency
        agent.epsilon = 0
        agent.exploration_constant = 0
    while not done:
        # 1. Select action according to policy
        action = agent.policy(state)
        # 2. Execute selected action
        next_state, reward, terminated, truncated, _ = env.step(action)
        # 3. Integrate new experience into agent
        if not evaluation_mode: 
            agent.update(state, action, reward, next_state, terminated, truncated)
        state = next_state
        done = terminated or truncated
        discounted_return += (discount_factor**time_step)*reward
        time_step += 1
    if verbose: print(nr_episode, ":", discounted_return, "steps: ", time_step)
    return discounted_return
  
def train(env, agent, episodes):
    """
    Function to train the agent over multiple episodes.

    Args:
        env: The environment instance.
        agent: The agent instance.
        episodes: Number of episodes to train on.

    Returns:
        train_returns: Returns from training episodes.
        eval_returns: Returns from evaluation episodes.
    """
    train_returns = []
    eval_returns = []
    plotted_eval_returns = []
    for seed in range(no_seeds): # Loop through different random seeds
        # set random seeds
        np.random.seed(seed)
        random.seed(seed) 
         # Train on episodes and store returns
        for i in range (episodes):
            if i % 10 == 0:
                print(f"Episode {i}")
                returns = evaluate(env, agent, runs=no_eval_runs, episodes=20)
                plotted_eval_returns.append(np.mean(returns))
            returns = episode(env, agent, nr_episode=i, verbose=False)
            train_returns.append(returns)
        train_returns.append(returns)
        # Evaluate and store returns
        returns = evaluate(env, agent, runs=no_eval_runs, episodes=evaluation_episodes)
        eval_returns.append(returns)
    return np.array(train_returns), np.array(eval_returns).reshape(no_seeds*no_eval_runs, evaluation_episodes), np.array(plotted_eval_returns)

# EVALUATION
def evaluate(env, agent, runs, episodes):
    """
    Function to evaluate the agent over multiple episodes.

    Args:
        env: The environment instance.
        agent: The agent instance.
        runs: Number of runs for evaluation.
        episodes: Number of episodes for each run.

    Returns:
        eval_returns: Returns from evaluation episodes.
    """
    eval_returns = []
    for _ in range(no_eval_runs): 
        returns = [episode(env, agent, nr_episode=i, verbose=False, evaluation_mode=True) for i in range(episodes)]
        eval_returns.append(returns)
    return eval_returns


params = {}
rooms_instance = sys.argv[1]
env = rooms.load_env(f"layouts/{rooms_instance}.txt", f"{rooms_instance}.mp4")

#  HYPER PARAMETERS
params["nr_actions"] = env.action_space.n
params["gamma"] = 0.99
params["epsilon_decay"] = 0.0001
params["alpha"] = 0.1
params["env"] = env
params["exploration_constant"] = np.sqrt(2)
params["epsilon"] = 1
training_episodes = 200
evaluation_episodes = 20
no_eval_runs = 10
no_seeds = 1

# AGENTS
#agent = a.RandomAgent(params)
# agent = a.SARSALearner(params)
# agent = a.QLearner(params)
agent = a.UCBQLearner(params)



# TRAINING
train_returns, eval_returns, plotted_eval_returns = train(env, agent, training_episodes)

print(f"Trained for {training_episodes} episodes with {no_seeds} seeds.")
print(f"Average training discounted return: {np.mean(train_returns)}")
print(f"For each training seed, evaluated for {evaluation_episodes} episodes with {no_eval_runs} runs.")
print(f"Average evaluation discounted return: {np.mean(eval_returns)}")
plt.plot(np.arange(len(plotted_eval_returns)), plotted_eval_returns)
plt.show()

# plot_returns(x=range(training_episodes), y=train_returns, evaluation_mode=False, instance=rooms_instance)
print("y", plotted_eval_returns,eval_returns)
# plot_returns(x=range(evaluation_episodes), y=eval_returns, evaluation_mode=True, instance=rooms_instance)
# save_agent(agent)

# EVALUATION
# agent = load_agent("saved_agents/agent: 2024-03-19 13:01:51.pkl")
# eval_returns = evaluate(env, agent, runs=no_eval_runs, episodes=evaluation_episodes)
# plot_returns(x=range(evaluation_episodes), y=eval_returns)

# env.save_video()
