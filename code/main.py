import rooms
import random
import agent as a
import pandas as pd
import sys
from utils import save_agent, load_agent, plot_returns, plot_value_map
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
    discount_factor = params["gamma"] # Discount factor for future rewards
    done = False # Flag indicating episode completion
    time_step = 0  # Initialize time step

    if evaluation_mode: # if evaluating set epsilon and exploration to 0 for greedy policy
        prev_epsilon = getattr(agent, "epsilon", 0) # Reset it after evaluation
        prev_exploration_constant = getattr(agent, "exploration_constant", 0)
        prev_temperature = getattr(agent, "temperature", 0)
        agent.epsilon = 0
        agent.exploration_constant = 0
        agent.temperature = 0

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
    if evaluation_mode: # reset epsilon and exploration after evaluation
        agent.epsilon = prev_epsilon
        agent.exploration_constant = prev_exploration_constant
        agent.temperature = prev_temperature
    return discounted_return
  
def train(env, agent, episodes, evaluation_frequency=10, evaluation_episodes=20, verbose=True):
    """
    Function to train the agent over multiple episodes.

    Args:
        env: The environment instance.
        agent: The agent instance.
        episodes: Number of episodes to train on.
        evaluation_frequency: Frequency of evaluation episodes.
        evaluation_episodes: Number of episodes for evaluation.

    Returns:
        train_returns: Returns from training episodes.
        eval_returns: Returns from evaluation episodes.
    """
    train_returns = []
    eval_returns = []
    # Train on episodes and store returns
    for i in range (episodes):
        if i % evaluation_frequency == 0:
            returns = evaluate(env, agent, runs=1, episodes=evaluation_episodes)
            print(f"Evaluation at episode {i}: {np.mean(returns)}")
            eval_returns.append(np.mean(returns))
        train_episode_return = episode(env, agent, nr_episode=i, evaluation_mode=False, verbose=verbose)
        train_returns.append(train_episode_return)
    return np.array(train_returns), np.array(eval_returns)

def evaluate(env, agent, runs, episodes):
    """
    Function to evaluate the agent over multiple runs and episodes.

    Args:
        env: The environment instance.
        agent: The agent instance.
        runs: Number of runs for evaluation.
        episodes: Number of episodes for each run.

    Returns:
        eval_returns: Returns from evaluation episodes.
    """
    eval_returns = []
    for _ in range(runs): 
        returns = [episode(env, agent, nr_episode=i, evaluation_mode=True, verbose=False) for i in range(episodes)]
        eval_returns.append(returns)
    return eval_returns

rooms_instance = sys.argv[1]
#  HYPER PARAMETERS
params = {}
params["gamma"] = 0.99
params["epsilon_decay"] = 0.001
params["alpha"] = 0.1
params["exploration_constant"] = np.sqrt(2)
params["epsilon"] = 1
params["temperature"] = 1

training_episodes = 200
evaluation_episodes = 20
evaluation_frequency = 10
test_runs = 10
test_episodes = 20
seeds = 5

agents = [a.RandomAgent, a.QLearner, a.UCBQLearner, a.BoltzmannQLearner, a.SARSALearner, a.UCBSARSALearner, a.BoltzmannSARSALearner]
agent_names = ["Random", "Eps-Q-learning", "UCB-Q-learning", "Boltzmann-Q-learning", "Eps-SARSA", "UCB-SARSA", "Boltzmann-SARSA"]
# agents = [a.UCBQLearner]
# agent_names = ["UCB-Q-learning"]

agent_train_returns = []
agent_eval_returns = []
agent_test_returns = []
for agent_class in agents:
    train_returns = []
    eval_returns = []
    test_returns = []
    for i in range(seeds):
        print(f"Seed: {i}")
        np.random.seed(i)
        random.seed(i) 
        env = rooms.load_env(f"layouts/{rooms_instance}.txt", f"{rooms_instance}.mp4")
        params["nr_actions"] = env.action_space.n
        params["env"] = env

        # AGENTS
        # agent = a.RandomAgent(params)
        # agent = a.SARSALearner(params)
        # agent = a.QLearner(params)
        # agent = a.UCBQLearner(params)
        # agent = a.BoltzmannQLearner(params)
        # agent = a.UCBSARSALearner(params)
        # agent = load_agent("saved_agents/agent: 2024-04-05 18:33:40.pkl")
        agent = agent_class(params)
        print(agent.__class__.__name__)

        # TRAINING
        i_train_returns, i_eval_returns = train(env, agent, training_episodes, evaluation_frequency, evaluation_episodes, verbose=False)
        train_returns.append(i_train_returns)
        eval_returns.append(i_eval_returns)
        # TESTING
        i_test_returns = evaluate(env, agent, test_runs, test_episodes)
        test_returns.append(i_test_returns)

        # save_agent(agent)
        print(f"Trained for {training_episodes} episodes. Average training discounted return: {np.mean(train_returns)}")
        print(f"Tested for {test_runs} runs with {test_episodes} episodes each. Average test discounted return: {np.mean(test_returns)}")
    train_returns = np.array(train_returns)
    eval_returns = np.array(eval_returns)
    test_returns = np.array(test_returns).reshape(-1, test_episodes)
    agent_train_returns.append(train_returns)
    agent_eval_returns.append(eval_returns)
    agent_test_returns.append(test_returns)

plot_returns(agent_train_returns, instance=rooms_instance, name="Training", agent_names=agent_names)
plot_returns(agent_eval_returns, evaluation_frequency=evaluation_frequency, instance=rooms_instance, name="Evaluation", agent_names=agent_names)
plot_returns(agent_test_returns, instance=rooms_instance, name="Testing", agent_names=agent_names)

plot_value_map(env, agent, instance=rooms_instance)

# env.save_video()