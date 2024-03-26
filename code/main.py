import rooms
import random
import agent as a
import pandas as pd
import sys
from utils import save_agent, load_agent, plot_returns
import numpy as np


def episode(env, agent, nr_episode=0, evaluation_mode=False, verbose=True):
    state = env.reset()
    discounted_return = 0
    discount_factor = 0.99
    done = False
    time_step = 0
    if evaluation_mode:
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
    train_returns = []
    eval_returns = []
    for seed in range(no_seeds):
        np.random.seed(seed)
        random.seed(seed)
        returns = [episode(env, agent, nr_episode=i, verbose=True) for i in range(episodes)]
        train_returns.append(returns)
        returns = evaluate(env, agent, runs=no_runs, episodes=evaluation_episodes)
        eval_returns.append(returns)
    return np.array(train_returns), np.array(eval_returns).reshape(no_seeds*no_runs, evaluation_episodes)

def evaluate(env, agent, runs, episodes):
    eval_returns = []
    for _ in range(no_runs):
        returns = [episode(env, agent, nr_episode=i, verbose=False, evaluation_mode=True) for i in range(episodes)]
        eval_returns.append(returns)
    return eval_returns


params = {}
rooms_instance = sys.argv[1]
env = rooms.load_env(f"layouts/{rooms_instance}.txt", f"{rooms_instance}.mp4")
params["nr_actions"] = env.action_space.n
params["gamma"] = 0.99
params["epsilon_decay"] = 0.0001
params["alpha"] = 0.1
params["env"] = env
params["exploration_constant"] = np.sqrt(2)
params["epsilon"] = 1

#agent = a.RandomAgent(params)
# agent = a.SARSALearner(params)
# agent = a.QLearner(params)
agent = a.UCBQLearner(params)

training_episodes = 200
evaluation_episodes = 20
no_runs = 100
no_seeds = 10

# TRAINING
train_returns, eval_returns = train(env, agent, training_episodes)
print(f"Trained for {training_episodes} episodes with {no_seeds} seeds.")
print(f"Average training discounted return: {np.mean(train_returns)}")
print(f"For each training seed, evaluated for {evaluation_episodes} episodes with {no_runs} runs.")
print(f"Average evaluation discounted return: {np.mean(eval_returns)}")
plot_returns(x=range(training_episodes), y=train_returns, evaluation_mode=False)
plot_returns(x=range(evaluation_episodes), y=eval_returns, evaluation_mode=True)
# save_agent(agent)

# EVALUATION
# agent = load_agent("saved_agents/agent: 2024-03-19 13:01:51.pkl")
# eval_returns = evaluate(env, agent, runs=no_runs, episodes=evaluation_episodes)
# plot_returns(x=range(evaluation_episodes), y=eval_returns)

# env.save_video()
