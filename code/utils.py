import pickle
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def save_agent(agent, filename="agent"):
    """
    Function to save the agent object to a pickle file.

    Args:
        agent: The agent object to be saved.
        filename: Name of the file to save the agent (default is "agent").

    Returns:
        None
    """
    os.makedirs("saved_agents", exist_ok=True)
    current_datetime = datetime.now()
    date_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    with open(f"saved_agents/{filename}: {date_string}.pkl", 'wb') as file:
        pickle.dump(agent, file)
    print(f"Agent saved at saved_agents/{filename}: {date_string}.pkl")

def load_agent(path, verbose=False):
    """
    Function to load an agent object from a pickle file.

    Args:
        path: Path to the pickle file containing the agent object.
        verbose: Flag indicating whether to print verbose information (default is False).

    Returns:
        agent: The loaded agent object.
    """
    print(f"Loading agent from {path}")
    with open(path, 'rb') as file:
        agent = pickle.load(file)
    if verbose: 
        agent.print()
    return agent


def plot_returns(y, evaluation_frequency=1, instance="rooms_instance", name="returns", agent_names=""):
    """
    Function to plot the returns (discounted rewards) over episodes.

    Args:
        y: Y-axis data (e.g., discounted returns).
        instance: Name of the instance/environment (default is "rooms_instance").

    Returns:
        None
    """
    if agent_names != "":
        for i in range(len(agent_names)):
            df = pd.DataFrame(y[i])
            df = df.melt(var_name="Episode", value_name="Discounted Return") # lineplot expects data in long format
            df["Episode"] = evaluation_frequency*df["Episode"]
            sns.lineplot(x="Episode", y="Discounted Return", data=df, errorbar=('ci', 95), label=agent_names[i])
        y = y[-1]

    if y.shape[1] < 25 and evaluation_frequency==1: 
        plt.xticks(range(0, y.shape[1], 1))
    
    plt.grid()
    plt.axhline(y=0.8, color='black', linestyle='--')
    plt.title(f"{name} returns")
    plt.tight_layout()
    plt.savefig(f"{instance}_{name}_returns.png", dpi=500)
    # plt.show()
    plt.close()

def plot_value_map(env, agent, instance="rooms_instance"):
    states = env.get_all_states()
    value_map = states[0][2].copy()
    value_map[value_map == 1] = -1
    value_dirs = np.zeros(value_map.shape, dtype=int)
    dirs = ["D", "U", "L", "R"]
    for s in states:
        q = agent.Q(s)
        index = np.where(s[0] == 1)
        value_map[index[0][0]][index[1][0]] = max(q)
        value_dirs[index[0][0]][index[1][0]] = np.argmax(q)
    goal_index = np.where(states[0][1] == 1)
    value_map[goal_index[0][0]][goal_index[1][0]] = 1
    fig, ax = plt.subplots(figsize=(30, 30))
    ax.clear()
    ax.grid(False)
    ax.imshow(value_map.T)
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelleft=False, labelbottom=False)
    for (i, j), v_s in np.ndenumerate(value_map):
        if v_s == -1:
            continue
        ax.text(i, j, '{:0.4f}'.format(v_s) + f"\n{dirs[value_dirs[int(i)][int(j)]]}", ha='center', va='center', fontsize=7, bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.03'))
    plt.tight_layout()
    # np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})
    plt.show()
    # print(value_map)
    fig.savefig(f"{agent.__class__.__name__}_value_map_{instance}.png", dpi=500)
    # plt.close()