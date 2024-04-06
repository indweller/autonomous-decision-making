import pickle
from datetime import datetime
import os
import matplotlib.pyplot as plot
import seaborn as sns
import pandas as pd

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


def plot_returns(y, instance="rooms_instance", name="returns"):
    """
    Function to plot the returns (discounted rewards) over episodes.

    Args:
        y: Y-axis data (e.g., discounted returns).
        instance: Name of the instance/environment (default is "rooms_instance").

    Returns:
        None
    """
    df = pd.DataFrame(y)
    df = df.melt(var_name="Episode", value_name="Discounted Return") # lineplot expects data in long format
    sns.lineplot(x="Episode", y="Discounted Return", data=df, errorbar=('ci', 95))

    if y.shape[1] < 25: 
        plot.xticks(range(0, y.shape[1], 1))
    
    plot.grid()
    plot.axhline(y=0.8, color='black', linestyle='--')
    plot.title(f"{name} returns")
    # plot.show()
    plot.savefig(f"{instance}_{name}_returns.png", dpi=500)
    plot.close()