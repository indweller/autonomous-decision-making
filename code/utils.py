import pickle
from datetime import datetime
import os
import matplotlib.pyplot as plot
import seaborn as sns
import pandas as pd

def save_agent(agent, filename="agent"):
    os.makedirs("saved_agents", exist_ok=True)
    current_datetime = datetime.now()
    date_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    with open(f"saved_agents/{filename}: {date_string}.pkl", 'wb') as file:
        pickle.dump(agent, file)

def load_agent(path, verbose=False):
    with open(path, 'rb') as file:
        agent = pickle.load(file)
    if verbose: 
        agent.print()
    return agent


def plot_returns(x, y, eval=True):
    df = pd.DataFrame(y)
    df = df.melt(var_name="Episode", value_name="Discounted Return") # lineplot expects data in long format
    sns.lineplot(x="Episode", y="Discounted Return", data=df, errorbar=('ci', 95))
    plot.axhline(y=0.8, color='black', linestyle='--')
    plot.title("Evaluation returns") if eval else plot.title("Training returns")
    plot.show()