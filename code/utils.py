import pickle
from datetime import datetime


def save_agent(agent, filename="agent"):
    current_datetime = datetime.now()
    date_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    with open(f"saved_agents/{filename}: {date_string}.pkl", 'wb') as file:
        pickle.dump(agent, file)

def load_agent(path):
    with open(path, 'rb') as file:
        agent = pickle.load(file)
    agent.print()
    return agent