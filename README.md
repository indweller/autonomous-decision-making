# Autonomous Decision-Making

## Setup

The code requires [Anaconda](https://www.anaconda.com/download).

Please create a virtual environment before running the code (see documentation for [Visual Code](https://code.visualstudio.com/docs/python/environments))

To install all dependencies run the following commands in a terminal:

```
cd code
pip install -r requirements.txt
```

## Available Maps

All available maps are provided in the folder `code/layouts` and listed in the table below.

| Map        | File                        |
| ---------- | --------------------------- |
| `easy_0`   | `code/layouts/easy_0.txt`   |
| `easy_1`   | `code/layouts/easy_1.txt`   |
| `medium_0` | `code/layouts/medium_0.txt` |
| `medium_1` | `code/layouts/medium_1.txt` |
| `hard_0`   | `code/layouts/hard_0.txt`   |
| `hard_1`   | `code/layouts/hard_1.txt`   |

## Usage

Run agent using the following commands in a terminal (`map-name` is provided in the "Map"-column of the table above):

```
cd code
python main.py <map-name>
```

The current version of the code creates comparative results for every agent, which can be time consuming. The agents evaluated are stored in the `agents` variable in `main.py`. For shorter run times, you can reduce the `test_runs`, `seeds` and `evaluation_frequency`. We suggest 200 training episodes for medium maps and 500 episodes for hard maps. All the parameters and the agents to be trained, can be set in the `main.py` file. 

Our proposed algorithm is `UCBQLearner` - for which you can find the plots and videos in `code/`. The results were obtained with `training_episodes = 200, evaluation_episodes = 20, evaluation_frequency = 10, test_runs = 100, test_episodes = 20, seeds = 10`.
