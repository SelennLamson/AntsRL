import pickle
from tqdm import tqdm
import datetime
import time
import math

from gui.visualize import Visualizer

from environment.RL_api import RLApi
from generator.environment_generator import EnvironmentGenerator
from generator.map_generators import *

from environment.rewards.explore_reward import ExploreReward
from environment.rewards.pick_up_food_reward import PickUpFoodReward
from environment.rewards.feed_anthill_reward import FeedAnthillReward

from agents.random_agent import RandomAgent
from agents.collect_agent_rework import CollectAgentRework
from agents.collect_agent_memory import CollectAgentMemory


# -------------------------------------------
#               Main parameters
# -------------------------------------------
save_file_name = "collect_agent.arl"
# -------------------------------------------

if __name__ == '__main__':
    visualiser = Visualizer()
    visualiser.big_dim = 900
    visualiser.visualize(save_file_name)
