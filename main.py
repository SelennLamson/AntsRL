import pickle
from tqdm import tqdm
import datetime
import time
import math
import gc

from gui.visualize import Visualizer

from environment.RL_api import RLApi
from generator.environment_generator import EnvironmentGenerator
from generator.map_generators import *

from environment.rewards.explore_reward import ExploreReward
from environment.rewards.pick_up_food_reward import PickUpFoodReward
from environment.rewards.feed_anthill_reward import FeedAnthillReward
from environment.rewards.anthill_heading_reward import AnthillHeadingReward

from agents.random_agent import RandomAgent
from agents.collect_agent_rework import CollectAgentRework
from agents.collect_agent_memory import CollectAgentMemory

import matplotlib.pyplot as plt


# -------------------------------------------
#               Main parameters
# -------------------------------------------
aggregate_stats_every = 5
save_model = True
visualize_every = 10      # Save every X episodes for visualisation
training = True
only_visualize = False
use_model = None
# use_model = "20_4_19_collect_agent_memory.h5"
save_file_name = "collect_agent.arl"

starting_point = 0
episodes = 100
steps = 1500

min_epsilon = 0.01
max_epsilon = 1
# -------------------------------------------

# Key: step of episode      Value: list of reward factors
reward_sequence = {
    # 0: [1, 0, 0, 0],
    # 0: [1, 2, 5, 10],
    0: [1, 2, 10, 100]
}

def main():
    states = []

    n_ants = 20

    # Setting up environment
    generator = EnvironmentGenerator(w=200,
                                     h=200,
                                     n_ants=n_ants,
                                     n_pheromones=2,
                                     n_rocks=0,
                                     # food_generator=AvoidingCirclesGenerator(15, 5, 10, 100, 100, 20),
                                     food_generator=CirclesGenerator(15, 5, 10),
                                     walls_generator=PerlinGenerator(scale=22.0, density=0.2),
                                     max_steps=steps,
                                     # seed=420)
                                     seed=None)

    # Setting up RL Rewards
    rewards = [
        ExploreReward(0.4),
        PickUpFoodReward(0.9),
        AnthillHeadingReward(0.7),
        FeedAnthillReward(0.99)
    ]

    # Setting up RL Api
    api = RLApi(rewards=rewards,
                reward_threshold=0,
                max_speed=1,
                max_rot_speed=50 / 180 * np.pi,
                carry_speed_reduction=0.05,
                backward_speed_reduction=0.5)
    api.save_perceptive_field = True

    # Setting up RL Agent
    agent = CollectAgentMemory(rewards=rewards,
                               epsilon=1,
                               learning_rate=1e-5,
                               rotations=3,
                               pheromones=3)

    agent_is_setup = False

    avg_loss = None
    avg_time = None
    first_save = True

    print("Starting simulation...")
    for episode in range(starting_point, episodes + starting_point):
        visualize_episode = (episode + 1) % visualize_every == 0 or not training

        env = generator.generate(api)
        print('\n--- Episode {}/{} --- {}'.format(episode + 1, episodes + starting_point, "VISUALIZED" if visualize_episode else ""))

        # Setups the agents only once
        if not agent_is_setup:
            agent.setup(api, use_model)
            agent_is_setup = True

        # Initializes the agents on the new environment
        agent.initialize(api)

        obs, agent_state, state = api.observation()
        episode_reward = np.zeros((n_ants, len(rewards)))

        # r0 = [0, 0, 0, 0]

        for s in range(steps):
            if s in reward_sequence:
                for i, r in enumerate(rewards):
                    r.factor = reward_sequence[s][i]

            now = time.time()

            # Compute the next action of the agents
            action = agent.get_action(obs, agent_state, training)

            # Execute the action
            new_state, new_agent_state, reward, done = api.step(*action[:2])

            # for i in range(4):
            #     r0[i] = rewards[i].discount * r0[i] + reward[0, i]
            #     if abs(r0[i]) < 0.001:
            #         r0[i] = 0
            # print(r0)

            # Add the reward values to total reward of episode
            episode_reward += reward

            # Update replay memory with new action and states
            agent.update_replay_memory(obs, agent_state, action, reward, new_state, new_agent_state, done)

            # Train the neural network
            if training and s % 4 == 0:
                loss = agent.train(done, s)

                if avg_loss is None:
                    avg_loss = loss
                else:
                    avg_loss = 0.99 * avg_loss + 0.01 * loss
            # else:
            #     avg_loss = 0

            # Set obs to the new state
            obs = new_state
            agent_state = new_agent_state

            if (s + 1) % 50 == 0:
                mean_reward = episode_reward.mean(axis=0)
                # max_reward = episode_reward.max(axis=0)
                # min_reward = episode_reward.min(axis=0)
                # var_reward = episode_reward.std(axis=0)
                total_reward = episode_reward.sum(axis=0)
                eta_seconds = int(((steps - s) * avg_time + (episodes + starting_point - episode - 1) * steps * avg_time) / 1000)

                print("\rAverage loss : {:.5f} --".format(avg_loss) if training else "\r",
                      # "Episode reward stats: mean {:.2f} - min {:.2f} - max {:.2f} - std {:.2f} - total {:.2f} --".format(
                      #     mean_reward, min_reward, max_reward, var_reward, total_reward),
                      "Episode rewards: {} --".format(
                           total_reward.astype(int)),
                      "Avg-time per step: {:.3f}ms, step {}/{}".format(avg_time, s+1, steps),
                      "-- E.T.A: {} min {} sec".format(eta_seconds // 60, eta_seconds % 60),
                      end="")

            # Pass new step
            env.update()

            elapsed = (time.time() - now) * 1000
            if avg_time is None:
                avg_time = elapsed
            else:
                avg_time = 0.99 * avg_time + 0.01 * elapsed

            if visualize_episode:
                states.append(env.save_state())

        if visualize_episode:
            if first_save:
                previous_states = []
                first_save = False
            else:
                previous_states = pickle.load(open("saved/" + save_file_name, "rb"))

            pickle.dump(previous_states + states, open("saved/" + save_file_name, "wb"))

            del states
            del previous_states
            states = []

        gc.collect()

        agent.epsilon = max(min_epsilon,  min(max_epsilon, 1.0 - math.log10((episode+1)/5)))
        print('\n Epsilon : ', agent.epsilon)

    if save_model and training:
        date = datetime.datetime.now()
        model_name = str(date.day) + '_' + str(date.month) + '_' + str(date.hour) + '_' + agent.name + '.h5'
        agent.save_model(model_name)



if __name__ == '__main__':
    if not only_visualize:
        main()

    visualiser = Visualizer()
    visualiser.big_dim = 900
    visualiser.visualize(save_file_name)
