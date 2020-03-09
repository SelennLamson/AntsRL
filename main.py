import pickle
from tqdm import tqdm
import datetime

from gui.visualize import Visualizer

from environment.RL_api import RLApi
from generator.environment_generator import EnvironmentGenerator
from generator.map_generators import *
from agent.random_agent import RandomAgent
from agent.Explore_Agent import ExploreAgent
import tensorflow as tf

AGGREGATE_STATS_EVERY = 5
SAVE_MODEL = False

TRAINING = False
USE_MODEL = "6_3_11_CNN.h5"

def main():
    save_file_name = "random_agent.arl"

    episodes = 10
    steps = 500
    n_ants = 50
    epsilon = 0.1
    states = []

    generator = EnvironmentGenerator(w=100,
                                     h=100,
                                     n_ants=n_ants,
                                     n_pheromones=2,
                                     n_rocks=0,
                                     food_generator=CirclesGenerator(10, 5, 10),
                                     walls_generator=PerlinGenerator(scale=22.0, density=0.05),
                                     max_steps=steps)
    api = RLApi(max_speed=1,
                max_rot_speed=45,
                carry_speed_reduction=0.05,
                backward_speed_reduction=0.5)
    api.save_perceptive_field = True
    visualizer = Visualizer()

    agent = ExploreAgent(n_ants, use_trained_model=USE_MODEL)
    ep_rewards = []

    print("Starting simulation...")
    for episode in range(episodes):
        env = generator.generate(api)
        api.ants.activate_all_pheromones(np.ones((api.ants.n_ants, 2)) * 10)
        print('\nStarting epoch {}...'.format(episode))

        obs, state = api.observation()
        episode_reward = np.zeros(n_ants)

        for s in range(steps):
            #print('Timesteps n°{}'.format(s))
            if (episode + 1) % 10 == 0 or episode == 0:
                states.append(env.save_state())

            if np.random.random() > epsilon:
                # Ask network for next action
                value = np.argmax(agent.get_qs(obs), axis=1)
                action = (np.ones(n_ants), value, np.zeros(n_ants), np.zeros(n_ants))

            else:
                # Random turn
                action = (np.ones(n_ants), np.random.randint(low=0, high=3, size=n_ants), np.zeros(n_ants), np.zeros(n_ants))
            # Execute the action
            new_state, reward, done = api.step(*action)
            # Add the reward to total reward of episode
            episode_reward += reward
            env.update()
            # Update replay memory with new action and states
            for i_ant in range(n_ants):
                agent.update_replay_memory((obs[i_ant], action[1][i_ant], reward[i_ant], new_state[i_ant], done))

            if TRAINING:
                # Train the neural network
                agent.train(done, s)
            # Set obs to the new state
            obs = new_state

        ep_rewards.append(episode_reward)
        print('\nReward for this episode :', episode_reward)


    pickle.dump(states, open("saved/" + save_file_name, "wb"))

    if SAVE_MODEL:
        date = datetime.datetime.now()
        model_name = str(date.day) + '_' + str(date.month) + '_' + str(date.hour) + '_' + agent.name+'.h5'
        agent.save_model(model_name)

    # VISUALIZE THE EPISODE
    visualizer.big_dim = 800
    visualizer.visualize(save_file_name)


if __name__ == '__main__':
    #visualiser = Visualizer()
    #visualiser.big_dim = 800
    #visualiser.visualize()
    main()
