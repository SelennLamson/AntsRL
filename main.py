import pickle
from tqdm import tqdm

from gui.visualize import Visualizer

from environment.RL_api import RLApi
from generator.environment_generator import EnvironmentGenerator
from generator.map_generators import *
from agent.random_agent import RandomAgent
from agent.DQN_agent import DQNAgent
import tensorflow as tf

AGGREGATE_STATS_EVERY = 5

def main():
    save_file_name = "random_agent.arl"

    episodes = 30
    steps = 200
    n_ants = 1
    epsilon = 0.1
    states = []

    generator = EnvironmentGenerator(w=200,
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

    agent = DQNAgent()
    ep_rewards = []

    print("Starting simulation...")
    for episode in range(episodes):
        env = generator.generate(api)
        print('Starting epoch {}...'.format(episode))

        obs, state = api.observation()
        episode_reward = 0

        for s in tqdm(range(steps)):
            #print('Timesteps n°{}'.format(s))
            if episodes>20:
                states.append(env.save_state())

            if np.random.random() > epsilon:
                # Ask network for next action
                value = np.argmax(agent.get_qs(obs))
                action = (1, value, np.zeros(n_ants), np.zeros(n_ants))

            else:
                # Random turn
                action = (1, np.random.randint(0, 5), np.zeros(n_ants), np.zeros(n_ants))
            # Execute the action
            new_state, reward, done = api.step(*action)
            # Add the reward to total reward of episode
            episode_reward += reward
            env.update()
            # Update replay memory with new action and states
            agent.update_replay_memory((obs, action[0], reward, new_state, done))
            # Train the neural network
            agent.train(done, s)
            # Set obs to the new state
            obs = new_state

        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            tf.summary.scalar("my_metric", 0.5, step=episode)
            agent.tensorboard.writer.flush()


    pickle.dump(states, open("saved/" + save_file_name, "wb"))

    # VISUALIZE THE EPISODE
    visualizer.visualize(save_file_name)


if __name__ == '__main__':
    main()
