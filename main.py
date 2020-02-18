import pickle
from tqdm import tqdm

from gui.visualize import Visualizer

from environment.RL_api import RLApi
from generator.environment_generator import EnvironmentGenerator
from generator.map_generators import *
from agent.random_agent import RandomAgent

def main():
    save_file_name = "random_agent.arl"

    episodes = 1
    steps = 10
    states = []

    generator = EnvironmentGenerator(w=200,
                                     h=100,
                                     n_ants=1,
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

    agent = RandomAgent(n_action=4)

    print("Starting simulation...")
    for episode in tqdm(range(episodes)):
        env = generator.generate(api)
        print('Starting epoch {}...'.format(episode))
        for s in range(steps):

            states.append(env.save_state())

            obs, state = api.observation()

            action = agent.choose_action(obs)

            obs, reward, done = api.step(action)


            env.update()
            print(done)

    pickle.dump(states, open("saved/" + save_file_name, "wb"))

    # VISUALIZE THE EPISODE
    visualizer.visualize(save_file_name)


if __name__ == '__main__':
    main()
