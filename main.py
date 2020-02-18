import pickle
from tqdm import tqdm

from gui.visualize import Visualizer

from environment.RL_api import RLApi
from generator.environment_generator import EnvironmentGenerator
from generator.map_generators import *
from Agent.random_agent import RandomAgent

def main():
    save_file_name = "random_agent.arl"

    generator = EnvironmentGenerator(w=200,
                                     h=100,
                                     n_ants=1,
                                     n_pheromones=2,
                                     n_rocks=0,
                                     food_generator=CirclesGenerator(10, 5, 10),
                                     walls_generator=PerlinGenerator(scale=22.0, density=0.05))
    api = RLApi(max_speed=1,
                max_rot_speed=45,
                carry_speed_reduction=0.05,
                backward_speed_reduction=0.5)
    api.save_perceptive_field = True
    visualizer = Visualizer()

    episodes = 1
    steps = 100
    states = []

    agent = RandomAgent(n_action=4)

    print("Starting simulation...")
    for episode in tqdm(range(episodes)):
        env = generator.generate(api)
        print('Starting epoch {}...'.format(episode))
        for s in range(steps):

            states.append(env.save_state())

            obs, state = api.observation()

            print(obs.shape)
            print(obs)
            action = agent.choose_action(obs)

            api.step(action)

            env.update()

    pickle.dump(states, open("saved/" + save_file_name, "wb"))

    # VISUALIZE THE EPISODE
    visualizer.visualize(save_file_name)


if __name__ == '__main__':
    main()
