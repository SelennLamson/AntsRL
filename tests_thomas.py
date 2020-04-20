from gui.visualize import Visualizer

# -------------------------------------------
#               Main parameters
# -------------------------------------------
save_file_name = "collect_agent.arl"
# -------------------------------------------

if __name__ == '__main__':
    visualiser = Visualizer()
    visualiser.big_dim = 900
    visualiser.visualize(save_file_name)
