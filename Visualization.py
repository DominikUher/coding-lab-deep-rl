import matplotlib.pyplot as plt
import time
import random

# make sure to initialize the environment so that the (validation) episodes are the exact same as
# the ones that achieved the high score
from environment_CNN_moving import Environment
env = Environment(variant=0, data_dir='./data')

def visualize_episode(actions: list(), startAt):
    env.reset('validation', startAt)
    plt.ion()
    for step in range(env.episode_steps):
        action = actions[step]
        agent_location = env.agent_loc
        target_location = env.target_loc
        item_locations = env.item_locs
        item_times = env.item_times
        draw_grid(agent_location, target_location, item_locations, item_times)
        plt.pause(1)
        env.step(action)


def draw_grid(agent_location, target_location, item_locations, item_times):

    fig, ax = plt.subplots()
    ax.set_xticks(range(6), minor=False)
    ax.set_yticks(range(6), minor=False)
    ax.grid(which='both', color='black', linewidth=1)

    agent_y, agent_x = agent_location
    target_y, target_x = target_location
    for i, tp in enumerate(item_locations):
        item_time = env.item_times[i] / env.max_response_time
        item_y = tp[0]
        item_x = tp[1]
        c1 = 240.0 / 255
        c2 = 248.0 / 255
        c3 = 255.0 / 255
        color = (c1, c2 * item_time, c3 * item_time)
        rect = plt.Rectangle((item_x, item_y), 1, 1, facecolor=color)
        ax.add_patch(rect)
    
    target_rect = plt.Rectangle((target_x, target_y), 1, 1, facecolor=(0.2, 0.2, 0.2))
    ax.add_patch(target_rect)

    agent_rect = plt.Rectangle((agent_x+0.25, agent_y+0.25), 0.5, 0.5, facecolor="blue")
    ax.add_patch(agent_rect)

    plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.gca().set_aspect('equal', adjustable='box')


if __name__ == '__main__':
    # example actions, replace with yours
    actions = [3, 3, 4, 4, 4, 3, 1, 4, 0, 1, 3, 2, 1, 3, 2, 3, 4, 4, 3, 2, 4, 3, 0, 4, 4, 4, 4, 0, 4, 4, 0, 3, 3, 0, 2, 0, 0, 4, 4, 3, 4, 4, 0, 3, 1, 3, 3, 4, 4, 0, 1, 0, 3, 0, 1, 3, 3, 0, 4, 2, 4, 1, 4, 2, 4, 1, 4, 3, 2, 3, 1, 0, 4, 0, 4, 1, 1, 3, 1, 3, 3, 4, 1, 3, 3, 2, 3, 4, 1, 3, 4, 0, 3, 4, 4, 4, 3, 3, 3, 0, 3, 2, 3, 3, 1, 3, 4, 3, 3, 3, 3, 4, 3, 4, 4, 3, 2, 3, 2, 3, 1, 4, 3, 4, 1, 3, 3, 4, 0, 2, 1, 4, 0, 3, 1, 1, 2, 2, 4, 1, 3, 3, 4, 3, 4, 4, 0, 0, 1, 1, 4, 4, 3, 4, 3, 3, 4, 1, 0, 3, 4, 1, 3, 3, 2, 2, 3, 3, 2, 4, 2, 3, 0, 3, 4, 2, 1, 1, 2, 2, 3, 2, 2, 0, 4, 1, 4, 1, 4, 2, 0, 0, 3, 2, 0, 2, 1, 4, 0, 0]
    visualize_episode(actions, 121)
