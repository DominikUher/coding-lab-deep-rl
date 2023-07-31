import matplotlib.pyplot as plt
import time
import random

# make sure to initialize the environment so that the (validation) episodes are the exact same as
# the ones that achieved the high score
from environment_CNN_moving import Environment
variant = 2
env = Environment(variant=variant, data_dir='./data')


def visualize_episode(actions: list(), startAt):
    env.reset('validation', startAt)
    plt.ion()
    fig, ax = plt.subplots()
    total_reward = 0
    for step in range(env.episode_steps):
        plt.cla()
        ax.set_xticks(range(6), minor=False)
        ax.set_yticks(range(6), minor=False)
        ax.grid(which='both', color='black', linewidth=1)
        ax.set_title('step: {}  total reward: {}'.format(step+1, total_reward))
        action = actions[step]
        agent_location = env.agent_loc
        target_location = env.target_loc
        item_locations = env.item_locs
        item_times = env.item_times
        draw_grid(agent_location, target_location, item_locations, item_times, fig, ax)
        plt.pause(0.5)
        reward, _, _ = env.step(action)

        total_reward += reward


def draw_grid(agent_location, target_location, item_locations, item_times, fig, ax):
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
        rect = plt.Rectangle((item_x, item_y), 1, 1, facecolor=color, figure=fig)
        ax.add_patch(rect)
    if variant == 2:
        ax.add_patch(plt.Rectangle((1, 0), 1, 1, facecolor=(0, 0, 0), figure=fig))
        ax.add_patch(plt.Rectangle((1, 1), 1, 1, facecolor=(0, 0, 0), figure=fig))
        ax.add_patch(plt.Rectangle((1, 2), 1, 1, facecolor=(0, 0, 0), figure=fig))
        ax.add_patch(plt.Rectangle((3, 1), 1, 1, facecolor=(0, 0, 0), figure=fig))
        ax.add_patch(plt.Rectangle((3, 2), 1, 1, facecolor=(0, 0, 0), figure=fig))
        ax.add_patch(plt.Rectangle((3, 3), 1, 1, facecolor=(0, 0, 0), figure=fig))
        ax.add_patch(plt.Rectangle((3, 4), 1, 1, facecolor=(0, 0, 0), figure=fig))

    target_rect = plt.Rectangle((target_x, target_y), 1, 1, facecolor=(1, 216/255, 0), figure=fig)
    ax.add_patch(target_rect)

    agent_rect = plt.Circle((agent_x + 0.5, agent_y + 0.5), 0.25, facecolor="blue", figure=fig)
    ax.add_patch(agent_rect)

    plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.gca().invert_yaxis()
    fig.set_size_inches((5, 5))
    plt.gca().set_aspect('equal', adjustable='box')


if __name__ == '__main__':
    # example actions, replace with yours
    actions = [0, 0, 0, 2, 4, 3, 2, 2, 1, 1, 3, 3, 4, 4, 1, 3, 2, 2, 1, 1, 3, 3, 4, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 4, 2, 2, 4, 4, 4, 4, 4, 2, 3, 3, 2, 2, 4, 4, 1, 1, 3, 2, 2, 1, 1, 3, 3, 4, 4, 1, 4, 0, 3, 2, 2, 1, 3, 4, 4, 1, 3, 2, 3, 2, 1, 4, 4, 1, 4, 0, 0, 2, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 4, 4, 4, 4, 2, 0, 2, 0, 4, 0, 4, 0, 3, 2, 2, 4, 4, 1, 0, 0, 2, 4, 0, 4, 2, 4, 2, 4, 4, 4, 4, 4, 4, 0, 2, 2, 0, 4, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 2, 4, 4, 3, 4, 4, 4, 3, 3, 1, 4, 0, 4, 4, 0, 0, 0, 4, 1, 4, 4, 4, 4, 0, 3, 2, 2]
    visualize_episode(actions, 75)
