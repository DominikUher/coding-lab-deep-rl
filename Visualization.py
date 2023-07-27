import matplotlib.pyplot as plt
import time
import random

# make sure to initialize the environment so that the (validation) episodes are the exact same as
# the ones that achieved the high score
from environment_CNN import Environment
env = Environment(variant=0, data_dir='./data')

def visualize_episode(actions: list()):
    env.reset(mode='validation')
    for step in range(env.episode_steps):
        action = actions[step]
        agent_location = env.agent_loc
        target_location = env.target_loc
        item_locations = env.item_locs
        item_times = env.item_times
        draw_grid(agent_location, target_location, item_locations, item_times)
        time.sleep(1)
        plt.close(plt.gcf())
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

    agent_rect = plt.Rectangle((agent_x, agent_y), 1, 1, facecolor="blue")
    ax.add_patch(agent_rect)

    target_rect = plt.Rectangle((target_x, target_y), 1, 1, facecolor=(0.2, 0.2, 0.2))
    ax.add_patch(target_rect)

    plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.show()


if __name__ == '__main__':
    # example actions, replace with yours
    actions = [random.randrange(5) for _ in range(200)]
    visualize_episode(actions=actions)
