# actions: 0 (nothing), 1 (up), 2 (right), 3 (down), 4 (left)

# positions in grid:
# - (0,0) is upper left corner
# - first index is vertical (increasing from top to bottom)
# - second index is horizontal (increasing from left to right)

# if new item appears in a cell into which the agent moves/at which the agent stays in the same time step,
# it is not picked up (if agent wants to pick it up, it has to stay in the cell in the next time step)

import random
import pandas as pd
from copy import deepcopy
from itertools import compress

# own imports
import numpy as np
import tensorflow as tf


class Environment(object):
    def __init__(self, variant, data_dir):
        self.variant = variant
        self.vertical_cell_count = 5
        self.horizontal_cell_count = 5
        self.vertical_idx_target = 2
        self.horizontal_idx_target = 0
        self.target_loc = (self.vertical_idx_target, self.horizontal_idx_target)
        self.episode_steps = 200
        self.max_response_time = 15 if self.variant == 2 else 10
        self.reward = 25 if self.variant == 2 else 15
        self.data_dir = data_dir

        self.training_episodes = pd.read_csv(self.data_dir + f'/variant_{self.variant}/training_episodes.csv')
        self.training_episodes = self.training_episodes.training_episodes.tolist()
        self.validation_episodes = pd.read_csv(self.data_dir + f'/variant_{self.variant}/validation_episodes.csv')
        self.validation_episodes = self.validation_episodes.validation_episodes.tolist()
        self.test_episodes = pd.read_csv(self.data_dir + f'/variant_{self.variant}/test_episodes.csv')
        self.test_episodes = self.test_episodes.test_episodes.tolist()

        self.remaining_training_episodes = deepcopy(self.training_episodes)
        self.validation_episode_counter = 0

        if self.variant == 0 or self.variant == 2:
            self.agent_capacity = 1
        else:
            self.agent_capacity = 3

        if self.variant == 0 or self.variant == 1:
            self.eligible_cells = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
                                   (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
                                   (2, 0), (2, 1), (2, 2), (2, 3), (2, 4),
                                   (3, 0), (3, 1), (3, 2), (3, 3), (3, 4),
                                   (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]
        else:
            self.eligible_cells = [(0, 0), (0, 2), (0, 3), (0, 4),
                                   (1, 0), (1, 2), (1, 4),
                                   (2, 0), (2, 2), (2, 4),
                                   (3, 0), (3, 1), (3, 2), (3, 4),
                                   (4, 0), (4, 1), (4, 2), (4, 4)]

    # initialize a new episode (specify if training, validation, or testing via the mode argument)
    def reset(self, mode):
        modes = ['training', 'validation', 'testing']
        if mode not in modes:
            raise ValueError('Invalid mode. Expected one of: %s' % modes)

        self.step_count = 0
        self.agent_loc = (self.vertical_idx_target, self.horizontal_idx_target)
        self.agent_load = 0  # number of items loaded (0 or 1, except for first extension, where it can be 0,1,2,3)
        self.item_locs = []
        self.item_times = []

        if mode == "testing":
            episode = self.test_episodes[0]
            self.test_episodes.remove(episode)
        elif mode == "validation":
            episode = self.validation_episodes[self.validation_episode_counter]
            self.validation_episode_counter = (self.validation_episode_counter + 1) % 100
        else:
            if not self.remaining_training_episodes:
                self.remaining_training_episodes = deepcopy(self.training_episodes)
            episode = random.choice(self.remaining_training_episodes)
            self.remaining_training_episodes.remove(episode)
        self.data = pd.read_csv(self.data_dir + f'/variant_{self.variant}/episode_data/episode_{episode:03d}.csv',
                                index_col=0)

        return self.get_obs()

    # take one environment step based on the action act
    def step(self, act):
        self.step_count += 1

        rew = 0

        # done signal (1 if episode ends, 0 if not)
        if self.step_count == self.episode_steps:
            done = 1
        else:
            done = 0

        # agent movement
        if act != 0:
            if act == 1:  # up
                new_loc = (self.agent_loc[0] - 1, self.agent_loc[1])
            elif act == 2:  # right
                new_loc = (self.agent_loc[0], self.agent_loc[1] + 1)
            elif act == 3:  # down
                new_loc = (self.agent_loc[0] + 1, self.agent_loc[1])
            elif act == 4:  # left
                new_loc = (self.agent_loc[0], self.agent_loc[1] - 1)

            if new_loc in self.eligible_cells:
                self.agent_loc = new_loc
                rew += -1

        # item pick-up
        if (self.agent_load < self.agent_capacity) and (self.agent_loc in self.item_locs):
            self.agent_load += 1
            idx = self.item_locs.index(self.agent_loc)
            self.item_locs.pop(idx)
            self.item_times.pop(idx)
            rew += self.reward / 2

        # item drop-off
        if self.agent_loc == self.target_loc:
            rew += self.agent_load * self.reward / 2
            self.agent_load = 0

        # track how long ago items appeared
        self.item_times = [i + 1 for i in self.item_times]

        # remove items for which max response time is reached
        mask = [i < self.max_response_time for i in self.item_times]
        self.item_locs = list(compress(self.item_locs, mask))
        self.item_times = list(compress(self.item_times, mask))

        # add items which appear in the current time step
        new_items = self.data[self.data.step == self.step_count]
        new_items = list(zip(new_items.vertical_idx, new_items.horizontal_idx))
        new_items = [i for i in new_items if i not in self.item_locs]  # not more than one item per cell
        self.item_locs += new_items
        self.item_times += [0] * len(new_items)

        # get new observation
        next_obs = self.get_obs()

        return rew, next_obs, done

    # TODO: implement function that gives the input features for the neural network(s)
    #       based on the current state of the environment
    def get_obs(self):

        state_rep = np.zeros(39, dtype=np.float32)
        state_rep[:] = -1
        # Agent distances to grid walls
        state_rep[0] = self.agent_loc[0]
        state_rep[1] = 4 - self.agent_loc[0]
        state_rep[2] = self.agent_loc[1]
        state_rep[3] = 4 - self.agent_loc[1]
        state_rep[4] = self.agent_capacity - self.agent_load

        # Agent distances to target
        dist_y_target = self.target_loc[0] - self.agent_loc[0]
        dist_x_target = self.target_loc[1] - self.agent_loc[1]
        state_rep[5], state_rep[6], state_rep[7], state_rep[8] = self.item_data(dist_y_target, dist_x_target)

        if len(self.item_times) > 0:

            # create a list (eligible) with all items that can be reached before they disappear
            items_time_left = [self.max_response_time - time for time in self.item_times]
            eligible = []
            eligible_t = []
            for i in range(len(self.item_times)):
                if items_time_left[i] - (np.abs(self.item_locs[i][0] - self.agent_loc[0]) +
                                         np.abs(self.item_locs[i][1] - self.agent_loc[1])) >= 0:
                    eligible.append(self.item_locs[i])
                    eligible_t.append(self.item_times[i])

            # calculate the net reward of all those reachable items
            net_reward = [(self.reward - (np.abs(item[0] - self.agent_loc[0]) + np.abs(item[1] - self.agent_loc[1]) +
                                          np.abs(item[0] - self.target_loc[0]) + np.abs(item[1] - self.target_loc[1])))
                          for item in eligible]

            k = np.minimum(len(eligible), 6)  # 15 - 4 * 2 - 2 * 4 = -1 -> sufficient
            # pick the one with the highest net reward
            for item in range(k):
                next_item = eligible[np.argmax(net_reward)]
                item_time = eligible_t[np.argmax(net_reward)]
                dist_y = next_item[0] - self.agent_loc[0]
                dist_x = next_item[1] - self.agent_loc[1]
                net_reward.pop(np.argmax(net_reward))
                state_rep[5 * item + 9], state_rep[5 * item + 10], state_rep[5 * item + 11], state_rep[5 * item + 12] = self.item_data(dist_y, dist_x)
                state_rep[5 * item + 13] = item_time

        # return agent-wall, agent-target, agent-next_item distances plus the remaining agent capacity as state
        return tf.constant(state_rep, dtype=tf.float32)

    def item_data(self, dist_y, dist_x):
        if dist_y > 0:
            dist_up = -1
            dist_down = np.abs(dist_y)
        elif dist_y < 0:
            dist_up = np.abs(dist_y)
            dist_down = -1
        else:
            dist_up = dist_down = 0

        if dist_x > 0:
            dist_right = np.abs(dist_x)
            dist_left = -1
        elif dist_x < 0:
            dist_right = -1
            dist_left = np.abs(dist_x)
        else:
            dist_right = dist_left = 0

        return dist_up, dist_down, dist_left, dist_right

