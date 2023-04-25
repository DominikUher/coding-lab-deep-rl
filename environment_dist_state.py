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

#own imports
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
            self.eligible_cells = [(0,0), (0,1), (0,2), (0,3), (0,4),
                                   (1,0), (1,1), (1,2), (1,3), (1,4),
                                   (2,0), (2,1), (2,2), (2,3), (2,4),
                                   (3,0), (3,1), (3,2), (3,3), (3,4),
                                   (4,0), (4,1), (4,2), (4,3), (4,4)]
        else:
            self.eligible_cells = [(0,0),        (0,2), (0,3), (0,4),
                                   (1,0),        (1,2),        (1,4),
                                   (2,0),        (2,2),        (2,4),
                                   (3,0), (3,1), (3,2),        (3,4),
                                   (4,0), (4,1), (4,2),        (4,4)]

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
        dist_up = self.agent_loc[0]
        dist_down = 4 - self.agent_loc[0]
        dist_left = self.agent_loc[1]
        dist_right = 4 - self.agent_loc[1]
        dist_y_target = self.target_loc[0] - self.agent_loc[0]
        dist_x_target = self.target_loc[1] - self.agent_loc[1]
        if len(self.item_locs) == 0:
          #  dist_y_next_item = dist_x_next_item = dist_y_farthest_item = dist_x_farthest_item = time_next_item = time_farthest_item = -1
          dist_y_next_item = dist_x_next_item = dist_y_second_item = dist_x_second_item = time_next_item = time_second_item = -1
        else:
            item_distances = [(np.abs(x[0] - self.agent_loc[0]) + np.abs(x[1] - self.agent_loc[1])) for x in self.item_locs]
            next_item = self.item_locs[np.argmin(item_distances)]
            dist_y_next_item = next_item[0] - self.agent_loc[0]
            dist_x_next_item = next_item[1] - self.agent_loc[1]
            time_next_item = self.max_response_time - self.item_times[np.argmin(item_distances)]
            if len(self.item_locs) > 1:
                item_distances.pop(np.argmin(item_distances))
                second_item = self.item_locs[np.argmin(item_distances)]
            else:
                second_item = next_item
            dist_y_second_item = second_item[0] - self.agent_loc[0]
            dist_x_second_item = second_item[1] - self.agent_loc[1]
            time_second_item = self.max_response_time - self.item_times[np.argmin(item_distances)]
         #   dist_y_farthest_item = self.item_locs[np.argmax(item_distances)][0] - self.agent_loc[0]
         #   dist_x_farthest_item = self.item_locs[np.argmax(item_distances)][1] - self.agent_loc[1]
         #   time_farthest_item = self.max_response_time - self.item_times[np.argmax(item_distances)]

        if dist_y_target > 0:
            dist_up_target = -1
            dist_down_target = np.abs(dist_y_target)
        else:
            dist_up_target = np.abs(dist_y_target)
            dist_down_target = -1
        if dist_x_target > 0:
            dist_left_target = -1
            dist_right_target = np.abs(dist_x_target)
        else:
            dist_left_target = np.abs(dist_x_target)
            dist_right_target = -1

        if dist_y_next_item > 0:
            dist_up_next = -1
            dist_down_next = np.abs(dist_y_next_item)
        else:
            dist_up_next = np.abs(dist_y_next_item)
            dist_down_next = -1

        if dist_x_next_item > 0:
            dist_right_next = np.abs(dist_x_next_item)
            dist_left_next = -1
        else:
            dist_right_next = -1
            dist_left_next = np.abs(dist_x_next_item)

        if dist_y_second_item > 0:
            dist_up_sec = -1
            dist_down_sec = np.abs(dist_y_second_item)
        else:
            dist_up_sec = np.abs(dist_y_second_item)
            dist_down_sec = -1

        if dist_x_second_item > 0:
            dist_right_sec = np.abs(dist_x_second_item)
            dist_left_sec = -1
        else:
            dist_right_sec = -1
            dist_left_sec = np.abs(dist_x_second_item)

        if len(self.item_locs) == 0:
            dist_up_next = dist_down_next = dist_left_next = dist_right_next = -1
            dist_up_sec = dist_down_sec = dist_left_sec = dist_right_sec = -1
            time_next_item = -1
        if len(self.item_locs) == 1:
            dist_up_sec = dist_down_sec = dist_left_sec = dist_right_sec = -1
            time_second_item = -1


        if self.variant == 0:
            return tf.constant([dist_up, dist_down, dist_left, dist_right,
                       dist_up_target, dist_down_target, dist_left_target, dist_right_target,
                       dist_up_next, dist_down_next, dist_left_next, dist_right_next, time_next_item,
                      # dist_up_sec, dist_down_sec, dist_left_sec, dist_right_sec, time_second_item,
                       self.agent_capacity - self.agent_load], dtype=tf.float32)
        else:
            return tf.constant([dist_up, dist_down, dist_left, dist_right,
                       dist_up_target, dist_down_target, dist_left_target, dist_right_target,
                       dist_up_next, dist_down_next, dist_left_next, dist_right_next, time_next_item,
                       dist_up_sec, dist_down_sec, dist_left_sec, dist_right_sec, time_second_item,
                       self.agent_capacity - self.agent_load], dtype=tf.float32)

    # this led to the 204 reward
   #     return tf.constant([dist_up, dist_down, dist_left, dist_right,
   #                            dist_y_target, dist_x_target,
   #                         dist_y_next_item, dist_x_next_item, time_next_item,
   #                      #   dist_y_farthest_item, dist_x_farthest_item, time_farthest_item,
   #                         dist_y_second_item, dist_x_second_item, time_second_item,
   #                         self.agent_capacity - self.agent_load], dtype=tf.float32)


