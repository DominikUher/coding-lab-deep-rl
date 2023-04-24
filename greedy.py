# greedy heuristic: average reward over 100 validation/testing episodes
#   - variant 0: validation 218.850, testing 221.770
#   - variant 1: validation 388.895, testing 385.425
#   - variant 2: validation 252.535, testing 250.685


import numpy as np
from itertools import compress
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from environment import Environment


variant = 2  # problem variant (0 for base variant, 1 for first extension, 2 for second extension)
mode = 'testing'  # validation or testing
data_dir = './data'  # relative path to data directory (e.g., './data', not './data/variant_0')

# set parameters
max_episode_steps = 200
max_response_time = 15 if variant == 2 else 10
reward = 25 if variant == 2 else 15

# initialize environment
env = Environment(variant, data_dir)


### get shortest paths and distances between any two nodes for each of the problem variants

# build matrix with information which grid cells are neighbors
if variant == 0 or variant == 1:
    # cell indices:
    # |  0 |  1 |  2 |  3 |  4 |
    # |  5 |  6 |  7 |  8 |  9 |
    # | 10 | 11 | 12 | 13 | 14 |
    # | 15 | 16 | 17 | 18 | 19 |
    # | 20 | 21 | 22 | 23 | 24 |

    neighbor_matrix = np.zeros((25, 25), int)
    for i in range(25):
        for j in range(i+1, 25):
            i_vert = int(i / 5)
            i_hori = i % 5
            j_vert = int(j / 5)
            j_hori = j % 5
            dist_vert = j_vert - i_vert
            dist_hori = j_hori - i_hori
            if (dist_vert == 0 and dist_hori == 1) or (dist_vert == 1 and dist_hori == 0):
                neighbor_matrix[i, j] = 1

else:
    # cell indices:
    # |  0 |  X |  7 | 12 | 13 |
    # |  1 |  X |  8 |  X | 14 |
    # |  2 |  X |  9 |  X | 15 |
    # |  3 |  5 | 10 |  X | 16 |
    # |  4 |  6 | 11 |  X | 17 |

    mapping = [(0,0), (1,0), (2,0), (3,0), (4,0),
               (3,1), (4,1),
               (0,2), (1,2), (2,2), (3,2), (4,2),
               (0,3),
               (0,4), (1,4), (2,4), (3,4), (4,4)]

    neighbor_matrix = np.zeros((18, 18), int)
    neighbor_matrix[0, 1] = 1
    neighbor_matrix[1, 2] = 1
    neighbor_matrix[2, 3] = 1
    neighbor_matrix[3, 4] = 1
    neighbor_matrix[3, 5] = 1
    neighbor_matrix[4, 6] = 1
    neighbor_matrix[5, 6] = 1
    neighbor_matrix[5, 10] = 1
    neighbor_matrix[6, 11] = 1
    neighbor_matrix[7, 8] = 1
    neighbor_matrix[7, 12] = 1
    neighbor_matrix[8, 9] = 1
    neighbor_matrix[9, 10] = 1
    neighbor_matrix[10, 11] = 1
    neighbor_matrix[12, 13] = 1
    neighbor_matrix[13, 14] = 1
    neighbor_matrix[14, 15] = 1
    neighbor_matrix[15, 16] = 1
    neighbor_matrix[16, 17] = 1

# run Dijkstra's algorithm
graph = csr_matrix(neighbor_matrix)
dist_matrix, predecessors = dijkstra(csgraph=graph, directed=False, return_predecessors=True, unweighted=True)

# derive information which is the next node on path between any two nodes
successors = np.zeros_like(predecessors)
for i in range(np.shape(successors)[0]):
    for j in range(np.shape(successors)[1]):
        if i == j:
            successors[i,j] = i
        else:
            not_finished = True
            predecessor = j
            while not_finished:
                new_predecessor = predecessors[i, predecessor]
                if new_predecessor == i:
                    successors[i,j] = predecessor
                    not_finished = False
                else:
                    predecessor = new_predecessor


### helper functions

# convert coordinates (x, y) to the corresponding cell index z
# (cell indices as defined above, coordinates as defined in the environment)
def convert_coord_to_idx(coord):
    if variant == 0 or variant == 1:
        idx = coord[0] * 5 + coord[1]
    else:
        idx = mapping.index(coord)
    return idx

# convert cell index z to the corresponding coordinates (x, y)
def convert_idx_to_coord(idx):
    if variant == 0 or variant == 1:
        coord_vert = int(idx / 5)
        coord_hori = idx % 5
        coord = (coord_vert, coord_hori)
    else:
        coord = mapping[idx]
    return coord

# get the distance between two cells given their coordinates based on the previously computed distance matrix
def get_dist_from_coord(coord1, coord2):
    idx1 = convert_coord_to_idx(coord1)
    idx2 = convert_coord_to_idx(coord2)
    return dist_matrix[idx1, idx2]

# given the coordinates of a start cell cur_coord and the coordinates of a target cell tar_coord, get the action that
# should be executed to reach the next node on the shortest path from the start cell to the target cell
def get_action_from_coord(cur_coord, tar_coord):
    if cur_coord == tar_coord:
        act = 0

    else:
        cur_idx = convert_coord_to_idx(cur_coord)
        tar_idx = convert_coord_to_idx(tar_coord)
        next_idx = successors[cur_idx, tar_idx]
        next_coord = convert_idx_to_coord(next_idx)

        vertical_delta = cur_coord[0] - next_coord[0]
        if vertical_delta != 0:
            if vertical_delta > 0:
                act = 1
            else:
                act = 3
        else:
            horizontal_delta = cur_coord[1] - next_coord[1]
            if horizontal_delta > 0:
                act = 4
            else:
                act = 2

    return act


### greedy policy

def greedy_policy():
    # if the agent has no free capacity, it moves to the target location
    if env.agent_load == env.agent_capacity:
        act = get_action_from_coord(env.agent_loc, env.target_loc)

    # if the agent has free capacity and there is an item available that can be reached before it disappears because the
    # maximum response time is reached, the agent moves towards that item if the item will lead to a positive profit;
    # if there is more than one such item, the one with the highest profit is chosen;
    # if there is no such item, the agent stays where it is
    else:
        act = 0
        if env.item_locs:
            dist = np.array([get_dist_from_coord(env.agent_loc, i) for i in env.item_locs])
            mask = dist <= max_response_time - np.array(env.item_times)
            item_locs = list(compress(env.item_locs, mask))
            if item_locs:
                dist = dist[mask]
                dist += np.array([get_dist_from_coord(i, env.target_loc) for i in item_locs])
                profit = reward - dist
                mask = profit > 0
                item_locs = list(compress(item_locs, mask))
                if item_locs:
                    profit = profit[mask]
                    idx = np.argmax(profit)
                    item_loc = item_locs[idx]
                    act = get_action_from_coord(env.agent_loc, item_loc)

    return act


### greedy policy evaluation for validation/testing data (specified by mode)

def evaluate_policy():
    reward = 0.
    for i in range(100):
        env.reset(mode)
        for j in range(max_episode_steps):
            act = greedy_policy()
            rew, _, _ = env.step(act)
            reward += rew
    avg_reward = reward / 100
    print(avg_reward)


### run greedy policy evaluation

if __name__ == '__main__':
    evaluate_policy()
