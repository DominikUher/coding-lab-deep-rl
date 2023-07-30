import os
import random
import numpy as np
import tensorflow as tf
from codinglab_utils import make_algorithm, make_environment, calculate_input_shape

# Setting seeds for reproducability
seed = 10
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


# Declaring algorithm and environment parameters
data_dir = './data'         # Only change if data directory changed
variant = 0                 # Possible values: 0, 1, 2
observation = 'NGreedy3'    # Possible values: 'Greedy', 'Neutral', 'ImageLike', 'NGreedyX' (with X in [1-9])
algorithm_name = 'PPO'      # Possible values: 'PPO', 'CNN_PPO'
algorithm_improvements = {  # Choose which improvements to use (to be implemented)
    'clip_ratio_annealing': True,
    'tanh_activations': True,
    'boltzmann_exploration': False,
    'noisy_nets': False
}

algorithm_parameters = {    # Choose hyperparameters here
    'seed': seed,
    'observation': observation,
    'environment': make_environment(observation, variant, data_dir),
    'variant': variant,
    'input_shape': calculate_input_shape(observation),
    'hidden_size': 256,
    'early_stopping': 20,
    'validation_after_episodes': 5,
    'lr_actor': 0.001,
    'lr_critic_1': 0.005,
    'lr_critic_2': 0.005,
    'return_lambda': 0.875,
    'gamma': 0.9,
    'clip_epsilon': 0.05,
    'episode_steps': 200,
    'no_of_actors': 10,
    'actor_updates_per_episode': 100,
    'critic_updates_per_episode': 100,
    'clip_annealing_factor': 0.99
}

best_parameters = algorithm_parameters

def line_search(paramater_name, values):
    print(f'Now searching for best value of {paramater_name}')
    best_score = -np.inf
    for value in values:
        algorithm_parameters[paramater_name] = value
        algorithm = make_algorithm(algorithm_name, best_parameters, algorithm_improvements)
        algorithm.train()
        score = algorithm.get_best_score()
        if score > best_score:
            best_score = score
            best_parameters[paramater_name] = value


line_search('return_lambda', [0.875, 0.9, 0.95, 0.99])