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
observation = 'Greedy5'      # Possible values: 'Greedy', 'Neutral', 'ImageLike'
algorithm_name = 'PPO'      # Possible values: 'PPO', 'CNN_PPO'
algorithm_improvements = {  # Choose which improvements to use
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
    'lr_actor': 0.0001,
    'lr_critic_1': 0.0001,
    'lr_critic_2': 0.0001,
    'return_lambda': 0.75,
    'gamma': 0.9,
    'clip_epsilon': 0.05,
    'episode_steps': 200,
    'no_of_actors': 5,
    'actor_updates_per_episode': 100,
    'critic_updates_per_episode': 100,
    'clip_annealing_factor': 0.99
}

best_parameters = algorithm_parameters

def line_search(paramater_name, values):
    for value in values:
        best_score = -np.inf
        algorithm_parameters[paramater_name] = value
        algorithm = make_algorithm(algorithm_name, best_parameters, algorithm_improvements)
        algorithm.train()
        score = algorithm.get_best_score()
        if score > best_score:
            best_score = score
            best_parameters[paramater_name] = value

line_search('actor_updates_per_episode', [1, 10, 100])
line_search('critic_updates_per_episode', [1, 10, 100])
line_search('lr_actor', [0.00001, 0.0001, 0.001])
line_search('lr_critic1', [0.00001, 0.0001, 0.001])
line_search('lr_critic2', [0.00001, 0.0001, 0.001])
line_search('return_lambda', [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])
line_search('gamma', [0.75, 0.875, 0.9, 0.95, 0.99])
line_search('clip_epsilon', [0.01, 0.05, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])
line_search('clip_annealing_factor', [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.9, 0.95, 0.99])
line_search('hidden_size', [32, 64, 128, 256, 512])
line_search('no_of_actors', [1, 10, 100])