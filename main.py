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
observation = 'Neutral'      # Possible values: 'Greedy', 'Neutral', 'Limited'?
algorithm_name = 'PPO'      # Possible values: 'PPO', 'CNN', 'ACER'
algorithm_improvements = {  # Choose which improvements to use
    'clip_ratio_annealing': True,
    'tanh_activations': True,
    'boltzmann_exploration': False,
    'noisy_nets': False
}
algorithm_parameters = {    # Choose hyperparameters here
    'seed': seed,
    'environment': make_environment(observation, variant, data_dir),
    'variant': variant,
    'input_shape': calculate_input_shape(observation),
    'hidden_size': 256,
    'early_stopping': 1,
    'lr_actor': 0.001,
    'lr_critic_1': 0.001,
    'lr_critic_2': 0.001,
    'return_lambda': 0.75,
    'gamma': 0.9,
    'clip_epsilon': 0.05,
    'episode_steps': 200,
    'no_of_actors': 30,
    'actor_updates_per_episode': 200,
    'critic_updates_per_episode': 200,
    'clip_annealing_factor': 0.99
}

for lr in [0.0005, 0.001, 0.005]:
    algorithm_parameters['lr_actor'] = lr
    algorithm_parameters['lr_critic_1'] = lr
    algorithm_parameters['lr_critic_2'] = lr

    # Initializing algorithm and environment
    algorithm = make_algorithm(algorithm_name, algorithm_parameters, algorithm_improvements)

    # Train algorithm
    algorithm.train()

'''
for e in [0.05, 0.1, 0.25]:
    for g in [0.9, 0.95, 0.99]:
        for l in [0.25, 0.5, 0.75]:
            for lr in [0.00001, 0.0001, 0.001]:
                algorithm_parameters['clip_epsilon'] = e
                algorithm_parameters['gamma'] = g
                algorithm_parameters['return_lambda'] = l
                algorithm_parameters['lr_actor'] = lr
                algorithm_parameters['lr_critic_1'] = lr
                algorithm_parameters['lr_critic_2'] = lr

                # Initializing algorithm and environment
                algorithm = make_algorithm(algorithm_name, algorithm_parameters, algorithm_improvements)

                # Train algorithm
                algorithm.train()
'''
