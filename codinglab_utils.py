from PPOtopGUN import PPO
from main_CNN import PPO as CNN_PPO
from environment_dist_possible import Environment as Greedy
from environment_CNN import Environment as Neutral

def make_algorithm(algorithm_name, hyperparameters, improvements):
    params_without_env = {k: hyperparameters[k] for k in set(list(hyperparameters.keys())) - {'environment'}}
    match algorithm_name:
        case 'PPO':
            print(f'Starting training for {algorithm_name} with params {params_without_env}')
            return PPO.from_dict(hyperparameters)
        case 'CNN_PPO':
            return CNN_PPO.from_dict(hyperparameters)
        case _:
            return PPO.from_dict(hyperparameters)

def make_environment(environment_name, variant, data_dir):
    match environment_name:
        case 'Greedy':
            return Greedy(variant, data_dir)
        case 'Neutral':
            print(f'Chosen observation type: {environment_name}')
            return Neutral(variant, data_dir)
        case _:
            return Neutral(variant, data_dir)
        
def calculate_input_shape(environment_name):
    match environment_name:
        case 'Greedy':
            return 13
        case 'Neutral':
            return 125
        case _:
            return 125