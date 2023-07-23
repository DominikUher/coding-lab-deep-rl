from PPOtopGUN import PPO
from main_CNN import PPO as CNN_PPO
from environment_dist_possible import Environment as Greedy
from environment_CNN import Environment as Neutral

def make_algorithm(algorithm_name, hyperparameters, improvements):
    match algorithm_name:
        case 'PPO':
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