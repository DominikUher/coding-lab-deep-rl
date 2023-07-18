from PPOtopGUN import PPO
from main_CNN import PPO as CNN_PPO

def make_algorithm(algorithm_name, hyperparameters, improvements):
    match algorithm_name:
        case 'PPO':
            return PPO.from_dict(hyperparameters)
        case 'CNN_PPO':
            return CNN_PPO.from_dict(hyperparameters)

    return PPO.from_dict(hyperparameters)