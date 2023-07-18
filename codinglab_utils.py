from PPOtopGUN import PPO

def make_algorithm(algorithm_name, hyperparameters, improvements):
    match algorithm_name:
        case 'PPO':
            return PPO.from_dict(hyperparameters)

    return PPO.from_dict(hyperparameters)