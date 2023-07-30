import numpy as np

def make_algorithm(algorithm_name, hyperparameters, improvements):
    from PPOtopGUN import PPO
    from main_CNN import PPO as CNN_PPO

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
    from environment_dist_possible import Environment as Greedy
    from environment_CNN_moving import Environment as ImageLike
    from environment import Environment as GreedyN

    match environment_name:
        case s if s.startswith('NGreedy'):
            return GreedyN(variant, data_dir, int(s[-1:]))
        case 'Greedy':
            return Greedy(variant, data_dir)
        case 'ImageLike':
            return ImageLike(variant, data_dir)
        case _:
            return ImageLike(variant, data_dir)
        

def calculate_input_shape(environment_name):
    match environment_name:
        case n if n.startswith('NGreedy'):
            return int(n[-1:])*4+9
        case 'Greedy':
            return 13
        case 'ImageLike':
            return 125
        case _:
            return 125


def calculate_net_reward(reward, item_loc, agent_loc, target_loc):
    net_reward = reward - (np.abs(item_loc[0] - agent_loc[0]) + np.abs(item_loc[1] - agent_loc[1]) + np.abs(item_loc[0] - target_loc[0]) + np.abs(item_loc[1] - target_loc[1]))
    return net_reward