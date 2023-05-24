from environment_dist_possible import Environment
#from ACER import *
from PPOtopGUN import *
# TODO: parse arguments
...
seed = 10
data_dir = './data'
variant = 0
env = Environment(variant, data_dir)

algo = 'PPO'

# TODO: execute training

if __name__ == '__main__':

    if algo == 'PPO':
        algorithm = PPO(variant=variant, lr_actor=0.0005, lr_critic_1=0.0005, lr_critic_2=0.0005,
                        return_lambda=0.5, gamma=0.95, clip_epsilon=0.15, episode_steps=200, no_of_actors=30,
                        actor_updates_per_episode=200, critic_updates_per_episode=200, clip_annealing_factor=0.99)
        algorithm.train_ppo()

#    if algo == 'ACER':
#        algorithm = ACER(variant=variant, data_dir='./data', lr_actor=0.0005, gamma=0.97, buffer_size=400, batch_size=200,
#                 no_of_actors=40, lr_global=0.0001)
#        algorithm.trainACER()
