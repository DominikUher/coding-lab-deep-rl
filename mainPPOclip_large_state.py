# TODO: parse arguments

# best: 14 features, gamma 0.95, lambda 0.5, policy lr = critic lr = 0.001, clip ratio 0.15., converged after 50 episodes

# 14 features: gamma = 0.95, lambda = 0.7, policy lr = 0.0005, critic lr = 0.001, clip ratio = 0.15, reward = 225 at 303, reward = 220 at 282, 298 and 326

# set seed
seed = 10  # TODO: set seed to allow for reproducibility of results

import os
os.environ['PYTHONHASHSEED'] = str(seed)
import random
random.seed(seed)
import numpy as np
np.random.seed(seed)
import tensorflow as tf
tf.random.set_seed(seed)
import scipy

# initialize environment
#from environment import Environment # for small_state
#from Environments.environment_dist_possible import Environment
from Environments.environment_40_ft import Environment
#from environment_coord_state import Environment
data_dir = './data'  # TODO: specify relative path to data directory (e.g., './data', not './data/variant_0')
variant = 1  # TODO: specify problem variant (0 for base variant, 1 for first extension, 2 for second extension)
env = Environment(variant, data_dir)



class PPO_Algo:
    def __init__(self, agent, environment):

        self.agent = agent
        self.env = environment

    def trainPPO(self):

        rew, total_episodes = self.runPPO(mode='training')

        return rew, total_episodes

    def runPPO(self, mode='validation'):

        if mode == 'training':
            used_data = self.env.training_episodes
            no = 800
        elif mode == 'validation':
            used_data = self.env.validation_episodes
            no = 100
        else:
            used_data = self.env.test_episodes
            no = 100

        ep = 0
        rew = np.zeros((self.agent.actors, no))

        for episode in used_data:

            agent.initialize_buffers()

            for n in range(0, agent.actors):
                tot_rew = 0
                state = tf.reshape(self.env.reset(mode=mode), (1, self.agent.no_of_features))

                for t in range(self.env.episode_steps):

                    logit, action = agent.select_action(state) # Sample action and get the output of the NN

                    # Get new transition
                    reward, next_state, done = self.env.step(action[0].numpy())
                    next_state = tf.reshape(next_state, (1, self.agent.no_of_features))

                    if mode == 'training':
                        value_t = agent.critic(state) # Get value of current state
                        logprobability_t = logprobabilities(logit, action) # Transform output of NN into log-probability for numerical stability
                        agent.record(state, action, reward, value_t, logprobability_t, n, t) # Record transition for usage in weight update

                    tot_rew += reward
                    state = next_state

                    # Finish trajectory if reached to a terminal state
                    if done:
                        if mode == 'training':
                            agent.calc_advantage(n, t) # Calculate advantages
                        break
                rew[n, ep] = tot_rew

            if mode == 'training':

                # Prepare data to be used in weight update
                reward_buffer,states_buffer,actions_buffer,logprobas_buffer,advantages_buffer,return_buffer = agent.prepare_training_data()

            # Calculate weight updates for policy and value network
                for _ in range(agent.train_policy_epochs):
                    agent.train_policy(states_buffer, actions_buffer, logprobas_buffer, advantages_buffer)
                for _ in range(agent.train_value_function_epochs):
                    agent.train_value_function(states_buffer, return_buffer)

                if ep > 1:
                    tot_rew_avg = np.average(rew[:, ep-2:ep+1]) # take average over 3 most recent episodes and all actors
                    benchmark = [221, 390, 250]
                    if tot_rew_avg > benchmark[self.env.variant]:
                        print('Converged after ', ep, ' episodes with mean total reward of ', tot_rew_avg)
                        self.agent.actor.save_weights("ppo_actor_weights_var_" + str(self.env.variant) + "_" + str(benchmark[self.env.variant]))
                        self.agent.critic.save_weights("ppo_critic_weights_var_" + str(self.env.variant) + "_" + str(benchmark[self.env.variant]))
                        return rew[:, :ep+1], ep
                        break
            tot_rew_avg = np.average(rew[:, ep])
            print("episode: {}/{} | score: {}".format(
                ep + 1, no, tot_rew_avg))

            ep += 1
        return rew[:, :ep+1], ep



# Compute the log-probabilities of taking actions a by using the outputs of actor NN, mainly done for numerical stability purposes
def logprobabilities(logit, a):
    logprobabilities_all = tf.nn.log_softmax(logit)
    logprobability = tf.reduce_sum(tf.one_hot(a, 5) * logprobabilities_all, axis=1)
    return logprobability

# Discounted cumulative sums of vectors for computing sum of discounted rewards and advantage estimates
def discounted_cumulative_sums(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class PPO_Agent:
    def __init__(self):
        no_of_squares = 40#13 # 81
        no_of_fts = 1 # 1
        self.no_of_features = no_of_squares * no_of_fts
        self.no_of_actions = 5

        # Hyperparameters
        self.gamma = 0.95  # discount rate
        self.lam = 0.5# TD(lambda) weight
        self.clip_ratio = 0.15# Clipping ratio # 0.25 achieved 204 at 397

        self.actors = 30 # No. of parallel actors
        self.max_time_steps = 200  # Maximum time steps

        self.actor = self.nn_model(self.no_of_features, self.no_of_actions) # Create actor network
        self.train_policy_epochs = 80 # Define number of epochs for multiple weight update iterations within one episode
        self.critic = self.nn_model(self.no_of_features, 1) # Create critic network
        self.train_value_function_epochs = 80 # Define number of epochs for multiple weight update iterations within one episode

        policy_learning_rate = 0.0005 # 0.0005 achieved 204 at 397
        value_function_learning_rate = 0.0002 # 0.0005 achieved 204 at 397
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=policy_learning_rate)
        self.value_optimizer = tf.keras.optimizers.Adam(learning_rate=value_function_learning_rate)

    def initialize_buffers(self): # Initialize arrays for usage in weight updates
        self.advantages = np.zeros((self.actors, self.max_time_steps), dtype=np.float32)
        self.state_buffer = np.zeros((self.actors, self.max_time_steps, self.no_of_features), dtype=np.float32)
        self.action_buffer = np.zeros((self.actors, self.max_time_steps), dtype=np.int32)
        self.reward_buffer = np.zeros((self.actors, self.max_time_steps), dtype=np.float32)
        self.value_buffer = np.zeros((self.actors, self.max_time_steps), dtype=np.float32)
        self.logprobability_buffer = np.zeros((self.actors, self.max_time_steps), dtype=np.float32)
        self.return_buffer = np.zeros((self.actors, self.max_time_steps), dtype=np.float32)

    # Sample action based on policy
    @tf.function
    def select_action(self, state):
        logit = self.actor(state)
        action = tf.squeeze(tf.random.categorical(logit, 1), axis=1)
        return logit, action

    # Calculate advantages based on lambda-return for policy update and sum of discounted rewards for value updates
    def calc_advantage(self, n, T):
        # Î´ = r(s_t,a_t)+Î³V(s_{t+1})-V(s_t)
        deltas = self.reward_buffer[n, :-1] + self.gamma * self.value_buffer[n, 1:] - self.value_buffer[n, :-1]

        # A(s_t,a_t) = Q(s_t,a_t)-V(s_t) = ð”¼[r(s_t,a_t)+Î³V(s_{t+1})|s_t,a] - V(s_t) ~ G^Î»_t(s_t,a_t)-VÌ‚(s_t) = Sum_{k=t}^{T} (Î³Î»)^{k-t} Î´_k
        # First two equalities from lecture on policy gradients and advanced policy gradients / last equality from exercise 5 in TD(Î») exercise sheet
        self.advantages[n, :-1] = discounted_cumulative_sums(deltas, self.gamma * self.lam)

        # Calculate total return (i.e., sum of discounted rewards) as target for value function update
        self.return_buffer[n, :T+1] = discounted_cumulative_sums(self.reward_buffer[n, :T+1], self.gamma)

    # Store newly observed transitions in buffer, for each actor and time step
    def record(self, state, action, reward, value, logprobability, n, t):
        self.state_buffer[n, t, :] = state
        self.action_buffer[n, t] = action
        self.reward_buffer[n, t] = reward
        self.value_buffer[n, t] = value
        self.logprobability_buffer[n, t] = logprobability

    # Store newly observed transitions in buffer
    def prepare_training_data(self):
        reward_buffer = np.reshape(self.reward_buffer, self.max_time_steps * self.actors)
        states_buffer = np.reshape(self.state_buffer, (self.max_time_steps * self.actors, self.no_of_features))
        actions_buffer = np.reshape(self.action_buffer, self.max_time_steps * self.actors)
        logprobas_buffer = np.reshape(self.logprobability_buffer, self.max_time_steps * self.actors)
        advantages_buffer = np.reshape(self.advantages, self.max_time_steps * self.actors)
        return_buffer = np.reshape(self.return_buffer, self.max_time_steps * self.actors)
        return reward_buffer, states_buffer, actions_buffer, logprobas_buffer, advantages_buffer, return_buffer


    # Define neural network
    def nn_model(self, state_size, output_size):
        observation_input = tf.keras.Input(shape=(state_size,), dtype=tf.float32)
        dense1 = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh)(observation_input) # tanh achieved 204 at 397
        dense2 = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh)(dense1) # tanh achieved 204 at 397
        output = tf.keras.layers.Dense(units=output_size, activation=None)(dense2)
        if output_size == 1:
            output = tf.squeeze(output, axis=1)
        return tf.keras.Model(inputs=observation_input, outputs=output)

    # Policy network's weight update
    @tf.function
    def train_policy(self, states, actions, logprobas, advantages):

        # Setup of policy loss function
        with tf.GradientTape() as tape:
            ratio = tf.exp(logprobabilities(self.actor(states), actions) - logprobas)  # Calculate pi_new/pi_old
            min_advantage = tf.where(advantages > 0, (1 + self.clip_ratio) * advantages,
                                     (1 - self.clip_ratio) * advantages, )  # Apply clipping
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, min_advantage))  # Setup loss function as mean of individual L_clip and negative sign, as tf minimizes by default

        # Use gradient based optimizer to optimize loss function and update weights
        policy_grads = tape.gradient(policy_loss, self.actor.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.actor.trainable_variables))

    # Update value network's weights
    @tf.function
    def train_value_function(self, states, returns):
        # Setup value network's loss function
        with tf.GradientTape() as tape:
            value_loss = tf.reduce_mean(
                (returns - self.critic(states)) ** 2)  # Train the value function by regression on mean-squared error

        # Use gradient based optimizer to optimize loss function and update weights
        value_grads = tape.gradient(value_loss, self.critic.trainable_variables)
        self.value_optimizer.apply_gradients(zip(value_grads, self.critic.trainable_variables))


if __name__ == "__main__":

    environment = env
    agent = PPO_Agent()

    algo = PPO_Algo(agent, environment)




    rew, total_episodes = algo.trainPPO()

    # Run your agent
    agent.actors = 1
    rew, total_episodes = algo.runPPO()

