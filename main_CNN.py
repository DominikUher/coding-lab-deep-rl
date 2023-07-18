''''
IDEA:
use a downscaled CNN as additional critic for "attention" effect.
Here, we don't need the exact grids but rather a value for a good state 
(which is e.g., high for being at the right edge)'''
import numpy as np
import tensorflow as tf
from collections import deque
from environment_CNN import Environment


data_dir = './data'  # TODO: specify relative path to data directory (e.g., './data', not './data/variant_0')
variant = 0  # TODO: specify problem variant (0 for base variant, 1 for first extension, 2 for second extension)


class PPO:
    def __init__(self, seed, env, variant, input_shape, lr_actor, lr_critic_1, lr_critic_2, return_lambda, gamma, clip_epsilon, episode_steps,
                 no_of_actors, actor_updates_per_episode, critic_updates_per_episode, clip_annealing_factor):
        np.random.seed(seed)
        self.env = env
        self.agent = Agent()

    @classmethod
    def from_dict(cls, d):
        ppo = cls(d['seed'], d['environment'], d['variant'], d['lr_actor'], d['lr_critic_1'], d['lr_critic_2'],
                  d['return_lambda'], d['gamma'], d['clip_epsilon'], d['episode_steps'], d['no_of_actors'],
                  d['actor_updates_per_episode'], d['critic_updates_per_episode'], d['clip_annealing_factor'])
        return ppo

    def train(self):
        self.run(mode='training')

    def run(self, mode='validation'):
        training = mode == 'training'

        if training:
            no_of_episodes = 800
        else:
            no_of_episodes = 100

        reward_history = deque(maxlen=no_of_episodes)

        for episode in range(no_of_episodes):

            episode_reward_history = np.zeros(shape=self.agent.no_of_actors, dtype=np.float32)
            action_counter = [0, 0, 0, 0, 0]    # just for our console output

            if training:  # before each episode, we initialize the trajectories
                state_buffer = np.zeros(
                    shape=(self.agent.no_of_actors, self.agent.episode_steps, self.agent.input_shape[0],
                           self.agent.input_shape[1], self.agent.input_shape[2]), dtype=np.float32)
                action_buffer = np.zeros(shape=(self.agent.no_of_actors, self.agent.episode_steps, 1), dtype=np.int32)
                reward_buffer = np.zeros(shape=(self.agent.no_of_actors, self.agent.episode_steps, 1), dtype=np.float32)
                next_state_buffer = np.zeros(
                    shape=(self.agent.no_of_actors, self.agent.episode_steps, self.agent.input_shape[0],
                           self.agent.input_shape[1], self.agent.input_shape[2]), dtype=np.float32)
                done_buffer = np.zeros(shape=(self.agent.no_of_actors, self.agent.episode_steps, 1), dtype=np.float32)
                action_log_probs_buffer = np.zeros(
                    shape=(self.agent.no_of_actors, self.agent.episode_steps, 1), dtype=np.float32)
                value_buffer = np.zeros(shape=(self.agent.no_of_actors, self.agent.episode_steps, 1), dtype=np.float32)

            for n in range(self.agent.no_of_actors):
                episode_reward = 0
                state = self.env.reset(mode)

                for t in range(self.agent.episode_steps):
                    state = tf.reshape(state, shape=(1, self.agent.input_shape[0], self.agent.input_shape[1], self.agent.input_shape[2]))
                    action, action_log_prob = self.agent.choose_action(state)
                    action_counter[action] += 1
                    reward, next_state, done = self.env.step(action)

                    if training:
                        v_value = self.agent.evaluate_state(state)
                        state_buffer[n, t] = state
                        action_buffer[n, t] = action
                        reward_buffer[n, t] = reward
                        next_state_buffer[n, t] = next_state
                        done_buffer[n, t] = done
                        action_log_probs_buffer[n, t] = action_log_prob
                        value_buffer[n, t] = v_value

                    episode_reward += reward
                    state = next_state

                    if done:
                        episode_reward_history[n] = episode_reward
                        break

            if training:

                return_buffer, advantage_buffer = self.agent.calculate_actor_update_prerequisites(
                    self.agent.no_of_actors, reward_buffer, value_buffer)
                states, actions, rewards, action_log_probs, returns, advantages = self.agent.get_trajectory_data(
                    state_buffer, action_buffer, reward_buffer, action_log_probs_buffer, return_buffer, advantage_buffer)

                actor_loss = np.zeros(shape=self.agent.actor_updates_per_episode)
                for i in range(self.agent.actor_updates_per_episode):
                    actor_loss[i] = self.agent.update_actor(states, actions, action_log_probs, advantages)

                for i in range(self.agent.critic_updates_per_episode):
                    self.agent.update_critics(states, returns)

            mean_actor_loss = np.mean(actor_loss)
            mean_reward = np.mean(episode_reward_history)
            actions_count = [np.round(frequency / self.agent.no_of_actors, 2) for frequency in action_counter]

            print(
                f'episode: {episode} | episode reward: {np.round(mean_reward, 2)} '
                f'| actor loss: {np.round(mean_actor_loss, 4)}  actions: {actions_count}')

            reward_history.append(episode_reward_history)


class Agent:
    def __init__(self, seed, env, variant, input_shape, lr_actor, lr_critic_1, lr_critic_2, return_lambda, gamma, clip_epsilon, episode_steps,
                 no_of_actors, actor_updates_per_episode, critic_updates_per_episode, clip_annealing_factor):
        self.input_shape = (5, 5, 5)
        self.num_actions = 5
        self.no_of_actors = 80
        self.gamma = 0.95 # 0.97
        self.return_lambda = 0.8 # 0.8
        self.clip_epsilon = 0.25
        self.episode_steps = 200
        self.actor_updates_per_episode = 100 # 80
        self.critic_updates_per_episode = 100 # 80

        self.agent = self.create_cnn(self.input_shape, self.num_actions, 'agent')
        self.agent_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0005)

        self.value_1 = self.create_cnn(self.input_shape, self.num_actions, 'value')
        self.value_1_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0005)

        self.value_2 = self.create_cnn(self.input_shape, self.num_actions, 'value')
        self.value_2_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0007)


    def create_cnn(self, input_shape, num_actions, network_type):
        """
        CNN idea:
        - we basically have a channel-wise FC layer
        - then, we "merge" the channels pixel-wise
        - then, we simply use two FC layers
        - output logits for actor, one value for critics
        """
        inputs = tf.keras.layers.Input(shape=input_shape)
        conv1 = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(inputs)
        conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(conv1)
        flattened_layer = tf.keras.layers.Flatten()(conv2)
        fc_layer_1 = tf.keras.layers.Dense(units=64, activation='relu')(flattened_layer)
        fc_layer_2 = tf.keras.layers.Dense(units=64, activation='relu')(fc_layer_1)
        if network_type == 'agent':
            outputs = tf.keras.layers.Dense(num_actions, activation=None)(fc_layer_2)
        if network_type == 'value':
            outputs = tf.keras.layers.Dense(1, activation=None)(fc_layer_2)
        return tf.keras.models.Model(inputs=inputs, outputs=outputs)

    @tf.function
    def choose_action(self, state):
        logits = self.agent(state)
        action = tf.random.categorical(logits, 1)
        index = tf.one_hot(action, 5)[0]
        action_log_prob = tf.math.reduce_sum(index * logits)
        return tf.squeeze(action), action_log_prob

    def sum_function(self, input_var, discount):
        n = len(input_var)
        result = np.zeros_like(input_var)

        for i in range(n):
            for j in range(i, n):
                result[i] += input_var[j] * discount ** (j - i)

        return result

    def get_trajectory_data(self, state_buffer, action_buffer, reward_buffer,
                            action_probs_buffer, return_buffer, advantage_buffer):
        states = np.reshape(state_buffer, newshape=(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        actions = np.reshape(action_buffer, newshape=(-1))
        rewards = np.reshape(reward_buffer, newshape=(-1))
        action_probs = np.reshape(action_probs_buffer, newshape=(-1))
        returns = np.reshape(return_buffer, newshape=(-1))
        advantages = np.reshape(advantage_buffer, newshape=(-1))

        return states, actions, rewards, action_probs, returns, advantages

    def calculate_actor_update_prerequisites(self, no_of_actors, reward_buffer, value_buffer):
        return_buffer = np.zeros(shape=(self.no_of_actors, self.episode_steps, 1), dtype=np.float32)
        advantage_buffer = np.zeros(shape=(self.no_of_actors, self.episode_steps, 1), dtype=np.float32)

        # the advantage calculation is based on the infinite sum from the TD-Lambda exercise sheet
        lambda_discount = self.gamma * self.return_lambda
        for n in range(no_of_actors):
            return_buffer[n] = self.sum_function(reward_buffer[n], self.gamma)

            td_errors = reward_buffer[n, :-1] + self.gamma * value_buffer[n, 1:] - value_buffer[n, :-1]
            advantage_buffer[n, :-1] = self.sum_function(td_errors, lambda_discount)

        return return_buffer, advantage_buffer

    @tf.function
    def update_actor(self, states, actions, action_log_probs, advantages):
        with tf.GradientTape() as tape:
            current_dist = self.agent(states)
            action_indices = tf.one_hot(actions, 5)
            current_log_probs = tf.reduce_sum(action_indices * current_dist, axis=1)

            ratio = tf.exp(current_log_probs - action_log_probs)
            clipped_ratio = tf.where(condition=advantages < 0, x=tf.maximum(ratio, 1 - self.clip_epsilon),
                                     y=tf.minimum(ratio, 1 + self.clip_epsilon))
            loss = -tf.reduce_mean(clipped_ratio * advantages)

        gradient = tape.gradient(loss, self.agent.trainable_variables)
        clipped_gradient = []
        for s in gradient:
            clipped_gradient.append(tf.clip_by_norm(s, clip_norm=2.0))
        self.agent_optimizer.apply_gradients(zip(clipped_gradient, self.agent.trainable_variables))

        return loss

    @tf.function
    def update_critics(self, state_buffer, return_buffer):
        with tf.GradientTape() as tape:
            td_error = 0.5 * (return_buffer - self.value_1(state_buffer)) ** 2
            value_1_loss = tf.reduce_mean(td_error)
        gradient_1 = tape.gradient(value_1_loss, self.value_1.trainable_variables)
        self.value_1_optimizer.apply_gradients(zip(gradient_1, self.value_1.trainable_variables))

        with tf.GradientTape() as tape:
            td_error = 0.5 * (return_buffer - self.value_2(state_buffer)) ** 2
            value_2_loss = tf.reduce_mean(td_error)
        gradient_2 = tape.gradient(value_2_loss, self.value_2.trainable_variables)
        self.value_2_optimizer.apply_gradients(zip(gradient_2, self.value_2.trainable_variables))

        return value_1_loss, value_2_loss

    @tf.function
    def evaluate_state(self, state):
        v_1 = self.value_1(state)
        v_2 = self.value_2(state)
        mean_value = tf.reduce_mean([v_1, v_2])
        return mean_value
