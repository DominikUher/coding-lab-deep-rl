# TODO: parse arguments
...
''''
IDEA:
use a downscaled CNN as additional critic for "attention" effect.
Here, we don't need the exact grids but rather a value for a good state 
(which is e.g., high for being at the right edge)'''

# set seed
seed = 7  # TODO: set seed to allow for reproducibility of results

import os
os.environ['PYTHONHASHSEED'] = str(seed)

import random
random.seed(seed)

import numpy as np
np.random.seed(seed)

import tensorflow as tf
#from keras.layers import LeakyReLU

tf.random.set_seed(seed)

from collections import deque

# initialize environment
from environment_CNN_moving import Environment

data_dir = './data'  # TODO: specify relative path to data directory (e.g., './data', not './data/variant_0')
variant = 0  # TODO: specify problem variant (0 for base variant, 1 for first extension, 2 for second extension)
env = Environment(variant, data_dir)


# TODO: execute training
class Agent:
    def __init__(self):
        self.input_shape = (5, 5, 5)

        self.num_actions = 5
        self.no_of_actors = 80
        self.gamma = 0.95 # 0.97
        self.return_lambda = 0.8 # 0.8
        self.clip_epsilon = 0.25
        self.episode_steps = 200
        self.actor_updates_per_episode = 20 # 80
        self.critic_updates_per_episode = 20 # 80
        self.window_size = 100
        self.update_interval = 20

        self.actor = self.create_cnn(self.input_shape, self.num_actions, 'agent')
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
       # conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(conv1)
        flattened_layer = tf.keras.layers.Flatten()(conv2)
       # flattened_layer = tf.keras.layers.Flatten()(inputs)
        fc_layer_1 = tf.keras.layers.Dense(units=64, activation='relu')(flattened_layer)
        fc_layer_2 = tf.keras.layers.Dense(units=64, activation='relu')(fc_layer_1)
        if network_type == 'agent':
            outputs = tf.keras.layers.Dense(num_actions, activation=None)(fc_layer_2)
        if network_type == 'value':
            outputs = tf.keras.layers.Dense(1, activation=None)(fc_layer_2)
        return tf.keras.models.Model(inputs=inputs, outputs=outputs)

    @tf.function
    def choose_action(self, state):
        logits = self.actor(state)
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

    def get_buffers(self, replay_buffer):
        states = np.zeros(shape=(self.no_of_actors, self.window_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]), dtype=np.float32)
        actions = np.zeros(shape=(self.no_of_actors, self.window_size, 1), dtype=np.int32)
        rewards = np.zeros(shape=(self.no_of_actors, self.window_size, 1), dtype=np.float32)
        next_states = np.zeros(shape=(self.no_of_actors, self.window_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]), dtype=np.float32)
        dones = np.zeros(shape=(self.no_of_actors, self.window_size, 1), dtype=np.float32)
        action_log_probs = np.zeros(shape=(self.no_of_actors, self.window_size, 1), dtype=np.float32)
        v_values = np.zeros(shape=(self.no_of_actors, self.window_size, 1), dtype=np.float32)
        next_values = np.zeros(shape=(self.no_of_actors, self.window_size, 1), dtype=np.float32)

        for i, step_data in enumerate(replay_buffer):
            for actor in range(self.no_of_actors):
                states[actor, i] = step_data[0][actor]
                actions[actor, i] = step_data[1][actor]
                rewards[actor, i] = step_data[2][actor]
                next_states[actor, i] = step_data[3][actor]
                dones[actor, i] = step_data[4][actor]
                action_log_probs[actor, i] = step_data[5][actor]
                v_values[actor, i] = step_data[6][actor]
                next_values[actor, i] = step_data[7][actor]

        return states, actions, rewards, next_states, dones, action_log_probs, v_values, next_values

    def get_trajectory_data(self, state_buffer, action_buffer,
                            action_probs_buffer, return_buffer, advantage_buffer):
        states = np.reshape(state_buffer, newshape=(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        actions = np.reshape(action_buffer, newshape=(-1))
        action_probs = np.reshape(action_probs_buffer, newshape=(-1))
        returns = np.reshape(return_buffer, newshape=(-1))
        advantages = np.reshape(advantage_buffer, newshape=(-1))

        return states, actions, action_probs, returns, advantages

    def calculate_actor_update_prerequisites(self, no_of_actors, reward_buffer, value_buffer, next_values):
        return_buffer = np.zeros(shape=(self.no_of_actors, self.window_size, 1), dtype=np.float32)
        advantage_buffer = np.zeros(shape=(self.no_of_actors, self.window_size, 1), dtype=np.float32)

        # the advantage calculation is based on the infinite sum from the TD-Lambda exercise sheet
        lambda_discount = self.gamma * self.return_lambda
        for n in range(no_of_actors):
            return_buffer[n] = self.sum_function(reward_buffer[n], self.gamma)

            td_errors = reward_buffer[n] + self.gamma * next_values[n] - value_buffer[n]
            advantage_buffer[n] = self.sum_function(td_errors, lambda_discount)

        return return_buffer, advantage_buffer

    @tf.function
    def update_actor(self, states, actions, action_log_probs, advantages):
        with tf.GradientTape() as tape:
            current_dist = self.actor(states)
            action_indices = tf.one_hot(actions, 5)
            current_log_probs = tf.reduce_sum(action_indices * current_dist, axis=1)

            ratio = tf.exp(current_log_probs - action_log_probs)
            clipped_ratio = tf.where(condition=advantages < 0, x=tf.maximum(ratio, 1 - self.clip_epsilon),
                                     y=tf.minimum(ratio, 1 + self.clip_epsilon))
            loss = -tf.reduce_mean(clipped_ratio * advantages)

        gradient = tape.gradient(loss, self.actor.trainable_variables)
        clipped_gradient = []
        for s in gradient:
            clipped_gradient.append(tf.clip_by_norm(s, clip_norm=2.0))
        self.agent_optimizer.apply_gradients(zip(clipped_gradient, self.actor.trainable_variables))

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


class Algorithm:
    def __init__(self, agent):
        self.agent = agent
        self.envs = list()
        for actor in range(self.agent.no_of_actors):
            self.envs.append(Environment(variant, data_dir))

    def train(self):
        self.run(mode='training')

    def run(self, mode='validation'):
        training = mode == 'training'

        if training:
            self.agent.actor.load_weights("actor_weights")
            self.agent.value_1.load_weights("value_1_weights")
            self.agent.value_2.load_weights("value_2_weights")

            no_of_episodes = 800
        else:
            no_of_episodes = 100

        reward_history = deque(maxlen=no_of_episodes)

        replay_buffer = deque(maxlen=self.agent.window_size)

        episode_reward = np.zeros(shape=(200, self.agent.no_of_actors), dtype=np.float32)

        next_state_saver = np.zeros(
            shape=(self.agent.no_of_actors, self.agent.input_shape[0],
                   self.agent.input_shape[1], self.agent.input_shape[2]), dtype=np.float32)

        for step in range(no_of_episodes * 200):
            if step > 0 and np.mod(step, 200) == 0:
                reward_history.append(np.mean(episode_reward))

            # every step for each actor is saved here
            state_buffer = np.zeros(
                shape=(self.agent.no_of_actors, self.agent.input_shape[0],
                       self.agent.input_shape[1], self.agent.input_shape[2]), dtype=np.float32)
            action_buffer = np.zeros(shape=self.agent.no_of_actors, dtype=np.int32)
            reward_buffer = np.zeros(shape=self.agent.no_of_actors, dtype=np.float32)
            next_state_buffer = np.zeros(
                shape=(self.agent.no_of_actors, self.agent.input_shape[0],
                       self.agent.input_shape[1], self.agent.input_shape[2]), dtype=np.float32)
            done_buffer = np.zeros(shape=self.agent.no_of_actors, dtype=np.float32)
            action_log_probs_buffer = np.zeros(
                shape=self.agent.no_of_actors, dtype=np.float32)
            value_buffer = np.zeros(shape=self.agent.no_of_actors, dtype=np.float32)
            next_value_buffer = np.zeros(shape=self.agent.no_of_actors, dtype=np.float32)


            for n in range(self.agent.no_of_actors):
                if np.mod(step, 200) == 0:     # necessary to access new episode data
                    state = self.envs[n].reset(mode)
                else:
                    state = next_state_saver[n]

                state = tf.reshape(state, shape=(1, self.agent.input_shape[0], self.agent.input_shape[1], self.agent.input_shape[2]))
                action, action_log_prob = self.agent.choose_action(state)

                reward, next_state, done = self.envs[n].step(action)

                if training:
                    v_value = self.agent.evaluate_state(state)
                    state_buffer[n] = state
                    action_buffer[n] = action
                    reward_buffer[n] = reward
                    next_state_buffer[n] = next_state
                    done_buffer[n] = done
                    action_log_probs_buffer[n] = action_log_prob
                    value_buffer[n] = v_value
                    next_state_reshape = tf.reshape(next_state, shape=(1, self.agent.input_shape[0],
                                                                       self.agent.input_shape[1],
                                                                       self.agent.input_shape[2]))

                    next_value_buffer[n] = self.agent.evaluate_state(next_state_reshape)

                next_state_saver[n] = next_state
                episode_reward[np.mod(step, 200), n] = reward

            if step > 0 and np.mod(step, 200) == 0:
                mean_reward = np.mean(np.sum(episode_reward, axis=0))
                rounded_reward = np.round(mean_reward, 2)
                reward_history.append(mean_reward)
                print(f'episode: {int(step / 200)} | episode reward: {rounded_reward} ')

            if not training:
                continue

            replay_buffer.append((state_buffer, action_buffer, reward_buffer, next_state_buffer, done_buffer, action_log_probs_buffer, value_buffer, next_value_buffer))

            if step >= self.agent.window_size - 1 and np.mod(step, self.agent.update_interval) == 0:       # if we have enough training samples

                states, actions, rewards, next_states, dones, action_log_probs, v_values, next_values = self.agent.get_buffers(replay_buffer)

                returns, advantages = self.agent.calculate_actor_update_prerequisites(
                    self.agent.no_of_actors, rewards, v_values, next_values)

                states, actions, action_log_probs, returns, advantages = self.agent.get_trajectory_data(states, actions, action_log_probs, returns, advantages)


                actor_loss = np.zeros(shape=self.agent.actor_updates_per_episode)
                for i in range(self.agent.actor_updates_per_episode):
                    actor_loss[i] = self.agent.update_actor(states, actions, action_log_probs, advantages)

                for i in range(self.agent.critic_updates_per_episode):
                    self.agent.update_critics(states, returns)



if __name__ == '__main__':
    ag = Agent()
    algo = Algorithm(ag)
    algo.train()
