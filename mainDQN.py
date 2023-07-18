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
from environment_CNN import Environment

data_dir = './data'  # TODO: specify relative path to data directory (e.g., './data', not './data/variant_0')
variant = 0  # TODO: specify problem variant (0 for base variant, 1 for first extension, 2 for second extension)
env = Environment(variant, data_dir)


# TODO: execute training
class Agent:
    def __init__(self):
        self.input_shape = (5, 5, 5)

        self.num_actions = 5
        self.gamma = 0.95

        self.epsilon = 0.8
        self.epsilon_decay = 0.99
        self.episode_steps = 200
        self.replay_buffer_size = 2000
        self.batch_size = 500
        self.target_update_interval = 700

        self.q_network = self.create_cnn(self.input_shape, self.num_actions)
        self.agent_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

        self.target = self.create_cnn(self.input_shape, self.num_actions)
        self.target.set_weights(self.q_network.get_weights())

    def create_cnn(self, input_shape, num_actions):
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
        outputs = tf.keras.layers.Dense(num_actions, activation=None)(fc_layer_2)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    @tf.function
    def choose_action(self, state):
        q_values = tf.squeeze(self.q_network(state))
        action = tf.argmax(q_values)
        return action

    def sum_function(self, input_var, discount):
        n = len(input_var)
        result = np.zeros_like(input_var)

        for i in range(n):
            for j in range(i, n):
                result[i] += input_var[j] * discount ** (j - i)

        return result

    def sample_minibatch(self, replay_buffer):
        samples = random.sample(replay_buffer, self.batch_size)
        states = np.zeros(shape=(self.batch_size, 5, 5, 5), dtype=np.float32)
        actions = np.zeros(shape=(self.batch_size, ), dtype=np.int32)
        rewards = np.zeros(shape=(self.batch_size, ), dtype=np.float32)
        next_states = np.zeros(shape=(self.batch_size, 5, 5, 5), dtype=np.float32)
        dones = np.zeros(shape=(self.batch_size, ), dtype=np.float32)
        for i, sample in enumerate(samples):
            states[i] = sample[0]
            actions[i] = sample[1]
            rewards[i] = sample[2]
            next_states[i] = sample[3]
            dones[i] = sample[4]
        return states, actions, rewards, next_states, dones

    @tf.function
    def update_actor(self, states, actions, rewards, next_states, dones):
        target_maxs = tf.reduce_max(self.target(next_states), axis=1)
        target_value = rewards + (1 - dones) * self.gamma * target_maxs
        with tf.GradientTape() as tape:
            one_hot_actions = tf.one_hot(actions, 5)
            q_values = self.q_network(states)
            q_a_values = tf.reduce_sum(one_hot_actions * q_values, axis=1)
            loss = tf.reduce_mean(0.5 * (target_value - q_a_values)**2)
        gradient = tape.gradient(loss, self.q_network.trainable_variables)
        self.agent_optimizer.apply_gradients(zip(gradient, self.q_network.trainable_variables))
        return loss



class Algorithm:
    def __init__(self, agent):
        self.agent = agent
        self.env = env

    def train(self):
        self.run(mode='training')

    def run(self, mode='validation'):
        training = mode == 'training'

        if training:
            no_of_episodes = 800
        else:
            no_of_episodes = 100

        reward_history = deque(maxlen=no_of_episodes)
        replay_buffer = deque(maxlen=self.agent.replay_buffer_size)
        step_counter = 0
        for episode in range(no_of_episodes):

            action_counter = [0, 0, 0, 0, 0]    # just for our console output

            episode_reward = 0
            state = self.env.reset(mode)

            for t in range(self.agent.episode_steps):
                state = tf.reshape(state, shape=(1, self.agent.input_shape[0], self.agent.input_shape[1], self.agent.input_shape[2]))
                action = self.agent.choose_action(state)
                if training and np.random.random() < self.agent.epsilon:
                    action = random.randint(0, 4)

                action_counter[action] += 1
                reward, next_state, done = self.env.step(action)

                if training:
                    replay_buffer.append((state, action, reward, next_state, done))

                    buffer_length = len(replay_buffer)
                    if buffer_length >= self.agent.batch_size:
                        states, actions, rewards, next_states, dones = self.agent.sample_minibatch(replay_buffer)
                        self.agent.update_actor(states, actions, rewards, next_states, dones)

                    if np.mod(step_counter, self.agent.target_update_interval) == 0:
                        self.agent.target.set_weights(self.agent.q_network.get_weights())


                episode_reward += reward
                state = next_state
                step_counter += 1
                if training:
                    self.agent.epsilon *= self.agent.epsilon_decay
                if done:
                    break




            print(f'episode: {episode} | episode reward: {episode_reward} | actions: {action_counter}')

            reward_history.append(episode_reward)



if __name__ == '__main__':
    ag = Agent()
    algo = Algorithm(ag)
    algo.train()