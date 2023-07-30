from collections import deque
import os
import tensorflow as tf
import numpy as np
import random


class PPO:
    def __init__(self, seed, env, variant, input_shape, hidden_size, early_stopping, lr_actor, lr_critic_1, lr_critic_2, return_lambda, gamma, clip_epsilon, episode_steps,
                 no_of_actors, actor_updates_per_episode, critic_updates_per_episode, clip_annealing_factor, hyperparameters):
        self.agent = PPOAgent(variant, input_shape, hidden_size, lr_actor, lr_critic_1, lr_critic_2, return_lambda, gamma, clip_epsilon, episode_steps, no_of_actors)
        self.hyperparameters = hyperparameters
        self.env = env
        self.variant = variant
        self.lr = lr_actor
        self.lbd = return_lambda
        self.gamma = gamma
        self.obs = hyperparameters['observation']
        self.epsilon = clip_epsilon
        self.actor_updates_per_episode = actor_updates_per_episode
        self.critic_updates_per_episode = critic_updates_per_episode
        self.clip_annealing_factor = clip_annealing_factor
        self.early_stopping = early_stopping
        self.validation_after_episodes = hyperparameters['validation_after_episodes']
        self.best_score = -np.inf
        self.best_validation_score = -np.inf
        os.environ['PYTHONHASHSEED'] = str(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    @classmethod
    def from_dict(cls, d):
        ppo = cls(d['seed'], d['environment'], d['variant'], d['input_shape'], d['hidden_size'], d['early_stopping'], d['lr_actor'], d['lr_critic_1'],
                  d['lr_critic_2'], d['return_lambda'], d['gamma'], d['clip_epsilon'], d['episode_steps'], d['no_of_actors'],
                  d['actor_updates_per_episode'], d['critic_updates_per_episode'], d['clip_annealing_factor'], d)
        return ppo
    
    def get_best_score(self):
        return self.best_score

    def train(self):
        self.run_ppo(mode='training')

    def run_ppo(self, mode='validation'):
        training = mode == 'training'
        episodes_without_improvement = 0

        if training:
            no_of_episodes = 800
        else:
            no_of_episodes = 100

        reward_history = deque(maxlen=no_of_episodes)

        for episode in range(no_of_episodes):
            no_actors = self.agent.no_of_actors if training else 1
            if (episodes_without_improvement >= self.early_stopping) and training:
                break
            self.agent.clip_epsilon *= self.clip_annealing_factor   # adaptive clip epsilon

            episode_reward_history = np.zeros(shape=no_actors, dtype=np.float32)
            action_counter = [0, 0, 0, 0, 0]    # just for our console output
                                                # -> we might want to have a look at our average policy

            if training:        # before each episode, we initialize the trajectories
                state_buffer = np.zeros(
                    shape=(no_actors, self.agent.episode_steps, self.agent.no_of_features),
                    dtype=np.float32)
                action_buffer = np.zeros(shape=(no_actors, self.agent.episode_steps, 1), dtype=np.int32)
                reward_buffer = np.zeros(shape=(no_actors, self.agent.episode_steps, 1), dtype=np.float32)
                next_state_buffer = np.zeros(shape=(no_actors, self.agent.episode_steps, self.agent.no_of_features),
                                             dtype=np.float32)
                done_buffer = np.zeros(shape=(no_actors, self.agent.episode_steps, 1), dtype=np.float32)
                action_probs_buffer = np.zeros(
                    shape=(no_actors, self.agent.episode_steps, 1), dtype=np.float32)
                value_buffer = np.zeros(shape=(no_actors, self.agent.episode_steps, 1), dtype=np.float32)

            episode_history = []
            action_history = []

            for n in range(no_actors):
                episode_reward = 0
                state = self.env.reset(mode)
                state = tf.reshape(state, shape=(1, self.agent.no_of_features))

                for t in range(self.agent.episode_steps):
                    action_prob, action = self.agent.choose_action(state)
                    action_counter[action] += 1
                    if not training:
                        action_history.append(action.numpy())
                    reward, next_state, done = self.env.step(action)
                    next_state = tf.reshape(next_state, shape=(1, self.agent.no_of_features))

                    if training:
                        v_value = self.agent.evaluate_state(state)
                        state_buffer[n, t] = state
                        action_buffer[n, t] = action
                        reward_buffer[n, t] = reward
                        next_state_buffer[n, t] = next_state
                        done_buffer[n, t] = done
                        action_probs_buffer[n, t] = action_prob
                        value_buffer[n, t] = v_value

                    episode_reward += reward
                    state = next_state

                    if done:
                        episode_reward_history[n] = episode_reward
                        break

                if not training:
                    episode_history.append(self.env.episode)
                    
            if training:

                return_buffer, advantage_buffer = self.agent.calculate_actor_update_prerequisites(no_actors, reward_buffer, value_buffer)
                states, actions, rewards, action_probs, returns, advantages = self.agent.get_trajectory_data(state_buffer, action_buffer, reward_buffer, action_probs_buffer, return_buffer, advantage_buffer)

                mean_validation_reward_str = 'None'
                if episode % self.validation_after_episodes == 0:
                    mean_validation_reward = self.run_ppo()
                    mean_validation_reward_str = '%.2f' % mean_validation_reward

                actor_loss = np.zeros(shape=self.actor_updates_per_episode)
                for i in range(self.actor_updates_per_episode):
                    actor_loss[i] = self.agent.update_actor(states, actions, action_probs, advantages)

                for i in range(self.critic_updates_per_episode):
                    self.agent.update_critics(states, returns)

                mean_actor_loss = np.mean(actor_loss)
                actor_loss_str = '%.2f' % mean_actor_loss

            mean_reward = np.mean(episode_reward_history)
            actions_count = [np.round(frequency / no_actors, 2) for frequency in action_counter]
            mean_reward_str = '%.2f' % mean_reward
            reward_history.append(mean_reward)

            if training:
                if mean_validation_reward > self.best_score:
                    self.best_score = mean_validation_reward
                    best_validation_score_str = '%.2f' % self.best_score
                    episodes_without_improvement = 0
                else:
                    episodes_without_improvement += 1
                
                print(
                    f'episode: {episode} | episode reward: {mean_reward_str} | validation reward: {mean_validation_reward_str}'
                    f' | actor loss: {actor_loss_str} | actions: {actions_count}')
                
            elif mean_reward > self.best_validation_score:
                    self.best_validation_score = mean_reward
                    best_validation_score_str = '%.2f' % self.best_validation_score
                    self.best_validation_episode = self.env.episode
                    self.best_episode_actions = action_history

        if mode == 'validation':
                return np.mean(reward_history)
    
        if mode == 'testing':
            test_score = np.mean(reward_history)
            with open('./output/.test_performance.txt', 'a') as output:
                output.write(f'Testing PPO with parameters: {self.hyperparameters}\n')
                output.write(f'Mean testing score of {test_score}\n')
                output.write(f'Best testing score of {best_validation_score_str} achieved in episode {self.best_validation_episode}\n')
                output.write(f'Actions taken during best testing episode: {self.best_episode_actions}\n\n')
            return
        
        f = open(f'./output/PPO_v{self.variant}_s{best_validation_score_str}_obs{self.obs}_lr{self.lr}_lb{self.lbd}_g{self.gamma}_e{self.epsilon}.txt', 'w')
        f.write(f'Best score of {best_validation_score_str} achieved after training {self.best_validation_episode} episodes\n')
        f.write(f'Chosen hyperparameters: {self.hyperparameters}\n\n')
        f.writelines(f'Episode number of best validation run: {self.best_validation_episode}\n\n')
        f.writelines(f'Actions taken in best validation run: {self.best_episode_actions}')
        f.close()

        self.run_ppo('testing')
        


class PPOAgent:
    def __init__(self, variant, input_shape, hidden_size, lr_actor, lr_critic_1, lr_critic_2, return_lambda, gamma, clip_epsilon, episode_steps,
                 no_of_actors):
        #self.no_of_features = [13, 19, 0][variant]  # depends on the number of features
        self.no_of_features = input_shape
        self.no_of_actions = 5
        self.hidden_size = hidden_size

        # some hyperparameters
        self.no_of_actors = no_of_actors
        self.gamma = gamma
        self.return_lambda = return_lambda
        self.clip_epsilon = clip_epsilon
        self.episode_steps = episode_steps

        # defining the actor
        self.actor = self.create_nn('actor')
        self.lr_actor = lr_actor
        self.actor_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.lr_actor)

        # defining the first critic
        self.critic_1 = self.create_nn('critic')
        self.lr_critic_1 = lr_critic_1
        self.critic_1_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.lr_critic_1)

        # defining the second critic
        self.critic_2 = self.create_nn('critic')
        self.lr_critic_2 = lr_critic_2
        self.critic_2_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.lr_critic_2)

    def create_nn(self, nn_type):
        input_layer = tf.keras.Input(shape=(self.no_of_features,), dtype=tf.float32)
        dense1 = tf.keras.layers.Dense(units=self.hidden_size, activation='tanh')(input_layer)
        dense2 = tf.keras.layers.Dense(units=self.hidden_size, activation='tanh')(dense1)
        if nn_type == 'actor':
            output = tf.keras.layers.Dense(self.no_of_actions, activation='softmax')(dense2)
        if nn_type == 'critic':
            output = tf.keras.layers.Dense(1, activation=None)(dense2)
            output = tf.squeeze(output, axis=1)

        return tf.keras.Model(inputs=input_layer, outputs=output)

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
    def update_actor(self, states, actions, action_probs, advantages):
        with tf.GradientTape() as tape:
            current_dist = self.actor(states)
            current_log_dist = tf.math.log(current_dist)
            action_indices = tf.one_hot(actions, 5)
            current_log_probs = tf.reduce_sum(action_indices * current_log_dist, axis=1)

            ratio = tf.exp(current_log_probs - tf.math.log(action_probs))
            clipped_ratio = tf.where(condition=advantages < 0, x=tf.maximum(ratio, 1 - self.clip_epsilon), y=tf.minimum(ratio, 1 + self.clip_epsilon))
            loss = -tf.reduce_mean(clipped_ratio * advantages)

        gradient = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(gradient, self.actor.trainable_variables))

        return loss

    @tf.function
    def update_critics(self, state_buffer, return_buffer):
        with tf.GradientTape() as tape:
            td_error = 0.5 * (return_buffer - self.critic_1(state_buffer)) ** 2
            critic_1_l = tf.reduce_mean(td_error)
        gradient = tape.gradient(critic_1_l, self.critic_1.trainable_variables)
        self.critic_1_optimizer.apply_gradients(zip(gradient, self.critic_1.trainable_variables))

        with tf.GradientTape() as tape:
            td_error = 0.5 * (return_buffer - self.critic_2(state_buffer)) ** 2
            critic_2_l = tf.reduce_mean(td_error)
        gradient = tape.gradient(critic_2_l, self.critic_2.trainable_variables)
        self.critic_2_optimizer.apply_gradients(zip(gradient, self.critic_2.trainable_variables))

        return critic_1_l, critic_2_l

    @tf.function
    def choose_action(self, state):
        action_dist = self.actor(state)
        action = tf.random.categorical(tf.math.log(action_dist), 1)
        action = tf.squeeze(action)
        action_prob = action_dist[0][action]
        return action_prob, action

    def sum_function(self, input_var, discount):
        n = len(input_var)
        result = np.zeros_like(input_var)

        for i in range(n):
            for j in range(i, n):
                result[i] += input_var[j] * discount ** (j - i)

        return result

    def get_trajectory_data(self, state_buffer, action_buffer, reward_buffer,
                            action_probs_buffer, return_buffer, advantage_buffer):
        states = np.reshape(state_buffer, newshape=(-1, self.no_of_features))
        actions = np.reshape(action_buffer, newshape=(-1))
        rewards = np.reshape(reward_buffer, newshape=(-1))
        action_probs = np.reshape(action_probs_buffer, newshape=(-1))
        returns = np.reshape(return_buffer, newshape=(-1))
        advantages = np.reshape(advantage_buffer, newshape=(-1))

        return states, actions, rewards, action_probs, returns, advantages

    @tf.function
    def evaluate_state(self, state_buffer):
        v_1 = self.critic_1(state_buffer)
        v_2 = self.critic_2(state_buffer)
        return tf.reduce_mean([v_1, v_2])
