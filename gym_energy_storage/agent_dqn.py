import collections
import os
import random
from typing import Deque

import gym
import numpy as np

from build_network import DQN  # TODO: check if this works
import gym
import gym_energy_storage
from gym_energy_storage.plot_learning_outcome import plot_learning_result

MODEL_PATH = os.path.join("../saved_model", "dqn_model.h5")


class Agent:
    def __init__(self, env: gym.Env):
        # DQN Env Variables
        self.env = env
        self.observations = self.env.observation_space
        self.actions = len(self.env.action_space)
        # DQN Agent Variables
        self.replay_buffer_size = 10_000  # ToDo: tune hyperparameter
        self.train_start = 1_000  # ToDo: tune hyperparameter
        self.memory: Deque = collections.deque(maxlen=self.replay_buffer_size)
        self.gamma = 1  # 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999
        # DQN Network Variables
        self.state_shape = self.observations
        self.learning_rate = 1e-3
        self.dqn = DQN(self.state_shape, self.actions, self.learning_rate)
        self.target_dqn = DQN(self.state_shape, self.actions, self.learning_rate)
        self.target_dqn.update_model(self.dqn)
        self.batch_size = 128

    def get_action(self, state: np.ndarray):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.actions)  # ToDo: check if this works or if set needs to be provided
        else:
            return np.argmax(self.dqn(self.normalize(state)))

    def train(self, num_episodes: int):
        last_rewards: Deque = collections.deque(maxlen=5)
        best_reward_mean = 0.0
        for episode in range(1, num_episodes + 1):
            total_reward = 0.0
            state = self.env.reset()
            state = np.reshape(state, newshape=(1, -1)).astype(np.float32)
            while True:
                action = self.get_action(state)
                next_state, reward, done, action = self.env.step(action)  # ToDo: negativer reward, wenn action out of bounds (in env aendern)
                next_state = np.reshape(next_state, newshape=(1, -1)).astype(np.float32)
                self.remember(state, action, reward, next_state, done)
                self.replay()
                total_reward += reward
                state = next_state
                if done:
                    self.target_dqn.update_model(self.dqn)  # target dqn weights are updated
                    print(f"Episode: {episode} Reward: {total_reward} Epsilon: {self.epsilon}")
                    last_rewards.append(total_reward)
                    current_reward_mean = np.mean(last_rewards)
                    if current_reward_mean > best_reward_mean:
                        best_reward_mean = current_reward_mean
                        self.dqn.save_model(MODEL_PATH)
                        print(f"New best mean: {best_reward_mean}")
                    break

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def normalize(self, state):
        """ normalizes observations to interval [0,1] """
        def scale(min_arg, max_arg, arg):
            return (arg - min_arg) / (max_arg - min_arg)

        normalized = state.copy()
        normalized[0, 0] = scale(0.0, env.len_period, state[0, 0])  # normalize time step
        normalized[0, 1] = scale(-200.0, 200.0, state[0, 1])  # normlize price
        normalized[0, 2] = scale(0.0, env.max_stor_lev, state[0, 1])  # normlize storage level
        normalized[0, 3] = scale(-200.0, 200.0, state[0, 3])  # normalize storage value
        normalized[0, 4] = scale(-100.0, 100.0, state[0, 4])  # normalize mean price
        return normalized

    def replay(self):
        if len(self.memory) < self.train_start:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, states_next, dones = zip(*minibatch)

        states = np.concatenate(states).astype(np.float32)
        states_next = np.concatenate(states_next).astype(np.float32)

        q_values = self.dqn(self.normalize(states))
        q_values_next = self.target_dqn(self.normalize(states_next))

        for i in range(self.batch_size):
            a = actions[i]
            done = dones[i]
            if done:
                q_values[i][a] = rewards[i]
            else:
                q_values[i][a] = rewards[i] + self.gamma * np.max(q_values_next[i])

        self.dqn.fit(self.normalize(states), q_values)
        self.plot()

    def plot(self):
        """ plot result of training"""
        plot = plot_learning_result(stor_val=40, mean_price=40, model=self.dqn, max_stor=env.max_stor_lev, len_period=env.len_period)
        plot.main()

    def play(self, num_episodes: int, render: bool = True):
        self.dqn.load_model(MODEL_PATH)
        for episode in range(1, num_episodes + 1):
            total_reward = 0.0
            state = self.env.reset()
            state = np.reshape(state, newshape=(1, -1)).astype(np.float32)
            while True:
                if render:
                    self.env.render()
                action = self.get_action(state)
                next_state, reward, done, action = self.env.step(action)
                next_state = np.reshape(next_state, newshape=(1, -1)).astype(np.float32)
                total_reward += reward
                state = next_state
                if done:
                    print(f"Episode: {episode} Reward: {total_reward}")
                    break


if __name__ == "__main__":
    env = gym.make('energy_storage-v0')
    agent = Agent(env)
    agent.train(num_episodes=200)
    # input("Play?")
    # agent.play(num_episodes=30, render=True)
