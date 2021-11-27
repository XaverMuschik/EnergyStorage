import gym
import gym_energy_storage
# import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical  #, normalize

import os
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

import time


class Agent:
    """Agent class for the cross-entropy learning algorithm."""

    def __init__(self, env):
        """Set up the environment, the neural network and member variables.

        Parameters
        ----------
        env : gym.Environment
            The game environment
        """
        self.env = env
        self.observations = self.env.observation_space
        self.actions = len(self.env.action_space)
        self.model = self.get_model()
        self.epsilon = 0.3
        self.epsilon_decay = 0.99

    def normalize(self, state):
        def scale(min_arg, max_arg, arg):
            return (arg - min_arg) / (max_arg - min_arg)

        normalized = state.copy()
        normalized[0, 0] = scale(1.0, 31.0, state[0, 0])
        normalized[0, 1] = scale(1.0, 12.0, state[0, 1])
        normalized[0, 2] = scale(2015.0, 2030.0, state[0, 2])
        normalized[0, 3] = scale(-200.0, 200.0, state[0, 3])
        normalized[0, 5] = scale(-200.0, 200.0, state[0, 5])
        return normalized

    def get_model(self):
        """Returns a keras NN model."""
        model = Sequential()
        model.add(Dense(units=32, input_dim=self.observations))
        model.add(Activation("sigmoid"))  # relu sigmoid
        model.add(Dense(16, activation="sigmoid"))  # sigmoid relu
        model.add(Dense(units=self.actions))  # Output: Action [L, R]
        model.add(Activation("softmax"))
        model.summary()
        model.compile(
            optimizer=Adam(learning_rate=0.01),  # lr=0.001
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def get_action(self, state: np.ndarray):
        """Based on the state, get an action."""
        if self.epsilon > np.random.uniform(low=0.0, high=1.0):
            action = np.random.choice([0, 1, 2])  # randomly sample from action space
            return action
        else:
            state = self.normalize(np.asarray(state).reshape(1, -1))  # [4,]< => [1, 4]
            action = self.model(state).numpy()[0]
            # print(action)
            action = np.random.choice(env.action_space, p=action)  # choice([0, 1], [0.5044534  0.49554658])
            return action

    def get_samples(self, num_episodes: int):
        """Sample games."""
        rewards = [0.0 for i in range(num_episodes)]
        episodes = [[] for i in range(num_episodes)]

        for episode in range(num_episodes):
            # start = time.time()

            state = self.env.reset()
            total_reward = 0.0

            while True:
                action = self.get_action(state)
                new_state, reward, done, action = self.env.step(action)  # overwrite selected action with actually executed action
                total_reward += reward
                episodes[episode].append((state, action))
                state = new_state
                if done:
                    rewards[episode] = total_reward
                    break
            # end = time.time()
            # print(f"time elapsed: {end - start}")

        return rewards, episodes

    def filter_episodes(self, rewards, episodes, percentile):
        """Helper function for the training."""
        reward_bound = np.percentile(rewards, percentile)
        x_train, y_train = [], []
        for reward, episode in zip(rewards, episodes):
            if reward >= reward_bound:
                observation = [step[0] for step in episode]
                action = [step[1] for step in episode]
                def _categorize_actions(x):
                    """ uses same order as env.action_space has """
                    if x == 'up':
                        return 0
                    elif x == 'down':
                        return 1
                    else:
                        return 2
                action = map(_categorize_actions, action)
                x_train.extend(observation)
                y_train.extend(action)
        x_train = np.asarray(x_train)
        y_train = to_categorical(y_train, num_classes=self.actions)  # L = 0 => [1, 0]
        return x_train, y_train, reward_bound

    def train(self, percentile, num_iterations, num_episodes):
        """Play games and train the NN."""
        for iteration in range(num_iterations):
            rewards, episodes = self.get_samples(num_episodes)
            # print("filter episodes")
            x_train, y_train, reward_bound = self.filter_episodes(rewards, episodes, percentile)
            x_train = self.normalize(x_train)
            # print("fitting model")
            self.model.fit(x=x_train, y=y_train, verbose=0)
            # print("model fitted")
            reward_mean = np.mean(rewards)
            print(f"Iteration: {iteration+1} of number iterations {num_iterations}, Reward mean: {reward_mean}, reward bound: {reward_bound}")
            self.epsilon *= self.epsilon_decay
            # if reward_mean > 500:
            #     break

    def play(self, num_episodes: int, render: bool = False):
        """Test the trained agent."""
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0
            while True:
                if render:
                    self.env.render()
                action = self.get_action(state)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                if done:
                    print(f"Total reward: {total_reward} in epsiode {episode + 1}")
                    break


if __name__ == "__main__":
    env = gym.make('energy_storage-v0')
    agent = Agent(env)
    # print(agent.observations)
    # print(agent.actions)

    agent.train(percentile=70.0, num_iterations=200, num_episodes=60)
    agent.play(num_episodes=10)

    # import cProfile
    # cProfile.run("agent.get_samples(1)")
