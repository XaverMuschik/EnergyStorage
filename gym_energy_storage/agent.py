import gym
import gym_energy_storage
import numpy as np
from torch.distributions import Categorical
import torch
import torch.nn as nn
import torch.optim as optim
import os
import tensorflow as tf
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")


class Pi(nn.Module):

    """ agent class for reinforce algorithm """

    def __init__(self, in_dim, out_dim):
        """Set up the environment, the neural network and member variables.

        """
        super(Pi, self).__init__()
        layers = [
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        ]

        self.model = nn.Sequential(*layers)
        self.onpolicy_reset()
        self.train()

    def onpolicy_reset(self):
        self.log_probs = []
        self.rewards = []

    def forward(self, x):
        pdparams = self.model(x)
        return pdparams

    def act(self, state):

        '''Epsilon-greedy policy: with probability epsilon, do random action, otherwise do default sampling.'''
        epsilon = 0.2
        if epsilon > np.random.uniform(low=0.0, high=1.0):
            action = np.random.choice([0, 1, 2])  # randomly sample from action space
            log_prob = np.log(torch.tensor(1/3))
            self.log_probs.append(torch.tensor(log_prob))
            return action
        else:
            x = torch.from_numpy(state.astype(np.float32))
            pdparam = self.forward(x)
            pd = Categorical(logits=pdparam)
            action = pd.sample()
            log_prob = pd.log_prob(action)
            self.log_probs.append(log_prob)
            return action.item()

def train(pi, optimizer, gamma):
    # inner gradient-acent loop of REINFORCE algorithm
    T = len(pi.rewards)
    rets = np.empty(T, dtype=np.float)
    future_ret = 0.0
    # compute the returns efficiently
    for t in reversed(range(T)):
        future_ret = pi.rewards[t] + gamma * future_ret
        rets[t] = future_ret
    rets = torch.tensor(rets)
    log_probs = torch.stack(pi.log_probs)
    loss = - log_probs * rets  # gradient term; Negative for maximization
    loss = torch.sum(loss)
    optimizer.zero_grad()
    loss.backward()  # backpropagate, compute gradients
    optimizer.step()  # gradient-ascent, update the weights
    return loss

def main():
        env = gym.make('energy_storage-v0')
        in_dim = env.observation_space  # shape of observations
        out_dim = len(env.action_space)  # shape of action space
        pi = Pi(in_dim, out_dim)
        optimizer = optim.Adam(pi.parameters(), lr=0.01)
        action_vector = []
        for epi in range(300):
            state = env.reset()
            for t in range(len(env.time_index)):
                action = pi.act(state)
                action_vector.append(action)
                state, reward, done, _ = env.step(action)  # ToDo: check out which actions are (why) taken
                pi.rewards.append(reward)
                if done:
                    break
            number_actions = {i: action_vector.count(i) for i in action_vector}
            loss = train(pi, optimizer, gamma=0.99)  # train per episode
            total_reward = sum(pi.rewards)
            pi.onpolicy_reset()
            print(f'Episode {epi}, loss: {loss}, \
            total_reward: {total_reward}, \
            actions taken: {number_actions}')
            action_vector = []


if __name__ == "__main__":
    # env = gym.make('energy_storage-v0')
    main()
