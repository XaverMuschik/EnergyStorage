import gym
import gym_energy_storage
import numpy as np
from torch.distributions import Categorical
import torch
import torch.nn as nn
import torch.optim as optim
import os
import tensorflow as tf
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
        x = torch.from_numpy(state.astype(np.float))
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
        env = gym.make('gym_energy_storage')
        in_dim = env.observation_space  # shape of observations
        out_dim = len(env.action_space)  # shape of action space
        pi = Pi(in_dim, out_dim)
        optimizer = optim.Adam(pi.parameters(), lr=0.01)
        for epi in range(300):
            state = env.reset()
            for t in range(len(env.time_index)):
                action = pi.act(state)
                state, reward, done, _ = env.step(action)
                pi.rewards.append(reward)
                if done:
                    break
            loss = train(pi, optimizer, gamma=0.99)  # train per episode
            total_reward = sum(pi.rewards)
            pi.onpolicy_reset()
            print(f'Episode {epi}, loss: {loss}. \
            total_reward: {total_reward}')


if __name__ == "__main__":
    env = gym.make('energy_storage-v0')
    main()
