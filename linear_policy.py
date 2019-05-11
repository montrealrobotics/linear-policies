import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from utils import *

eps = np.finfo(np.float32).eps.item()


class LinearPolicy:
    def __init__(self, state_dim=8, action_dim=2, policy_type='gaussian'):
        if policy_type = 'gaussian':
            network = GaussianPolicy
        else:
            network = CategoricalPolicy

        self.policy_type = policy_type
        self.policy = network(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)

    def select_action(self, state, deterministic=False, save_log_probs=True):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)

        if self.policy_type is not 'gaussian':
            probs = Categorical(probs)

        if deterministic:
            action = probs.mode()
        else:
            action = probs.sample()
       
        if save_log_probs:
            self.policy.saved_log_probs.append(probs.log_prob(action))

        return action.cpu().data.numpy()[0]

    def log(self, reward):
        self.policy.rewards.append(reward)

    def load_from_file(self, file):
        self.policy.load_state_dict(torch.load(file))

    def load_from_policy(self, original):
        with torch.no_grad():
            for param, target_param in zip(self.policy.parameters(), original.policy.parameters()):
                param.data.copy_(target_param.data)
                param.requires_grad = False
    
    def perturb(self, alpha, weight_noise, bias_noise):
        with torch.no_grad():
            weight_noise = torch.from_numpy(alpha * weight_noise).float()
            bias_noise = torch.from_numpy(alpha * bias_noise).float()
            for param, noise in zip(self.policy.parameters(), [weight_noise, bias_noise]):
                param.add_(noise)
                param.requires_grad = False

    def finish_episode(self, gamma):
        R = 0
        policy_loss = []
        rewards = []
        for r in self.policy.rewards[::-1]:
            R = r + gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
        for log_prob, reward in zip(self.policy.saved_log_probs, rewards):
            policy_loss.append(-log_prob * reward)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.policy.rewards[:]
        del self.policy.saved_log_probs[:]