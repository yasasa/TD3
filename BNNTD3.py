import numpy as np
import torch
import torch.nn as nn
from torch.nn import ReLU
from torch.autograd import Variable
import torch.nn.functional as F
import utils
from models import BNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.model = BNN([state_dim, 400, 400, 300, action_dim])
        self.max_action = max_action

    def forward(self, x):
        x = self.model(x)
        x = self.max_action * torch.tanh(x)
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = BNN([state_dim + action_dim, 400, 300, 300, 1])

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)
        xs = self.model(xu)

        return xs

class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor.eval()
        self.actor_target.train()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.eval()
        self.critic_target.train()
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action

    def select_action(self, state, samples=100):
        state = torch.FloatTensor(state.reshape(1, -1)).expand(100, -1).to(device)
        self.actor.train()
        action = self.actor(state)
        self.actor.eval()
        mu = action.mean(dim=0)
        if samples > 0:
            var = action.std(dim=0)
        return mu.detach().cpu().numpy()

    def train(self,
              replay_buffer,
              iterations,
              batch_size=100,
              discount=0.99,
              tau=0.005,
              policy_noise=0.2,
              noise_clip=0.5,
              policy_freq=2):

        for it in range(iterations):

            # Sample replay buffer
            x, y, u, r, d, mt = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Select action according to policy and add clipped noise
            noise = torch.FloatTensor(u).data.normal_(0,
                                                      policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (
                self.actor_target(next_state) + noise).clamp(
                    -self.max_action, self.max_action)
            next_action = next_action.clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_q1 = self.critic_target(next_state, next_action)
            target_q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_q1, target_q2)
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimates
            current_q = self.critic(state, action)

            # Test to check if the unused Q functions are being 0'd out while the used ones are preserved!!
            #print((torch.tensor((target_Q != current_qs), dtype=torch.int64) - mt).nonzero())
            # Compute critic loss
            critic_loss = F.mse_loss(current_q, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:
                # Compute actor loss
                actions = self.actor(state)
                actor_loss = -self.critic(state, actions).mean(dim=0)
                actor_loss = actor_loss.sum()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(
                        self.critic.parameters(),
                        self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data +
                                            (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(),
                                               self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data +
                                            (1 - tau) * target_param.data)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(),
                   '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(),
                   '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(
            torch.load('%s/%s_critic.pth' % (directory, filename)))
