import numpy as np
import torch
import torch.nn as nn
from torch.nn import ReLU
from torch.autograd import Variable
import torch.nn.functional as F
import utils
from models import BootstrapModel, Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, branches=None):
        super(Actor, self).__init__()

        self.branches = branches
        if branches:
            self.model = BootstrapModel(branches, [state_dim, 400, 400, 300, action_dim])
        else:
            self.model = Model([state_dim, 400, 400, 300, action_dim])
        self.max_action = max_action

    def forward(self, x, mask=None):
        if self.branches:
            if mask is None:
                mask = torch.ones(x.shape[0], self.branches)
            x = self.model(x, mask)
        else:
            x = self.model(x)
        x = self.max_action * torch.tanh(x)
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, branches):
        super(Critic, self).__init__()
        self.branches = branches
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = BootstrapModel(branches,
                                    [state_dim + action_dim, 400, 300, 300, 1])

    def forward(self, x, u, mask=None):
        if mask is None:
            mask = torch.ones(x.shape[0], self.branches)
        xu = torch.cat([x, u], 1)
        xs = self.model(xu, mask)

        return xs

    def bootstrap_actions(self, x, us, mask=None):
        u = us.unfold(1, self.action_dim, self.action_dim)
        x = x.unsqueeze(1).expand(x.shape[0], u.shape[1], -1)
        xu = torch.cat([x, u], dim=2)
        xu = xu.view(us.shape[0], -1)
        return self.model.multiforward(xu, mask)


class TD3(object):
    def __init__(self, state_dim, action_dim, max_action, actor_branches, value_branches):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor_branches = actor_branches
        self.value_branches = value_branches

        self.actor = Actor(state_dim, action_dim, max_action, actor_branches).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action, value_branches).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim, branches).to(device)
        self.critic_target = Critic(state_dim, action_dim, branches).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action

    def select_action(self, state, mask=None):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        if mask is not None:
            mask = torch.LongTensor(mask.reshape(1, -1)).to(device)
        action = self.actor(state, mask)
        if mask is None:
            action = action.unfold(1, self.action_dim, self.action_dim)
            action = action.mean(1)
        else:
            action = action.sum(1)
        return action.cpu().data.numpy().flatten()

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
            mt = torch.LongTensor(mt).to(device)

            # Select action according to policy and add clipped noise
            tu = self.actor_target(next_state)
            if self.actor_branches != self.critic_branches:
                tu = tu.unfold(1, self.action_dim, self.action_dim)
                tu = tu.mean(1)

            noise = torch.empty_like(tu).normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (tu + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_qs = self.critic_target.bootstrap_actions(next_state, next_action)
            target_Q, _ = target_qs.min(dim=1)
            target_Q = target_Q.view(-1, 1).expand(-1, self.branches)
            target_Q = reward + (done * discount * target_Q).detach()
            target_Q = target_Q.unfold(1, 1, 1)
            target_Q = target_Q * torch.tensor(
                mt.view(batch_size, -1, 1).expand_as(target_Q),
                dtype=torch.float32,
                device=device)
            target_Q = target_Q.view(batch_size, -1)

            # Get current Q estimates
            current_qs = self.critic(state, action, mt)

            # Test to check if the unused Q functions are being 0'd out while the used ones are preserved!!
            #print((torch.tensor((target_Q != current_qs), dtype=torch.int64) - mt).nonzero())
            # Compute critic loss
            critic_loss = F.mse_loss(current_qs, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:
                # Compute actor loss
                actions = self.actor(state)
                actor_loss = -self.critic.bootstrap_actions(state, actions).mean(dim=0)
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
