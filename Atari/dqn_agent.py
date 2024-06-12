import torch
import random
import time
import copy
import wandb

import torch.nn as nn
import torch.nn.functional as F

from tqdm.notebook import tqdm
from abstract_agent import Agent
from replay_memory import Transition

class DQNAgent(Agent):
    def __init__(self, env, model, obs_processing_func, memory_buffer_size, batch_size,
                  learning_rate, gamma, epsilon_i, epsilon_f,
                    epsilon_decay, episode_block, device='cpu', second_model_update=None, epsilon_anneal_time=None):
        super().__init__(env, obs_processing_func, memory_buffer_size, batch_size,
                          learning_rate, gamma, epsilon_i, epsilon_f,
                            epsilon_decay, episode_block, device, epsilon_anneal_time)
        self.policy = model.to(self.device)
        self.second_model_update = second_model_update
        self.target_policy = None
        if self.second_model_update is not None:
            self.target_policy = copy.deepcopy(self.policy)
            
    def select_action(self, state, current_steps=0, train=True):
      self.epsilon = self.compute_epsilon(current_steps)
      if train and random.random() < self.epsilon:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)
      else:
            with torch.no_grad():
              return self.policy(state).max(1).indices.view(1, 1)
      
    def get_optimizer(self):
        return torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)

    def update_weights(self, total_steps):
      optimizer = self.get_optimizer()
      if len(self.memory) > self.batch_size:
            transitions = self.memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))

            states = torch.cat(batch.state)
            actions = torch.cat(batch.action)
            rewards = torch.cat(batch.reward)
            dones = torch.cat(batch.done)
            next_states = torch.cat(batch.next_state)

            q_values = self.policy(states).gather(1, actions)

            preds = self.policy(next_states) if self.target_policy is None \
                                      else self.target_policy(next_states)
            next_values = rewards + (self.gamma * preds.max(1)[0].detach() * (1-dones))

            loss = F.mse_loss(q_values, next_values.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_value_(self.policy.parameters(), 100)
            optimizer.step()

            if self.second_model_update is not None and total_steps % self.second_model_update == 0:
                self.target_policy.load_state_dict(self.policy.state_dict())

    def save_model(self, on_wandb=False):
        timestamp = time.time()
        path = f"model_{timestamp}.pt"
        torch.save(self.policy.state_dict(), path)
        if on_wandb:
            wandb.save(path)
            