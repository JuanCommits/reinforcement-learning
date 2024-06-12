import torch
import random
import wandb
import time

import torch.nn as nn
import torch.nn.functional as F

from abstract_agent import Agent
from replay_memory import Transition

class DoubleDQNAgent(Agent):
    def __init__(self, env, model_a, model_b, obs_processing_func, memory_buffer_size, batch_size,
                  learning_rate, gamma, epsilon_i, epsilon_f, epsilon_decay, episode_block, device='cpu'):
        super().__init__(env, obs_processing_func, memory_buffer_size, batch_size,
                          learning_rate, gamma, epsilon_i, epsilon_f, epsilon_decay, episode_block, device)
        self.policy = model_a.to(self.device)
        self.policy2 = model_b.to(device)

    def select_action(self, state, current_steps=0, train=True):
        self.epsilon = self.compute_epsilon(current_steps)
        if train and random.random() < self.epsilon:
                return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
              pred1 = self.policy(state)
              pred2 = self.policy2(state)
              return (pred1 + pred2).max(1).indices.view(1, 1)
            
    def get_optimizer(self, parameters):
        return torch.optim.Adam(parameters, lr=self.learning_rate)
    
    def update_weights(self, total_steps):
        if len(self.memory) > self.batch_size:
            transitions = self.memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))

            states = torch.cat(batch.state)
            actions = torch.cat(batch.action)
            rewards = torch.cat(batch.reward)
            dones = torch.cat(batch.done)
            next_states = torch.cat(batch.next_state)
            
            use_model_a = random.random() > 0.5
            if use_model_a:
                q_values = self.policy(states).gather(1, actions)
                optimizer = self.get_optimizer(self.policy.parameters())
                preds = self.policy2(next_states)
            else:
                q_values = self.policy2(states).gather(1, actions)
                optimizer = self.get_optimizer(self.policy2.parameters())
                preds = self.policy(next_states)
            
            next_values = rewards + (self.gamma * preds.max(1)[0].detach() * (1-dones))
            
            loss = F.mse_loss(q_values, next_values.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            
            if use_model_a:
                nn.utils.clip_grad_value_(self.policy.parameters(), 100)
            else:
                nn.utils.clip_grad_value_(self.policy2.parameters(), 100)
            optimizer.step()

    def save_model(self, on_wandb=False):
        timestamp = time.time()
        model_a_path = f'model_a{timestamp}.pth'
        model_b_path = f'model_b{timestamp}.pth'
        torch.save(self.policy.state_dict(), model_a_path)
        torch.save(self.policy2.state_dict(), model_b_path)
        if on_wandb:
            wandb.save(model_a_path)
            wandb.save(model_b_path)