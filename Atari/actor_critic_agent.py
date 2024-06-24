import torch
import random
import wandb
import time

import torch.nn as nn
import torch.nn.functional as F

from tqdm.notebook import tqdm
from abstract_agent import Agent
from replay_memory import Transition

class ActorCriticAgent(Agent):
    def __init__(self, env, model, obs_processing_func, memory_buffer_size, batch_size,
                  learning_rate, gamma, epsilon_i, epsilon_f,
                    epsilon_decay, episode_block, critic_model, actor_lr, device='cpu', gamma_i=0.99):
        super().__init__(env, obs_processing_func, memory_buffer_size, batch_size,
                          learning_rate, gamma, epsilon_i, epsilon_f,
                            epsilon_decay, episode_block, device)
        self.actor_net = model.to(self.device)
        self.critic_net = critic_model.to(self.device)
        self.gamma_I = gamma_i
        self.actor_lr = actor_lr
        self.actor_optimizer, self.critic_optimizer = self.get_optimizer()
            
    def select_action(self, state, current_steps=0, train=True):
        try:
          return torch.multinomial(self.actor_net(state), 1)
        except:
          print(state)
          print(self.actor_net(state))
      
    def get_optimizer(self):
        return torch.optim.Adam(self.actor_net.parameters(), lr=self.actor_lr), \
                  torch.optim.Adam(self.critic_net.parameters(), lr=self.learning_rate)
    
    def get_values(self, states):
        return [0]

    def update_weights(self, total_steps):
      I = 1
      if len(self.memory) > self.batch_size:
            transitions = self.memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))

            states = torch.cat(batch.state).to(self.device)
            actions = torch.cat(batch.action).to(self.device)
            rewards = torch.cat(batch.reward).unsqueeze(1).to(self.device)
            dones = torch.cat(batch.done).unsqueeze(1).to(self.device)
            next_states = torch.cat(batch.next_state).to(self.device)
            
            next_state_value = self.critic_net(next_states)* (1 - dones)
            delta = rewards + self.gamma * next_state_value - self.critic_net(states)

            action_probs = self.actor_net(states)

            # Actualizar el modelo del Critic
            critic_target = rewards + self.gamma * next_state_value.detach()
            critic_loss = F.mse_loss(self.critic_net(states), critic_target)
            critic_loss = critic_loss * I
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_value_(self.critic_net.parameters(), 1)
            self.critic_optimizer.step()

            # Actualizar el modelo del Actor
            actor_loss = -torch.log(action_probs.gather(1, actions)) * delta.detach()
            actor_loss = actor_loss.mean() * I
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_value_(self.actor_net.parameters(), 1)
            self.actor_optimizer.step()
            
            # Actualizar el estado
            I = I * self.gamma
            
    def save_model(self, on_wandb=False):
        timestamp = time.time()
        actor_path = f'actor-{timestamp}.pth'
        critic_path = f'critic-{timestamp}.pth'
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        if on_wandb:
            wandb.save(actor_path)
            wandb.save(critic_path)

            
            