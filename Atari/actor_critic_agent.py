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
                    epsilon_decay, episode_block, critic_model, device='cpu'):
        super().__init__(env, obs_processing_func, memory_buffer_size, batch_size,
                          learning_rate, gamma, epsilon_i, epsilon_f,
                            epsilon_decay, episode_block, device)
        self.actor = model.to(self.device)
        self.critic = critic_model.to(self.device)
            
    def select_action(self, state, current_steps=0, train=True):
        try:
          return torch.multinomial(self.actor(state), 1)
        except:
          print(self.actor(state))
          
      
    def get_optimizer(self):
        return torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate), \
                  torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)

    def update_weights(self, total_steps):
      actor_optimizer, critic_optimizer = self.get_optimizer()
      if len(self.memory) > self.batch_size:
            # Obtener un minibatch de la memoria. Resultando en tensores de estados, acciones, recompensas, flags de terminacion y siguentes estados.
            transitions = self.memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))

            # Enviar los tensores al dispositivo correspondiente.
            states = torch.cat(batch.state).to(self.device)
            actions = torch.cat(batch.action).to(self.device)
            rewards = torch.cat(batch.reward).to(self.device)
            dones = torch.cat(batch.done).to(self.device)
            next_states = torch.cat(batch.next_state).to(self.device)
            
            next_state_val = self.critic(next_states) * (1 - dones)
            state_val = self.critic(states)

            # Update actor weights
            actor_optimizer.zero_grad()
            delta = rewards + self.gamma * next_state_val - state_val
            loss = (-torch.log(self.actor(states).gather(1, actions)) * delta.detach()).mean()
            loss.backward()
            nn.utils.clip_grad_value_(self.actor.parameters(), 100)
            actor_optimizer.step()

            # Update critic weights
            critic_optimizer.zero_grad()
            critic_loss = F.mse_loss(rewards + self.gamma * next_state_val.detach(), state_val)
            critic_loss.backward()
            critic_optimizer.step()
            # En Pytorch la funcion de costo se llaman con (predicciones, objetivos) en ese orden.

    def save_model(self, on_wandb=False):
        timestamp = time.time()
        actor_path = f'actor-{timestamp}.pth'
        critic_path = f'critic-{timestamp}.pth'
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        if on_wandb:
            wandb.save(actor_path)
            wandb.save(critic_path)

            
            