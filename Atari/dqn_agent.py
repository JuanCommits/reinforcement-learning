import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from replay_memory import ReplayMemory, Transition
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
import numpy as np
import random
from abstract_agent import Agent

class DQNAgent(Agent):
    def __init__(self, gym_env, model, obs_processing_func, memory_buffer_size, batch_size, 
                 learning_rate, gamma, epsilon_i, epsilon_f, episode_block, 
                 epsilon_anneal_time=None, epsilon_decay=None, optimizer=None, 
                 loss_function=nn.MSELoss(), device='cpu'):
        super().__init__(gym_env, obs_processing_func, memory_buffer_size, batch_size,
                         learning_rate, gamma, epsilon_i, epsilon_f, epsilon_anneal_time, 
                         epsilon_decay, episode_block, device)
        self.policy = model.to(self.device)
        self.criterion = loss_function
        self.optim = Adam(model.parameters(), learning_rate) if optimizer is None else optimizer
    
    def select_action(self, state, current_steps, train=True):
      if train and (random.random() < self.compute_epsilon(current_steps)):
          return torch.tensor([[self.env.action_space.sample()]], device=self.device)
      with torch.no_grad():
        return torch.argmax(self.policy(state)).view(1,1)

    def update_weights(self):
      if len(self.memory) > self.batch_size:
            # Resetear gradientes
            self.optim.zero_grad()

            # Obtener un minibatch de la memoria. Resultando en tensores de estados, acciones, recompensas, flags de terminacion y siguentes estados. 
            batch = Transition(*zip(*self.memory.sample(self.batch_size)))

            states = torch.cat(batch.state).to(self.device)
            actions = torch.cat(batch.action).to(self.device)
            next_states = torch.cat(batch.next_state).to(self.device)
            rewards = torch.cat(batch.reward).to(self.device)
            dones = torch.cat(batch.done).to(self.device).unsqueeze(1)

            # Obetener el valor estado-accion (Q) de acuerdo a la policy net para todo elemento (estados) del minibatch.
            preds = self.policy(states).gather(1, actions) * (1 - dones)

            next_state_values = None
            with torch.no_grad():
              next_state_values = ((self.policy(next_states).detach().max(1).values * self.gamma) + rewards)

            loss = self.criterion(preds, next_state_values.unsqueeze(1))
            loss.backward()
            nn.utils.clip_grad_value_(self.policy.parameters(), 100)
            self.optim.step()