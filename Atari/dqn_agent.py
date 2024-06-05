import torch
import torch.nn as nn
import torch.nn.functional as F
from replay_memory import ReplayMemory, Transition
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
import numpy as np
from abstract_agent import Agent
import random
from dqn_model import DQN_Model
import copy

class DQNAgent(Agent):
    def __init__(self, env, model, obs_processing_func, memory_buffer_size, batch_size,
                  learning_rate, gamma, epsilon_i, epsilon_f, epsilon_anneal_time,
                    epsilon_decay, episode_block, device='cpu', second_model_update=None):
        super().__init__(env, obs_processing_func, memory_buffer_size, batch_size,
                          learning_rate, gamma, epsilon_i, epsilon_f, epsilon_anneal_time,
                            epsilon_decay, episode_block, device)
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
              if self.target_policy is not None:
                  return self.target_policy(state).max(1).indices.view(1, 1)
              return self.policy(state).max(1).indices.view(1, 1)
      
    def get_optimizer(self):
        return torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)

    def update_weights(self, total_steps):
      optimizer = self.get_optimizer()
      if len(self.memory) > self.batch_size:
            # Obtener un minibatch de la memoria. Resultando en tensores de estados, acciones, recompensas, flags de terminacion y siguentes estados.
            transitions = self.memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))

            # Enviar los tensores al dispositivo correspondiente.
            states = torch.cat(batch.state)
            actions = torch.cat(batch.action)
            rewards = torch.cat(batch.reward)
            dones = torch.cat(batch.done)
            next_states = torch.cat(batch.next_state)

            q_values = self.policy(states).gather(1, actions)
            
            next_states_values = self.policy(next_states).max(1).values.detach() if self.target_policy is None \
                                      else self.target_policy(next_states).max(1).values
            next_values = (self.gamma * next_states_values * (1 - dones)) + rewards

            # Compute el costo y actualice los pesos.
            criterion = nn.SmoothL1Loss()
            loss = criterion(q_values, next_values.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            
            nn.utils.clip_grad_value_(self.policy.parameters(), 100)
            optimizer.step()
            # En Pytorch la funcion de costo se llaman con (predicciones, objetivos) en ese orden.

            if self.second_model_update is not None and total_steps % self.second_model_update == 0:
                self.target_policy.load_state_dict(self.policy.state_dict())
            