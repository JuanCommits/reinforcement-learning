import torch
import torch.nn as nn
import numpy as np
import math
import wandb
import matplotlib
import matplotlib.pyplot as plt
from replay_memory import ReplayMemory, Transition
from abc import ABC, abstractmethod
from tqdm import tqdm
#from torch.utils.tensorboard import SummaryWriter
from utils import show_video, wrap_env
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


class Agent(ABC):
    def __init__(self, gym_env, obs_processing_func, memory_buffer_size, batch_size, learning_rate, gamma,
                 epsilon_i, epsilon_f, epsilon_decay, episode_block, device='cpu', epsilon_anneal_time=None):
        # Funcion phi para procesar los estados.
        self.state_processing_function = obs_processing_func

        # Asignarle memoria al agente 
        self.memory = ReplayMemory(memory_buffer_size)
        self.device = device

        self.env = gym_env

        # Hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.random_states = 60

        self.epsilon_i = epsilon_i
        self.epsilon_f = epsilon_f
        self.epsilon_anneal = epsilon_anneal_time
        self.epsilon_decay = epsilon_decay
        self.episode_block = episode_block

        self.total_steps = 0

    def get_random_states(self):
        states = []
        for _ in range(self.random_states):
            states.append(self.env.observation_space.sample())
        return torch.tensor(states).to(torch.float32)
    
    def train(self, number_episodes = 50000, max_steps=1000000, use_wandb=False):
      self.epsilon = self.epsilon_i
      rewards = []
      total_steps = 0
      states = self.get_random_states()
      #writer = SummaryWriter(comment="-" + writer_name)

      for ep in tqdm(range(number_episodes), unit='episode'):
        if total_steps > max_steps:
            break
        
        obs, _ = self.env.reset()
        state = self.state_processing_function(obs, self.device)
        current_episode_reward = 0.0

        for s in range(max_steps):
            
            # Seleccionar accion usando una política epsilon-greedy.
            action = self.select_action(state, total_steps)
            # Ejecutar la accion, observar resultado y procesarlo como indica el algoritmo.
            obs, reward, terminated, truncated, _ = self.env.step(action.item())
            done = terminated or truncated

            next_state = self.state_processing_function(obs, self.device)
            current_episode_reward += reward
            total_steps += 1
            
            reward = torch.tensor([reward], device=self.device)

            # Guardar la transicion en la memoria
            self.memory.push(state, action, reward, torch.tensor([done], dtype=torch.float16, device=self.device), next_state)
            # Actualizar el estado
            state = next_state

            # Actualizar el modelo
            self.update_weights(total_steps)

            if done:
                #self.plot_durations(rewards)
                break

        rewards.append(current_episode_reward)
        mean_reward = np.mean(rewards[-self.episode_block:])
        mean_values = torch.mean(self.get_values(states)).item()

        # Report on the traning rewards every EPISODE BLOCK episodes
        if ep % self.episode_block == 0:
            if use_wandb:
                wandb.log({"Mean Reward": mean_reward, "Epsilon": self.compute_epsilon(total_steps), "Total Steps": total_steps, "Episode": ep, "Mean Value": mean_values})
            else:
                print(f"Episode {ep} - Avg. Reward over the last {self.episode_block} episodes {mean_reward} epsilon {self.epsilon} total steps {total_steps} mean value {mean_values}")

      #torch.save(self.policy_net.state_dict(), "GenericDQNAgent.dat")
      #writer.close()

      return rewards
    
    def save_model(self, path):
        pass
    
    def play(self, env=None):
        if env is None:
            env = self.env
        observation, _ = self.env.reset()
        while True:
            env.render()

            action = self.select_action(self.state_processing_function(observation, self.device), train=False)
            observation, reward, done, truncated, _ = env.step(action.item())

            if done or truncated:
                break

    def make_video(self, name, show=False):
        env = wrap_env(self.env, name=name)
        env.start_video_recorder()

        self.play(env)

        env.close_video_recorder()
        env.close()
        if show: show_video()

    
    def plot_durations(self, episode_durations, show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        plt.legend()
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())
    
    def compute_epsilon(self, steps):
        if self.epsilon_decay is not None:
            return self.epsilon_f + (self.epsilon_i - self.epsilon_f) * math.exp(-1. * steps / self.epsilon_decay)
        elif self.epsilon_anneal is not None:
            return max(self.epsilon_f, self.epsilon_i - (self.epsilon_i - self.epsilon_f) * min(1.0, steps / self.epsilon_anneal))
        return self.epsilon_f

    @abstractmethod
    def select_action(self, state, current_steps=0, train=True):
        pass

    @abstractmethod
    def update_weights(self):
        pass