import torch
import torch.nn as nn
import numpy as np
from replay_memory import ReplayMemory, Transition
from abc import ABC, abstractmethod
from tqdm import tqdm
#from torch.utils.tensorboard import SummaryWriter
from utils import show_video

class Agent(ABC):
    def __init__(self, gym_env, obs_processing_func, memory_buffer_size, batch_size, learning_rate, gamma,
                 epsilon_i, epsilon_f, epsilon_anneal_time, epsilon_decay, episode_block, device='cpu'):
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

        self.epsilon_i = epsilon_i
        self.epsilon_f = epsilon_f
        self.epsilon_anneal = epsilon_anneal_time
        self.epsilon_decay = epsilon_decay
        self.episode_block = episode_block

        self.total_steps = 0
    
    def train(self, number_episodes = 50000, max_steps_episode = 10000, max_steps=1000000, writer_name="default_writer_name"):
      self.epsilon = self.epsilon_i
      rewards = []
      total_steps = 0
      #writer = SummaryWriter(comment="-" + writer_name)

      for ep in tqdm(range(number_episodes), unit=' episodes'):
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

            print(f"States {state} -")
            print(f"Actions {action.item()} -")
            print(f"Rewards {reward} -")
            print(f"Dones {done} -")
            print(f"Next states {obs} -")


            # Guardar la transicion en la memoria
            self.memory.push(state, action, reward, torch.tensor([done], dtype=torch.float16), next_state)
            # Actualizar el estado
            state = next_state

            # Actualizar el modelo
            self.update_weights(total_steps)

            if done: 
                break

        rewards.append(current_episode_reward)
        mean_reward = np.mean(rewards[-100:])
        #writer.add_scalar("epsilon", self.epsilon, total_steps)
        #writer.add_scalar("reward_100", mean_reward, total_steps)
        #writer.add_scalar("reward", current_episode_reward, total_steps)

        # Report on the traning rewards every EPISODE BLOCK episodes
        if ep % self.episode_block == 0:
          print(f"Episode {ep} - Avg. Reward over the last {self.episode_block} episodes {np.mean(rewards[-self.episode_block:])} epsilon {self.epsilon} total steps {total_steps}")

      print(f"Episode {ep + 1} - Avg. Reward over the last {self.episode_block} episodes {np.mean(rewards[-self.episode_block:])} epsilon {self.epsilon} total steps {total_steps}")

      #torch.save(self.policy_net.state_dict(), "GenericDQNAgent.dat")
      #writer.close()

      return rewards
    
    def compute_epsilon(self, steps):
        if self.epsilon_decay is not None:
            return self.epsilon_f + (self.epsilon_i - self.epsilon_f) * np.exp(-1. * steps * (1-self.epsilon_decay))
        elif self.epsilon_anneal is not None:
            return max(self.epsilon_f, self.epsilon_i - (self.epsilon_i - self.epsilon_f) * min(1.0, steps / self.epsilon_anneal))
        return self.epsilon_f
    
    def record_test_episode(self, env):
        done = False

        obs, _ = env.reset()
        state = self.state_processing_function(obs, self.device)
        
        env.start_video_recorder()
        while not done:
            env.render()

            action = self.select_action(state, train=False)
            obs, reward, terminated, truncated, _ = self.env.step(action.item())
            done = terminated or truncated

            if done:
                break      

            state = self.state_processing_function(obs, self.device)  
        env.close_video_recorder()
        env.close()
        show_video()

    @abstractmethod
    def select_action(self, state, current_steps=0, train=True):
        pass

    @abstractmethod
    def update_weights(self):
        pass