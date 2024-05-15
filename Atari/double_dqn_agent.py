import torch
import torch.nn as nn
import torch.nn.functional as F
from replay_memory import ReplayMemory, Transition
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
import numpy as np
from abstract_agent import Agent

class DoubleDQNAgent(Agent):
    def __init__(self, gym_env, model_a, model_b, obs_processing_func, memory_buffer_size, batch_size, learning_rate, gamma,
                 epsilon_i, epsilon_f, epsilon_anneal_time, epsilon_decay, episode_block, sync_target =  100):
        
        super().__init__(gym_env, obs_processing_func, memory_buffer_size, batch_size, learning_rate, gamma,
                 epsilon_i, epsilon_f, epsilon_anneal_time, epsilon_decay, episode_block)              
        
        # Asignar los modelos al agente (y enviarlos al dispositivo adecuado)
        # self.q_a = ?
        # self.q_b = ?

        # Asignar una función de costo (MSE)  (y enviarla al dispositivo adecuado)
        # self.loss_function = ?

        # Asignar un optimizador para cada modelo (Adam)
        # self.optimizer_A = ?
        # self.optimizer_B = ?



    def select_action(self, state, current_steps, train=True):
        # Implementar. Seleccionando acciones epsilongreedy-mente (sobre Q_a + Q_b)
        # si estamos entranando y completamente greedy en otro caso.
        pass
    
    def update_weights(self):
        if len(self.memory) > self.batch_size:
            pass
            # Obtener un minibatch de la memoria. Resultando en tensores de estados, acciones, recompensas, flags de terminacion y siguentes estados. 

            # Enviar los tensores al dispositivo correspondiente.
            #states = ?
            #actions = ?
            #rewards = ?
            #dones = ?  # Dones deberia ser 0 y 1; no True y False. Pueden usar .float() en un tensor para convertirlo
            #next_states = ?
            
            # Actualizar al azar Q_a o Q_b usando el otro para calcular el valor de los siguientes estados.
            
            # Para el Q elegido:
                # Obetener el valor estado-accion (Q) de acuerdo al Q seleccionado.
                #q_actual = ?

                # Obtener max a' Q para los siguientes estados (del minibatch) (Usando el Q no seleccionado).
                # Es importante hacer .detach() al resultado de este computo.
                # Si el estado siguiente es terminal (done) este valor debería ser 0.
                #max_q_next_state = ?

                # Compute el target de DQN de acuerdo a la Ecuacion (3) del paper.    
                #target = ?

                # Resetear gradientes

                # Compute el costo y actualice los pesos.
                # En Pytorch la funcion de costo se llaman con (predicciones, objetivos) en ese orden.