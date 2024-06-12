import torch
import numpy as np
import random
import numpy as np
import gymnasium
import sys
import wandb
import configparser
import time

from dqn_model import DQN_Model
from dqn_agent import DQNAgent
from double_dqn_agent import DoubleDQNAgent
from dqn_cnn_model import DQN_CNN_Model
from utils import show_video, wrap_env, make_env
from tqdm import tqdm

# Seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

# Constants
ENVS = ["MountainCar-v0", "CartPole-v1", "Acrobot-v1"]
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
EPISODE_BLOCK = 10
VIDEO_NAME_SUFFIX = "-episode-0.mp4"
VIDEO_FOLDER = "./video/"

# Settings
ENTITY = 'jpds_mm'
PROJECT = 'Reinforcement Learning'
SWEEP_ID = '7wsxvgmr'

def process_state(obs, device):
    return torch.tensor(obs, device=device).unsqueeze(0)

def process_packed_state(obs, device):
    return torch.tensor(obs[:], device=device).unsqueeze(0)

def main(iterations):
    print(f"Running {iterations} iterations on device {DEVICE}.")
    wandb.login(key=get_api_key())
    wandb.agent(SWEEP_ID, function=make_run, count=iterations, entity=ENTITY, project=PROJECT)
    wandb.finish()
    pass

def make_run():
    wandb.init()

    if wandb.config.env == "ALE/Galaxian-v5":
        env = make_env(wandb.config.env, "rgb_array")
        process_state_function = process_packed_state
        input_dim = env.observation_space.shape
    else:
        env = gymnasium.make(wandb.config.env, render_mode='rgb_array')
        input_dim = env.observation_space.shape[0]
        process_state_function = process_state

    output_dim = env.action_space.n

    config = wandb.config

    net = get_model(config, input_dim, output_dim)

    if config.algo == "DQN":
        agent = DQNAgent(env, net, process_state_function, config.buffer_size, config.batch_size, 
                    config.lr, config.gamma, epsilon_i=config.eps_i, 
                    epsilon_f=config.eps_f, epsilon_decay=config.eps_decay, 
                    episode_block=EPISODE_BLOCK, device=DEVICE, second_model_update=config.target_update_steps)
    elif config.algo == "DDQN":
        net2 = get_model(config, input_dim, output_dim)
        agent = DoubleDQNAgent(env, net, net2, process_state_function, config.buffer_size, config.batch_size, 
                    config.lr, config.gamma, epsilon_i=config.eps_i, 
                    epsilon_f=config.eps_f, epsilon_decay=config.eps_decay, 
                    episode_block=EPISODE_BLOCK, device=DEVICE)

    agent.train(config.episodes, config.max_total_steps, use_wandb=True)
    
    video_name = f"agent-{time.time()}"
    agent.make_video(video_name, show=True)

    video_path = VIDEO_FOLDER + video_name + VIDEO_NAME_SUFFIX
    wandb.log({"Video": wandb.Video(video_path, fps=4, format="mp4")})

def get_api_key():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config.get('wandb', 'api-key')

def get_model(model_config: dict, input_dim, output_dim):
    if model_config.get('model', "FF") == "CNN":
        return DQN_CNN_Model(input_dim, output_dim).to(DEVICE)
    else:
        return DQN_Model(input_dim, output_dim).to(DEVICE)
        

if __name__ == '__main__':
    args = sys.argv
    iterations = int(args[1])
    main(iterations)