import wandb
import yaml


config_file_name = 'Galaxian_1_AC.yaml'
with open(f"./sweep_configs/{config_file_name}", 'r') as stream:
    config = yaml.safe_load(stream)

wandb.sweep(project='Reinforcement Learning', sweep=config, entity='jpds_mm')