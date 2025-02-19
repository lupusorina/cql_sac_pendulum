import gymnasium as gym
import numpy as np
import torch
import argparse
from utils import evaluate
import random
from agent import CQLSAC
from train_offline import get_config

config = get_config()

np.random.seed(config.seed)
random.seed(config.seed)
torch.manual_seed(config.seed)
env = gym.make(config.env, render_mode="human")
eval_env = gym.make(config.env, render_mode="human")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

agent = CQLSAC(state_size=env.observation_space.shape[0] - 1,
                action_size=env.action_space.shape[0],
                tau=config.tau,
                hidden_size=config.hidden_size,
                learning_rate=config.learning_rate,
                temp=config.temperature,
                with_lagrange=config.with_lagrange,
                cql_weight=config.cql_weight,
                target_action_gap=config.target_action_gap,
                device=device)

FOLDER_MODELS = 'trained_models'
MODEL_NAME = 'CQL-SAC-OFFLINE_Pendulum0'

state_dict = torch.load(f'{FOLDER_MODELS}/{MODEL_NAME}.pth', map_location=device)
agent.actor_local.load_state_dict(state_dict)

eval_reward = evaluate(eval_env, agent)
print("Test Reward", eval_reward)