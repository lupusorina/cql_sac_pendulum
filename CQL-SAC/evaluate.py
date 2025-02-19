import gymnasium as gym
import numpy as np
import torch
import argparse
from utils import evaluate
import random
from agent import CQLSAC


def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="CQL-SAC", help="Run name, default: CQL-SAC")
    parser.add_argument("--env", type=str, default="Pendulum-v1", help="Gym environment name, default: Pendulum-v0")
    parser.add_argument("--episodes", type=int, default=300, help="Number of episodes, default: 200")
    parser.add_argument("--buffer_size", type=int, default=100_000, help="Maximal training dataset size, default: 100_000")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--log_video", type=int, default=0, help="Log agent behaviour to wanbd when set to 1, default: 0")
    parser.add_argument("--save_every", type=int, default=20, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size, default: 256")
    parser.add_argument("--hidden_size", type=int, default=256, help="")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="")
    parser.add_argument("--temperature", type=float, default=1.0, help="")
    parser.add_argument("--cql_weight", type=float, default=1.0, help="")
    parser.add_argument("--target_action_gap", type=float, default=10, help="")
    parser.add_argument("--with_lagrange", type=int, default=0, help="")
    parser.add_argument("--tau", type=float, default=5e-3, help="")
    parser.add_argument("--eval_every", type=int, default=1, help="")

    args = parser.parse_args()
    return args

config = get_config()

np.random.seed(config.seed)
random.seed(config.seed)
torch.manual_seed(config.seed)
env = gym.make(config.env, render_mode="human")
eval_env = gym.make(config.env, render_mode="human")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

agent = CQLSAC(state_size=env.observation_space.shape[0],
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