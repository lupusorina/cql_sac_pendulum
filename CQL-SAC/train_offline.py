import gymnasium as gym
import numpy as np
from collections import deque
import torch
import wandb
import argparse
import glob
from utils import save
import random
from agent import CQLSAC
from torch.utils.data import DataLoader, TensorDataset
import minari
import pandas as pd
import ast
from pendulum import PendulumEnv
from matplotlib import pyplot as plt
import os

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="CQL", help="Run name, default: CQL")
    parser.add_argument("--env", type=str, default="Pendulum-v1", help="Gym environment name, default: Pendulum-v0")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes, default: 100")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--log_video", type=int, default=0, help="Log agent behaviour to wanbd when set to 1, default: 0")
    parser.add_argument("--save_every", type=int, default=100, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size, default: 256")
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

def prep_dataloader(env_id="halfcheetah-medium-v2", batch_size=256, seed=1):
    env = PendulumEnv()

    DATASET_FOLDER = 'data'
    FILENAME = 'data_pendulum_5000.csv'
    EXTRACT_DATA = False
    df = pd.read_csv(f'{DATASET_FOLDER}/{FILENAME}')

    if EXTRACT_DATA == True:
        observation_list = []
        action_list = []
        for item in df['states']:
            observation_list.append(ast.literal_eval(item))
        for item in df['actions']:
            action_list.append(ast.literal_eval(item))

        actions_np = np.array(action_list)
        observations_np = np.array(observation_list)
        # save the numpy arrays.
        np.save(DATASET_FOLDER + '/observations.npy', observations_np)
        np.save(DATASET_FOLDER + '/actions.npy', actions_np)
    else:
        observations_np = np.load(DATASET_FOLDER + '/observations.npy')
        actions_np = np.load(DATASET_FOLDER + '/actions.npy')

    rewards = df['reward'].values
    terminations = df['done'].values

    tensors = {}
    tensors["observations"] = torch.tensor(observations_np, dtype=torch.float32)
    tensors["actions"] = torch.tensor(actions_np, dtype=torch.float32)
    tensors["rewards"] = torch.tensor(rewards, dtype=torch.float32)
    tensors["next_observations"] = torch.cat([tensors["observations"][1:], torch.tensor(observations_np[-1:])[0][None]], dim=0)
    tensors["terminals"] = torch.tensor(terminations, dtype=torch.float32)

    tensordata = TensorDataset(tensors["observations"],
                               tensors["actions"],
                               tensors["rewards"][:, None],
                               tensors["next_observations"],
                               tensors["terminals"][:, None])
    dataloader = DataLoader(tensordata, batch_size=batch_size, shuffle=True)
    
    eval_env = PendulumEnv() # render_mode='human'
    return dataloader, eval_env

def evaluate(env, policy, eval_runs=5): 
    """
    Makes an evaluation run with the current policy
    """
    print('Evaluating')
    reward_batch = []
    for i in range(eval_runs):
        state = env.reset()

        rewards = 0
        counter = 0
        EPISODE_DONE_ANGLE_THRESHOLD_DEG = 0.5
        upright_angle_buffer = []
        done = False
        while True:
            action = policy.get_action(state, eval=True)
            state, reward, _, _, _ = env.step(action)
            rewards += reward
            counter += 1
            cos_theta = state[0]
            sin_theta = state[1]
            theta = np.arctan2(sin_theta, cos_theta)
            if abs(theta) < np.deg2rad(EPISODE_DONE_ANGLE_THRESHOLD_DEG):
                upright_angle_buffer.append(theta)
            if len(upright_angle_buffer) > 40 or counter > 500:
                done = True
            if done:
                break
        reward_batch.append(rewards)
    print(reward_batch)
    print('duration:', counter)
    return np.mean(reward_batch)

def train(config):
    print('Start training')
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    dataloader, env = prep_dataloader(env_id=config.env, batch_size=config.batch_size, seed=config.seed)
    env.action_space.seed(config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    batches = 0
    average10 = deque(maxlen=10)
    with wandb.init(project="CQL-offline", name=config.run_name, config=config):
        
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

        wandb.watch(agent, log="gradients", log_freq=10)
        if config.log_video:
            # TODO: this is not working
            env = gym.wrappers.Monitor(env, './video', video_callable=lambda x: x%10==0, force=True)

        eval_reward = evaluate(env, agent)
        wandb.log({"Test Reward": eval_reward, "Episode": 0, "Batches": batches}, step=batches)
        print("Episode: {} | Reward: {}".format(0, eval_reward))
        for i in range(1, config.episodes+1):
            print('i:', i)

            for batch_idx, experience in enumerate(dataloader):
                states, actions, rewards, next_states, dones = experience
                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                next_states = next_states.to(device)
                dones = dones.to(device)

                # # Create a plot with the states and actions:
                # fig, ax = plt.subplots(4, 1, sharex=True, figsize=(8, 8))
                # angle = np.arctan2(states[:, 1].detach().numpy(), states[:, 0].detach().numpy())
                # ax[0].plot(angle, label='angle')
                # ax[0].set_ylabel('angle')
                # ax[0].legend()
                # ax[1].plot(states[:, 2].detach().numpy(), label='angular velocity')
                # ax[1].set_ylabel('angular velocity')
                # ax[1].legend()
                # ax[2].plot(actions[:, 0].detach().numpy(), label='action')
                # ax[2].set_ylabel('action')
                # ax[2].legend()
                # ax[3].plot(rewards.detach().numpy(), label='reward')
                # ax[3].set_ylabel('reward')
                # ax[3].legend()
                # plt.xlabel('time')
                # plt.savefig(f'{PLOTS_FOLDER}/states_actions_{i}_{batch_idx}.png')
                # plt.close(fig)

                policy_loss, alpha_loss, bellmann_error1, bellmann_error2, \
                                cql1_loss, cql2_loss, current_alpha, lagrange_alpha_loss, lagrange_alpha \
                                = agent.learn((states, actions, rewards, next_states, dones))
                batches += 1

            if i % config.eval_every == 0:
                eval_reward = evaluate(env, agent)
                wandb.log({"Test Reward": eval_reward, "Episode": i, "Batches": batches}, step=batches)

                average10.append(eval_reward)
                print("Episode: {} | Reward: {} | Policy Loss: {} | Batches: {}".format(i, eval_reward, policy_loss, batches,))
            
            wandb.log({
                       "Average10": np.mean(average10),
                       "Policy Loss": policy_loss,
                       "Alpha Loss": alpha_loss,
                       "Lagrange Alpha Loss": lagrange_alpha_loss,
                       "CQL1 Loss": cql1_loss,
                       "CQL2 Loss": cql2_loss,
                       "Bellman error 1": bellmann_error1,
                       "Bellman error 2": bellmann_error2,
                       "Alpha": current_alpha,
                       "Lagrange Alpha": lagrange_alpha,
                       "Batches": batches,
                       "Episode": i})

            if i % config.save_every == 0:
                save(config, save_name="IQL", model=agent.actor_local, wandb=wandb, ep=0)

PLOTS_FOLDER = 'plots'
if not os.path.exists(PLOTS_FOLDER):
    os.makedirs(PLOTS_FOLDER)

if __name__ == "__main__":
    config = get_config()
    train(config)
