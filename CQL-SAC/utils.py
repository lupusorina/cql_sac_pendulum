import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import wandb
import argparse
from typing import Optional
import gymnasium as gym

def save(args: argparse.Namespace,
         save_name: str,
         model: torch.nn.Module,
         wandb: wandb,
         ep: Optional[int] = None) -> None:
    save_dir = './trained_models/' 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not ep == None:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + str(ep) + ".pth")
        wandb.save(save_dir + args.run_name + save_name + str(ep) + ".pth")
    else:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + ".pth")
        wandb.save(save_dir + args.run_name + save_name + ".pth")

def evaluate(env: gym.Env,
             policy: torch.nn.Module,
             eval_runs: int = 5,
             episode_duration: int = 500,
             episode_nb: int = 0,
             plots_folder: str = 'plots') -> float:
    """
    Makes evaluation runs with the current policy
    """
    print('Evaluating')
    EPISODE_DONE_ANGLE_THRESHOLD_DEG = 0.5
    EPISODE_DURATION = episode_duration
    UPRIGHT_BUFFER_LENGTH = 10
    reward_batch = []
    durations_episodes = []
    for i in range(eval_runs):
        obs, _ = env.reset()
        obs_angle_angular_vel = np.array([np.arctan2(obs[1], obs[0]), obs[2]])
        cumulative_rewards = 0
        counter = 0
        upright_angle_buffer = []
        done = False
        output = {
            'angle': [],
            'angular_velocity': [],
            'action': [],
            'reward': [],
            'cumulative_reward': []
        }
        while True:
            action = policy.get_action(obs_angle_angular_vel, eval=True)
            obs, reward, _, _, _ = env.step(action)
            cumulative_rewards += reward

            cos_theta = obs[0]
            sin_theta = obs[1]
            obs_angle_angular_vel = np.array([np.arctan2(sin_theta, cos_theta), obs[2]])
            theta = obs_angle_angular_vel[0]

            if abs(obs_angle_angular_vel[0]) < np.deg2rad(EPISODE_DONE_ANGLE_THRESHOLD_DEG):
                upright_angle_buffer.append(theta)
            if len(upright_angle_buffer) > UPRIGHT_BUFFER_LENGTH or counter > EPISODE_DURATION:
                done = True
            if done:
                break
            counter += 1

            # Save data for plotting.
            output['angle'].append(obs_angle_angular_vel[0])
            output['angular_velocity'].append(obs_angle_angular_vel[1])
            output['action'].append(action)
            output['reward'].append(reward)
            output['cumulative_reward'].append(cumulative_rewards)

        reward_batch.append(cumulative_rewards)
        durations_episodes.append(counter)
        plot_output(data=output, episode=episode_nb, eval_run=i, plots_folder=plots_folder)

    print('Evaluation:  rewards', reward_batch)
    print('             durations_episodes:', durations_episodes)
    print('\n')
    return np.mean(reward_batch), durations_episodes

def plot_output(data: list,
                episode: int,
                eval_run: int,
                plots_folder: str) -> None:
    """
    Plots the observations, actions, and rewards of an episode
    """
    NB_ROWS = len(data.keys())
    fig, ax = plt.subplots(NB_ROWS, 1, sharex=True, figsize=(8, 8))
    for key, i in zip(data.keys(), range(NB_ROWS)):
        ax[i].plot(data[key], label=key)
        ax[i].grid()
        ax[i].legend()
    plt.xlabel('time')
    plt.savefig(f'{plots_folder}/states_actions_episode_{episode}_{eval_run}.png')
    plt.close(fig)
