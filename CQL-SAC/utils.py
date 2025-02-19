import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def save(args, save_name, model, wandb, ep=None):
    save_dir = './trained_models/' 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not ep == None:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + str(ep) + ".pth")
        wandb.save(save_dir + args.run_name + save_name + str(ep) + ".pth")
    else:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + ".pth")
        wandb.save(save_dir + args.run_name + save_name + ".pth")

def evaluate(env, policy, eval_runs=5, episode_nb=0, plots_folder='plots'):
    """
    Makes an evaluation run with the current policy
    """
    print('Evaluating')
    EPISODE_DONE_ANGLE_THRESHOLD_DEG = 0.5
    reward_batch = []
    durations_episodes = []
    for i in range(eval_runs):
        state, _ = env.reset()
        state_angle_angular_vel = np.array([np.arctan2(state[1], state[0]), state[2]])
        rewards = 0
        counter = 0
        upright_angle_buffer = []
        done = False
        output = []
        while True:
            action = policy.get_action(state_angle_angular_vel, eval=True)
            state, reward, _, _, _ = env.step(action)
            rewards += reward
            counter += 1
            cos_theta = state[0]
            sin_theta = state[1]
            state_angle_angular_vel = np.array([np.arctan2(sin_theta, cos_theta), state[2]])
            theta = state_angle_angular_vel[0]
            if abs(state_angle_angular_vel[0]) < np.deg2rad(EPISODE_DONE_ANGLE_THRESHOLD_DEG):
                upright_angle_buffer.append(theta)
            if len(upright_angle_buffer) > 10 or counter > 500:
                done = True
            if done:
                break
            output.append([theta, state[2], action[0], reward])
        reward_batch.append(rewards)
        durations_episodes.append(counter)
        plot_output(data=output, episode=episode_nb, eval_run=i, plots_folder=plots_folder)

    print('Evaluation:  rewards', reward_batch)
    print('             durations_episodes:', durations_episodes)
    print('\n')
    return np.mean(reward_batch)

def plot_output(data, episode, eval_run, plots_folder):
    """
    Plots the states and actions of an episode
    """
    output_np = np.array(data)
    fig, ax = plt.subplots(5, 1, sharex=True, figsize=(8, 8))
    ax[0].plot(output_np[:, 0], label='angle')
    ax[0].set_ylabel('angle')
    ax[0].legend()
    ax[1].plot(output_np[:, 1], label='angular velocity')
    ax[1].set_ylabel('angular velocity')
    ax[1].legend()
    ax[2].plot(output_np[:, 2], label='action')
    ax[2].set_ylabel('action')
    ax[2].legend()
    ax[3].plot(output_np[:, 3], label='reward')
    ax[3].set_ylabel('reward')
    ax[3].legend()
    ax[4].plot(np.cumsum(output_np[:, 3]), label='cumulative reward')
    ax[4].set_ylabel('cumulative reward')
    ax[4].legend()
    plt.xlabel('time')
    plt.savefig(f'{plots_folder}/states_actions_episode_{episode}_{eval_run}.png')
    plt.close(fig)
