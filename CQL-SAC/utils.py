import torch
import numpy as np

def save(args, save_name, model, wandb, ep=None):
    import os
    save_dir = './trained_models/' 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not ep == None:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + str(ep) + ".pth")
        wandb.save(save_dir + args.run_name + save_name + str(ep) + ".pth")
    else:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + ".pth")
        wandb.save(save_dir + args.run_name + save_name + ".pth")


EPISODE_DONE_ANGLE_THRESHOLD_DEG = 0.5

def collect_random(env, dataset, num_samples=200):
    state, _ = env.reset()
    done = False
    counter = 0
    upright_angle_buffer = []
    for _ in range(num_samples):
        action = env.action_space.sample()
        next_state, reward, done, _, _ = env.step(action)
        cos_theta = state[0]
        sin_theta = state[1]
        theta = np.arctan2(sin_theta, cos_theta)
        if abs(theta) < np.deg2rad(EPISODE_DONE_ANGLE_THRESHOLD_DEG):
            upright_angle_buffer.append(theta)
        if len(upright_angle_buffer) > 40 or counter > 500:
            done = True
        dataset.add(state, action, reward, next_state, done)
        state = next_state
        if done:
            state, _ = env.reset()
            upright_angle_buffer = []
            done = False
            counter = 0
        counter = counter + 1


def evaluate(env, policy, eval_runs=5):
    """
    Makes an evaluation run with the current policy
    """
    print("Evaluating policy")
    reward_batch = []
    for i in range(eval_runs):
        print(f"Run {i+1}/{eval_runs}")
        state, _ = env.reset()
        upright_angle_buffer = []
        done = False
        rewards = 0
        counter = 0
        while True:
            action = policy.get_action(state, eval=True)

            state, reward, _, _, _ = env.step(action)
            rewards += reward
            cos_theta = state[0]
            sin_theta = state[1]
            theta = np.arctan2(sin_theta, cos_theta)
            if abs(theta) < np.deg2rad(EPISODE_DONE_ANGLE_THRESHOLD_DEG):
                upright_angle_buffer.append(theta)
            if len(upright_angle_buffer) > 40 or counter > 500:
                done = True
            if done:
                break
            counter = counter + 1
        reward_batch.append(rewards)
    return np.mean(reward_batch)