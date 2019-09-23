"""
Collects pairs (obs, position) from random arenas and saves them in
a .npy file.
"""

import tqdm
import torch
import random
import numpy as np
from ppo.envs import make_vec_envs
from ppo.model import Policy, FixupCNNBase
from vision_functions import make_animal_env


def collect_data(target_dir, args, list_arenas, list_params, num_samples=1000, frames_episode=20):

    args.det = not args.non_det
    device = torch.device(args.device)

    maker = make_animal_env(list_arenas, list_params)

    env = make_vec_envs(
        make=maker, seed=0, num_processes=1, gamma=None,
        device=device, log_dir=None, allow_early_resets=True,
        num_frame_stack=1)

    actor_critic = Policy(
        env.observation_space.shape, env.action_space,
        base=FixupCNNBase,
        base_kwargs={'recurrent': args.recurrent_policy})

    actor_critic.to(device)
    recurrent_hidden_states = torch.zeros(
        1, actor_critic.recurrent_hidden_state_size).to(device)
    masks = torch.zeros(1, 1).to(device)

    obs_rollouts = np.zeros([num_samples, 3, 84, 84])
    pos_rollouts = np.zeros([num_samples, 3])
    rot_rollouts = np.zeros([num_samples, 1])

    t = tqdm.tqdm(range(num_samples // frames_episode))
    for episode_num in t:

        obs = env.reset()

        for step in range(frames_episode + 10):

            with torch.no_grad():
                _, action, _, _, _ = actor_critic.act(
                    obs, recurrent_hidden_states, masks,
                    deterministic=args.det)

            if step < 10:  # wait for things to fall down
                action = torch.zeros_like(action)

            # Observation reward and next obs
            obs, reward, done, info = env.step(action)

            if step >= 10:
                idx = episode_num * frames_episode + step - 10
                obs_rollouts[idx, :, :, :] = obs[0].cpu().numpy()
                pos_rollouts[idx, :] = info[0]['agent_position']
                rot_rollouts[idx, :] = info[0]['agent_rotation']

            masks.fill_(0.0 if done else 1.0)

    np.savez(target_dir,
             observations=np.array(obs_rollouts),
             positions=pos_rollouts,
             rotations=rot_rollouts,
             frames_per_episode=frames_episode)

if __name__ == "__main__":

    import argparse
    from animal.arenas.utils import (
        create_c1_arena,
        create_c2_arena,
        create_c3_arena,
        create_c4_arena,
        create_c5_arena,
        create_c6_arena,
        create_c7_arena,
    )

    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--target_dir',
        help='Directory to save data.')
    parser.add_argument(
        '--device', default='cuda:0',
        help='Cuda device  or cpu (default:cuda:0 )')
    parser.add_argument(
        '--non-det', action='store_true', default=True,
        help='whether to use a non-deterministic policy')
    parser.add_argument(
        '--recurrent-policy', action='store_true', default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--frame-skip', type=int, default=0,
        help='Number of frame to skip for each action')

    args = parser.parse_args()

    collect_data(
        args.target_dir + "train_position_data",
        args, num_samples=250000,
        list_arenas=[
            create_c1_arena,
            create_c2_arena,
            create_c3_arena,
            create_c4_arena,
            create_c5_arena,
            create_c6_arena,
            create_c7_arena,
        ],
        list_params=[
            {"max_reward": 5, "time": 250},
            {"max_reward": 5, "time": 250},
            {"time": 250, "num_movable": 1, "num_immovable": 1},
            {"time": 250, "num_red_zones": 8, "max_orange_zones": 3},
            {"time": 250},
            {"time": 250},
            {"time": 250},
        ])

    collect_data(
        args.target_dir + "test_position_data",
        args, num_samples=30000,
        list_arenas=[
            create_c1_arena,
            create_c2_arena,
            create_c3_arena,
            create_c4_arena,
            create_c5_arena,
            create_c6_arena,
            create_c7_arena,
        ],
        list_params=[
            {"max_reward": 5, "time": 250},
            {"max_reward": 5, "time": 250},
            {"time": 250, "num_movable": 1, "num_immovable": 1},
            {"time": 250, "num_red_zones": 8, "max_orange_zones": 3},
            {"time": 250},
            {"time": 250},
            {"time": 250},
        ])
