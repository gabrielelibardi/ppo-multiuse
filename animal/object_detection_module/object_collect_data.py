"""
Collects pairs (obs, position) from random arenas and saves them in
a .npy file.
"""

import tqdm
import torch
import numpy as np
from ppo.envs import make_vec_envs
from ppo.model import (Policy, CNNBase, FixupCNNBase, ImpalaCNNBase,
                       StateCNNBase)
from animal.object_detection_module.object_functions import make_animal_env

CNN = {'CNN': CNNBase, 'Impala': ImpalaCNNBase, 'Fixup': FixupCNNBase,
       'State': StateCNNBase}


def collect_data(target_dir, args, list_arenas, list_params, num_samples=1000,
                 frames_episode=50):
    maker = make_animal_env(
        list_arenas, list_params, inference_mode=args.realtime,
        frame_skip=args.frame_skip, reduced_actions=args.reduced_actions,
        state=args.state)

    env = make_vec_envs(
        make=maker, num_processes=1, device=device, log_dir=None,
        num_frame_stack=args.frame_stack, state_shape=None, num_state_stack=0)

    actor_critic = Policy(
        env.observation_space.shape, env.action_space,
        base=CNN[args.cnn],
        base_kwargs={'recurrent': args.recurrent_policy})

    if args.load_model:
        actor_critic.load_state_dict(
            torch.load(args.load_model, map_location=device))
    actor_critic.to(device)
    recurrent_hidden_states = torch.zeros(
        1, actor_critic.recurrent_hidden_state_size).to(device)
    masks = torch.zeros(1, 1).to(device)

    obs_rollouts = np.zeros([num_samples, 3, 84, 84])
    labels_rollouts = np.zeros([num_samples, 10])

    t = tqdm.tqdm(range((num_samples // frames_episode) - 1))
    for episode_num in t:

        obs = env.reset()
        step = 0
        episode_obs = np.zeros([frames_episode, 3, 84, 84])
        episode_labels = np.zeros([frames_episode, 10])

        while step < frames_episode + 10:

            with torch.no_grad():
                _, action, _, _, _ = actor_critic.act(
                    obs, recurrent_hidden_states, masks,
                    deterministic=args.det)

            if step < 10:  # wait for things to fall down
                action = torch.zeros_like(action)

            # Observation reward and next obs
            obs, reward, done, info = env.step(action)

            if done:
                step = 0

            if step >= 10:
                episode_obs[step - 10, :, :, :] = obs[0].cpu().numpy()[0:3, :,
                                                  :]
                episode_labels[step - 10, :] = info[0]['label']

            masks.fill_(0.0 if done else 1.0)

            step += 1

        idx = episode_num * frames_episode + step - 10
        obs_rollouts[idx:idx + frames_episode, :, :, :] = episode_obs
        labels_rollouts[idx:idx + frames_episode, :] = episode_labels

    np.savez(target_dir,
             observations=np.array(obs_rollouts),
             labels=np.array(labels_rollouts),
             frames_per_episode=frames_episode)


if __name__ == "__main__":
    import random
    import argparse
    from animal.arenas.utils import (
        create_c1_arena,
        create_c2_arena,
        create_c3_arena,
        create_c3_arena_basic,
        create_c4_arena,
        create_c5_arena,
        create_c6_arena,
        create_c6_arena_basic,
        create_c7_arena,
        create_maze,
        create_mix_maze,
        create_arena_choice,
    )

    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--target-dir',
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
    parser.add_argument(
        '--frame-stack', type=int, default=4, help='Number of frame to stack')
    parser.add_argument(
        '--load-model', default='',
        help='directory to save agent logs (default: )')
    parser.add_argument(
        '--reduced-actions', action='store_true', default=False,
        help='Use reduced actions set')
    parser.add_argument(
        '--cnn', default='Fixup',
        help='Type of cnn. Options are CNN,Impala,Fixup,State')
    parser.add_argument(
        '--state-stack', type=int, default=4,
        help='Number of steps to stack in states')
    parser.add_argument(
        '--realtime', action='store_true', default=False,
        help='If to plot in realtime. ')

    args = parser.parse_args()
    args.det = not args.non_det
    args.state = args.cnn == 'State'
    device = torch.device(args.device)

    list_arenas = [
        create_c1_arena,
        create_c2_arena,
        create_c3_arena,
        create_c3_arena_basic,
        create_c4_arena,
        create_c5_arena,
        create_c6_arena,
        create_c6_arena_basic,
        create_c7_arena,
        create_maze,
        create_mix_maze,
        create_arena_choice,
    ]

    list_params = [
        {"max_reward": 5, "time": 250},
        {"max_reward": 5, "time": 250},
        {"time": 250},
        {"time": 250, "num_walls": np.random.randint(5, 15)},
        {"time": 250, "num_red_zones": 8, "max_orange_zones": 3},
        {"time": 250},
        {"time": 250},
        {"time": 250, "num_walls": np.random.randint(5, 15)},
        {"time": 250},
        {"time": 250, "num_cells": np.random.randint(2, 5),
         "obj": random.choice(['CylinderTunnel', 'door', 'Cardbox1'])},
        {"time": 250, "num_cells": np.random.randint(2, 7),
         "max_movable": np.random.randint(2, 7),
         "max_immovable": np.random.randint(2, 7),
         "num_red_zones": np.random.randint(0, 3),
         "max_orange_zones": np.random.randint(0, 3)},
        {"time": 250},
    ]

    collect_data(
        args.target_dir + "train_position_data",
        args, num_samples=150000,
        list_arenas=list_arenas,
        list_params=list_params)

    collect_data(
        args.target_dir + "test_position_data",
        args, num_samples=50000,
        list_arenas=list_arenas,
        list_params=list_params)
