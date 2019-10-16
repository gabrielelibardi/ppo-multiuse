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


def collect_data(target_dir, args, num_samples=1000, frames_episode=50):

    maker = make_animal_env(
        inference_mode=args.realtime,
        frame_skip=args.frame_skip, reduced_actions=args.reduced_actions,
        state=args.state)

    env = make_vec_envs(
        make=maker, num_processes=args.num_processes, device=device, log_dir=None,
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
        args.num_processes, actor_critic.recurrent_hidden_state_size).to(device)
    masks = torch.zeros(1, 1).to(device)

    obs_rollouts = np.zeros([num_samples, 3, 84, 84])
    labels_rollouts = np.zeros([num_samples, 1])

    global_step = 0
    steps = [0 for _ in range(args.num_processes)]

    while global_step < (num_samples // frames_episode) - 1:

        print(global_step * frames_episode)

        obs = env.reset()
        episode_obs = np.zeros([args.num_processes, frames_episode, 3, 84, 84])
        episode_labels = np.zeros([args.num_processes, frames_episode, 1])

        with torch.no_grad():
            _, actions, _, _, _ = actor_critic.act(
                obs, recurrent_hidden_states, masks,
                deterministic=args.det)

        # wait for things to fall down
        for num_process, step in enumerate(steps):
            if step < 10:
                actions[num_process] = 0

        # Observation reward and next obs
        obs, _, dones, infos = env.step(actions)

        # after 10 steps start saving obs
        for num_process, step in enumerate(steps):
            if (frames_episode - 1) > step >= 10:

                episode_obs[
                num_process, step - 10, :, :, :
                ] = obs[num_process].cpu().numpy()[0:3, :, :]

                episode_labels[num_process, step - 10, :] = infos[
                    num_process]['label']

        # if done and max step not reached
        for num_process, done in enumerate(dones):
            if done and steps[num_process] < (frames_episode - 1):
                steps[num_process] = 0

        # if max step reached
        for num_process, done in enumerate(dones):
            if steps[num_process] == (frames_episode - 1):
                idx = global_step * frames_episode
                obs_rollouts[idx:idx + frames_episode,
                :, :, :] = episode_obs[num_process, :, :, :]
                labels_rollouts[idx:idx + frames_episode,
                :] = episode_labels[num_process, :, :]
                steps[num_process] = 0
                global_step += 1

        masks = torch.FloatTensor(
            [[0.0] if done_ else [1.0] for done_ in dones]).to(device)

        for i in range(len(steps)):
            steps[i] += 1


    np.savez(target_dir,
             observations=np.array(obs_rollouts).astype(np.uint8),
             labels=np.array(labels_rollouts).astype(np.uint8),
             frames_per_episode=frames_episode)


if __name__ == "__main__":

    import argparse

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
    parser.add_argument(
        '--num-processes',type=int,default=25,help='how many training CPU processes to use (default: 16)')

    args = parser.parse_args()
    args.det = not args.non_det
    args.state = args.cnn == 'State'
    device = torch.device(args.device)

    collect_data(
        args.target_dir + "train_object_data",
        args, num_samples=250000,
    )

    collect_data(
        args.target_dir + "test_object_data",
        args, num_samples=50000,
    )
