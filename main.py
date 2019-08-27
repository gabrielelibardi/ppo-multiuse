import copy
import glob
import os
import time
from collections import deque
from copy import deepcopy
import random

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ppo import algo, utils
from animal.arguments import get_args
from ppo.envs import make_vec_envs
from ppo.model import Policy
from ppo.storage import RolloutStorage

from animal.animal import make_animal_env

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)
    device = torch.device(args.device)

    utils.cleanup_log_dir(args.log_dir)

    env_make = make_animal_env(log_dir = args.log_dir, allow_early_resets=False,
            inference_mode=args.realtime,  frame_skip=args.frame_skip , greyscale=False, 
            arenas_dir=args.arenas_dir, info_keywords=('arena',))
 
    envs = make_vec_envs(env_make, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    actor_critic = Policy(envs.observation_space.shape,envs.action_space,
                         base_kwargs={'recurrent': args.recurrent_policy})

    if args.restart_model:
        actor_critic.load_state_dict(torch.load(args.restart_model,map_location=device))
    actor_critic.to(device)
    actor_behaviors = None
    if args.behavior: 
        actor_behaviors = []
        for a in args.behavior:
            actor = Policy(envs.observation_space.shape,envs.action_space,
                        base_kwargs={'recurrent': args.recurrent_policy})
            actor.load_state_dict(torch.load(a,map_location=device))
            actor.to(device)
            actor_behaviors.append(actor) 


    agent = algo.PPOKL(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        actor_behaviors=actor_behaviors)


    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=20)

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes


    for j in range(num_updates):
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states, dist_entropy = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])

            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()


        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy, kl_div = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0 or j == num_updates - 1) and args.save_dir != "":
            save_path = args.save_dir
            fname='animal'
            try:
                os.makedirs(save_path)
            except OSError:
                pass
            torch.save(actor_critic.state_dict(), os.path.join(save_path, fname + ".state_dict"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            s =  "Update {}, num timesteps {}, FPS {} \n".format(j, total_num_steps,int(total_num_steps / (end - start)))
            s += "Last {} training episodes: mean/std reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".format(len(episode_rewards), np.mean(episode_rewards),
                        np.std(episode_rewards), np.min(episode_rewards), np.max(episode_rewards))
            s += "Entropy {}, value_loss {}, action_loss {}, kl_divergence {}".format(dist_entropy, value_loss,action_loss,kl_div)
            print(s,flush=True)


if __name__ == "__main__":
    main()
