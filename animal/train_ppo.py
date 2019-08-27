import copy
import glob
import os
import sys
import time
import random

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.append('../')

from ppo import algo, utils
from arguments import get_args
from ppo.envs import make_vec_envs
from ppo.model import Policy
from ppo.storage import RolloutStorage
from ppo.algo.ppokl import ppo_rollout, ppo_update
from animal import make_animal_env


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
        actor_critic.load_state_dict(torch.load(args.restart_model, map_location=device))
    actor_critic.to(device)

    actor_behaviors = None
    if args.behavior: 
        actor_behaviors = []
        for a in args.behavior:
            actor = Policy(envs.observation_space.shape, envs.action_space,
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
    rollouts.to(device)  #they live in GPU, converted to torch from the env wrapper

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

    for j in range(num_updates):

        ppo_rollout(args.num_steps, envs, actor_critic, rollouts)

        value_loss, action_loss, dist_entropy, kl_div = ppo_update(agent, actor_critic, rollouts,
                                    args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits)

        if (j % args.save_interval == 0 or j == num_updates - 1) and args.save_dir != "":
            actor_critic.save(os.path.join(args.save_dir, "animal.state_dict"))

        if j % args.log_interval == 0:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            s =  "Update {}, num timesteps {}, FPS {} \n".format(j, total_num_steps,int(total_num_steps / ( time.time() - start)))
            s += "Entropy {}, value_loss {}, action_loss {}, kl_divergence {}".format(dist_entropy, value_loss,action_loss,kl_div)
            print(s,flush=True)


if __name__ == "__main__":
    main()
