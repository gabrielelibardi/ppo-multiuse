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

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
from animal.animal import make_animal_env

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = os.path.join(log_dir,"eval")
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    if args.device is None:
        device = torch.device("cuda" if args.cuda else "cpu")
    else:
        device = torch.device(args.device)

    env_make = args.env_name
    if args.env_name.startswith("animal"):
        env_make = make_animal_env(log_dir = args.log_dir, allow_early_resets=False,
            inference_mode=False,  frame_skip=args.frame_skip , greyscale=False, arenas_dir=args.arenas_dir, info_keywords=('arena',))

 
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

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppokl':
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
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))

        gail_train_loader = torch.utils.data.DataLoader(
            gail.ExpertDataset(
                file_name, num_trajectories=4, subsample_frequency=20),
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=20)

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

    maml_step = 0

    for j in range(num_updates):

        if args.maml_inner_steps and maml_step==0:
            weights_before = deepcopy(actor_critic.state_dict())
            task_ids = args.num_processes*[random.randint(0,1)] #fix num tasks. Gianni
            envs.unwrapped.reset_task(task_ids)
            

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

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
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])

            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy, kl_div = agent.update(rollouts)

        rollouts.after_update()

        if args.maml_inner_steps:
            maml_step += 1
            if maml_step == args.maml_inner_steps:
                maml_step = 0 
                #update general parameters
                weights_after = actor_critic.state_dict()
                beta  = 0.2*args.lr
                actor_critic.load_state_dict({name : 
                weights_before[name] + (weights_after[name] - weights_before[name]) * beta 
                        for name in weights_before})

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            #save_path = os.path.join(args.save_dir, args.algo)
            save_path = args.save_dir
            try:
                os.makedirs(save_path)
            except OSError:
                pass
            torch.save(actor_critic.state_dict(), os.path.join(save_path, args.env_name + ".state_dict"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            s =  "Update {}, num timesteps {}, FPS {} \n".format(j, total_num_steps,int(total_num_steps / (end - start)))
            s += "Last {} training episodes: mean/std reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".format(len(episode_rewards), np.mean(episode_rewards),
                        np.std(episode_rewards), np.min(episode_rewards), np.max(episode_rewards))
            s += "Entropy {}, value_loss {}, action_loss {}, kl_divergence {}".format(dist_entropy, value_loss,action_loss,kl_div)
            print(s,flush=True)

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)


if __name__ == "__main__":
    main()
