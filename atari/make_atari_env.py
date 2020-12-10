import os
from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from atari.wrappers import wrap_deepmind
from ppo.envs import TransposeImage
import gym
import sys
import numpy as np
import random
import torch
import glob

def make_atari_env(env_id, log_dir=None, allow_early_resets=False, test=False, base_seed=0, record = None):

    def make_env(rank):

        def _thunk():

            env = make_atari(env_id)
            env.seed(base_seed + rank)  # TODO. should be changed ?

            env = wrap_deepmind(
                env, episode_life=True if not test else False,
                clip_rewards=True if not test else False,
                scale=False)
            if record:
                env = AtariRecord(env, record)
            if log_dir is not None:
                if not os.path.isdir(log_dir):
                    os.mkdir(log_dir)
                env = bench.Monitor(
                    env, os.path.join(log_dir, "{}".format( str(rank))),
                    allow_early_resets=False)

            env = TruncateAtari(env)
            # If the input has shape (W,H,3), wrap for PyTorch convolutions
            obs_shape = env.observation_space.shape
            if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
                env = TransposeImage(env, op=[2, 0, 1])

            return env

        return _thunk

    return make_env

class TruncateAtari(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.sum_reward = 0


    def step(self, action):

        #action, other = action
        obs, reward, done, info = self.env.step(action)
        self.sum_reward += reward

        if self.sum_reward >= 5:
            obs =  obs = self.env.reset()
            self.sum_reward = 0
            done = True

        return obs, reward, done, info        

    def reset(self, **kwargs):
        
        return self.env.reset(**kwargs)


class AtariRecord(gym.Wrapper):
    def __init__(self, env, record):
        gym.Wrapper.__init__(self, env)
        
        self.obs_rollouts = []
        self.rews_rollouts = []
        self.actions_rollouts = []
        self.directory = record
        self.env_reward = 0


    def step(self, action):

        #action, other = action
        obs, reward, done, info = self.env.step(action)
        self.obs_rollouts.append(obs)
        self.rews_rollouts.append(reward)
        self.actions_rollouts.append(action)
        self.steps += 1
        self.env_reward += reward

        
        return obs, reward, done, info        

    def reset(self, **kwargs):
        
        print(self.env_reward)
        if (len (self.actions_rollouts) > 0) and (self.env_reward > 0) :
            
            self.filename = '{}/recording_{}'.format( self.directory , random.randint(0,1000000))

            print(os.path.exists('{}.npz'.format(self.filename)))
            print(self.filename)
            if not os.path.exists('{}.npz'.format(self.filename)):
            
                np.savez(self.filename,observations=np.array(self.obs_rollouts),
                            rewards=np.array(self.rews_rollouts),
                            actions=np.array(self.actions_rollouts))

        self.steps = 0
        self.env_reward = 0
        

        self.obs_rollouts = []
        self.rews_rollouts = []
        self.actions_rollouts = []

        return self.env.reset(**kwargs)