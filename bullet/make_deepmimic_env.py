import os
import gym
#import pybullet_envs
import numpy as np
import pybulletgym 
from baselines import bench
from ppo.envs import TransposeImage
import random
#from pybullet_envs.deep_mimic.env.pybullet_deep_mimic_env import PyBulletDeepMimicEnv
from pybullet_envs.deep_mimic.gym_env.deep_mimic_env import HumanoidDeepBulletEnv
import pybullet_data
from pybullet_utils.arg_parser import ArgParser
from pybullet_utils.logger import Logger

def make_deepmimic_env(args= '', renders = False, enable_draw = True, base_seed=0, log_dir=None, frame_skip=0, frame_stack=1, allow_early_resets=False):

    def make_env(rank):

        def _thunk():
            
            arg_parser = build_arg_parser(args)
            print("enable_draw=", enable_draw)
            env = HumanoidDeepBulletEnv(renders = renders, arg_file = "run_humanoid3d_backflip_args.txt")
            env.seed(index_worker + rank) if base_seed is None else env.seed(
                base_seed + rank)
            #if record:
            #    env = BulletlRecord(env, record)
            if log_dir is not None:
                if not os.path.isdir(log_dir):
                    os.mkdir(log_dir)
                env = bench.Monitor(
                    env, os.path.join(log_dir, "{}".format( str(rank))),
                    allow_early_resets=False)

            # If the input has shape (W,H,3), wrap for PyTorch convolutions
            obs_shape = env.observation_space.shape
            if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
                env = TransposeImage(env, op=[2, 0, 1])

            return env

        return _thunk

    return make_env


def build_arg_parser(args):
  arg_parser = ArgParser()
  arg_parser.load_args(args)

  arg_file = arg_parser.parse_string('arg_file', '')
  if arg_file == '':
    arg_file = "run_humanoid3d_backflip_args.txt"
  if (arg_file != ''):
    path = pybullet_data.getDataPath() + "/args/" + arg_file
    succ = arg_parser.load_file(path)
    Logger.print2(arg_file)
    assert succ, Logger.print2('Failed to load args from: ' + arg_file)
  return arg_parser


class BulletlRecord(gym.Wrapper):
    def __init__(self, env, record):
        gym.Wrapper.__init__(self, env)
        
        self.obs_rollouts = []
        self.rews_rollouts = []
        self.actions_rollouts = []
        self.directory = record


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