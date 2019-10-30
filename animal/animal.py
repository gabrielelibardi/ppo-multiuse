import os
import sys
import gym
import glob
from os.path import join
import random
import numpy as np
from gym import error, spaces
from baselines.bench import load_results
from baselines import bench
from gym.spaces.box import Box
import animalai
from animalai.envs.gym.environment import AnimalAIEnv
import time
from animalai.envs.arena_config import ArenaConfig
from animalai.envs.gym.environment import ActionFlattener
from ppo.envs import FrameSkipEnv,TransposeImage
from PIL import Image
from wrappers import RetroEnv,Stateful,FilterActionEnv
import os.path as osp

def make_animal_env(log_dir, inference_mode, frame_skip, arenas_dir, info_keywords, reduced_actions, seed, state):
    base_port = 100*seed  # avoid collisions
    def make_env(rank):
        def _thunk():
            
            if 'DISPLAY' not in os.environ.keys():
                os.environ['DISPLAY'] = ':0'
            exe = os.path.join(os.path.dirname(animalai.__file__),'../../env/AnimalAI')
            env = AnimalAIEnv(environment_filename = exe,
                               retro=False, worker_id=base_port+rank, docker_training=False, 
                               seed = 0, n_arenas = 1, arenas_configurations=None, 
                               greyscale=False, inference=inference_mode,resolution=None)
            env = RetroEnv(env)
            if reduced_actions:
                env = FilterActionEnv(env)
            env = LabAnimal(env,arenas_dir)

            env = RewardShaping(env)
            
            if state:
                env = Stateful(env)

            if frame_skip > 0: 
                env = FrameSkipEnv(env, skip=frame_skip)   #TODO:Is this wrong here? Are we double counting rewards? Infos?
                print("Frame skip: ", frame_skip, flush=True)

            if log_dir is not None:
                env = bench.Monitor(env, os.path.join(log_dir, str(rank)),
                                    allow_early_resets=False,
                                    info_keywords=info_keywords)

            # If the input has shape (W,H,3), wrap for PyTorch convolutions
            obs_shape = env.observation_space.shape
            if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
               env = TransposeImage(env, op=[2, 0, 1])

            return env

        return _thunk

    return make_env



def analyze_arena(arena):
    tot_reward = 0
    max_good = 0
    goods = []
    goodmultis = []
    for i in arena.arenas[0].items:
        if i.name in ['GoodGoal','GoodGoalBounce']:
            if len(i.sizes)==0: #arena max cannot be computed
                return -1
            max_good = max(i.sizes[0].x,max_good)
            goods.append(i.sizes[0].x)
        if i.name in ['GoodGoalMulti','GoodGoalMultiBounce']:
            if len(i.sizes)==0: #arena max cannot be computed
                return -1
            tot_reward += i.sizes[0].x
            goodmultis.append(i.sizes[0].x)  

    tot_reward += max_good
    goods.sort()
    goodmultis.sort()
    return tot_reward

def random_size_reward():
    #according to docs it's 0.5-5
    s = random.randint(5, 50)/10
    return (s,s,s)

from animalai.envs.arena_config import Vector3

def set_reward_arena(arena, force_new_size = False):
    tot_reward = 0
    max_good = 0
    goods = []
    goodmultis = []
    for i in arena.arenas[0].items:
        if i.name in ['GoodGoal','GoodGoalBounce']:
            if len(i.sizes)==0 or force_new_size:
                x,y,z = random_size_reward() 
                i.sizes = [] #remove previous size if there
                i.sizes.append(Vector3(x,y,z))
            max_good = max(i.sizes[0].x,max_good)
            goods.append(i.sizes[0].x)
        if i.name in ['GoodGoalMulti','GoodGoalMultiBounce']:
            if len(i.sizes)==0 or force_new_size: 
                x,y,z = random_size_reward() 
                i.sizes = [] #remove previous size if there
                i.sizes.append(Vector3(x,y,z))
            tot_reward += i.sizes[0].x
            goodmultis.append(i.sizes[0].x)  

    tot_reward += max_good
    goods.sort()
    goodmultis.sort()
    return tot_reward

class LabAnimal(gym.Wrapper):
    def __init__(self, env, arenas_dir):
        gym.Wrapper.__init__(self, env)
        if os.path.isdir(arenas_dir):
            files = glob.glob("{}/*.yaml".format(arenas_dir))
        else:
            #assume is a pattern
            files = glob.glob(arenas_dir)
        
        self.env_list = [(f,ArenaConfig(f)) for f in files]
        self._arena_file = ''

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.steps += 1
        self.env_reward += reward
        info['arena']=self._arena_file  #for monitor
        info['max_reward']=self.max_reward
        info['max_time']=self.max_time
        info['ereward'] = self.env_reward
        return obs, reward, done, info        

    def reset(self, **kwargs):
        self.steps = 0
        self.env_reward = 0
        self._arena_file, arena = random.choice(self.env_list)
#        self.max_reward = analyze_arena(arena)
        self.max_reward = set_reward_arena(arena, force_new_size=False)
        self.max_time = arena.arenas[0].t
        return self.env.reset(arenas_configurations=arena,**kwargs)


# NOT WORKING< WORK IN PROGRESS
class LabAnimalSampler(gym.Wrapper):
    def __init__(self, env, arenas_dir, log_dir=None):
        gym.Wrapper.__init__(self, env)
        if os.path.isdir(arenas_dir):
            files = glob.glob("{}/*.yaml".format(arenas_dir))
        else:
            #assume is a pattern
            files = glob.glob(arenas_dir)
        
        self.env_list = [(f,ArenaConfig(f)) for f in files]
        self._num_arenas = len(self.env_list)
        self._arena_file = ''
        self._num_episodes = 0
        self._log_dir = log_dir
        self._arena_weights = 1/self._num_arenas*np.ones((self._num_arenas,))  #equal prob
        

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.steps += 1
        self.env_reward += reward
        info['arena']=self._arena_file  #for monitor
        info['max_reward']=self.max_reward
        info['max_time']=self.max_time
        info['ereward'] = self.env_reward
        return obs, reward, done, info        

    def reset(self, **kwargs):
        self.steps = 0
        self.env_reward = 0
        stats_period = 100 # every 100 episodes of this worker 
        if self._num_episodes % stats_period == 0  and self._num_episodes > 0:
            self._arena_weights = self._compute_stats()
        idx = np.random.choice(self._num_arenas,p=self._arena_weights)
        self._arena_file, arena = self.env_list[idx]
        self.max_reward = set_reward_arena(arena, force_new_size=False)
        self.max_time = arena.arenas[0].t
        self._num_episodes += 1
        return self.env.reset(arenas_configurations=arena,**kwargs)

    def _compute_stats():
        df = load_results(log_dir)
        last = df.iloc[-num_episodes:]  #last episodes


class RewardShaping(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if reward>-0.005 and reward<0:#remove time negative reward
            reward = 0
        if done: #give time penalty at the end
            reward -= self.steps/self.max_time
        if reward>0 and done and self.steps<60: #explore first
            reward = 0
        if reward>0 and not done:#brown ball, go for it first
            reward+=3
        if reward > 0 and self.env_reward>self.max_reward-1 and done: #prize for finishing well
            reward += 10
        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class LabAnimal2(gym.Wrapper):
    def __init__(self, env, arenas_dir):
        gym.Wrapper.__init__(self, env)
        print('START LOADING ARENAS')
        
        files1 = glob.glob("{}/*.yaml".format(arenas_dir))
        print('LENGTH FILE1',len(files1))
        #import pdb; pdb.set_trace()
        arenas_dir_all = osp.dirname(arenas_dir)
        #arenas_dir_all = "/".join(arenas_dir.split("/")[:-1])
        files2 = glob.glob("{}/*/*.yaml".format(arenas_dir_all))
        print('LENGTH FILE2',len(files2))
            
        
        self.env_list_1 = [(f,ArenaConfig(f)) for f in files1]
        print('LENGTH FILE1',len(self.env_list_1))

        self.env_list_2 = [(f,ArenaConfig(f)) for f in files2]
        print('LENGTH FILE2',len(self.env_list_2))

        self._arena_file = ''

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.steps += 1
        self.env_reward += reward
        info['arena']=self._arena_file  #for monitor
        info['max_reward']=self.max_reward
        info['max_time']=self.max_time
        info['ereward'] = self.env_reward
        return obs, reward, done, info        

    def reset(self, **kwargs):
        self.steps = 0
        self.env_reward = 0
        if random.random() > 0.2:
            self._arena_file, arena = random.choice(self.env_list_1)
        else:
            self._arena_file, arena = random.choice(self.env_list_2)

#        self.max_reward = analyze_arena(arena)
        self.max_reward = set_reward_arena(arena, force_new_size=False)
        self.max_time = arena.arenas[0].t
        return self.env.reset(arenas_configurations=arena,**kwargs)

