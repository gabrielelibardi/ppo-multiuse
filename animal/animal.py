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

def make_animal_env(log_dir, inference_mode, frame_skip, arenas_dir, info_keywords, reduced_actions):
    base_port = random.randint(0,100)
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

            if frame_skip > 0: 
                env = FrameSkipEnv(env, skip=frame_skip)
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
    for i in arena.arenas[0].items:
        if i.name in ['GoodGoal','GoodGoalBounce']:
            if len(i.sizes)==0: #arena max cannot be computed
                return -1
            max_good = max(i.sizes[0].x,max_good)
        if i.name in ['GoodGoalMulti','GoodGoalMultiBounce']:
            if len(i.sizes)==0: #arena max cannot be computed
                return -1
            tot_reward += i.sizes[0].x  

    tot_reward += max_good
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
        self.env_reward += reward
        info['arena']=self._arena_file  #for monitor
        info['max_reward']=self._max_reward
        info['max_time']=self._max_time
        info['ereward'] = self.env_reward
        return obs, reward, done, info        

    def reset(self, **kwargs):
        self.env_reward = 0
        self._arena_file, arena = random.choice(self.env_list)
        self._max_reward = analyze_arena(arena)
        self._max_time = arena.arenas[0].t
        return self.env.reset(arenas_configurations=arena,**kwargs)
        
class RewardShaping(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if reward < 0 and done: #dead for end of time or hit a killing obj
            reward += -2
        if reward > 0 and done: #prize for finishing well
            ratio = self.env_reward/self.max_reward
            ratio = ratio if ratio < 1 else 0.99  #avoid division by zero  
            reward += max(0.5/(1-ratio),5)
        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class RetroEnv(gym.Wrapper):
    def __init__(self,env):
        gym.Wrapper.__init__(self, env)
        self.flattener = ActionFlattener([3,3])
        self.action_space = self.flattener.action_space
        self.observation_space = gym.spaces.Box(0, 255,dtype=np.uint8,shape=(84, 84, 3))

    def step(self, action): 
        action = int(action)
        action = self.flattener.lookup_action(action) # convert to multi
        obs, reward, done, info = self.env.step(action)  #non-retro
        visual_obs, vector_obs = self._preprocess_obs(obs)
        info['vector_obs']=vector_obs
        return visual_obs,reward,done,info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        visual_obs, _ = self._preprocess_obs(obs)
        return visual_obs

    def _preprocess_obs(self,obs):
        visual_obs, vector_obs = obs
        visual_obs = self._preprocess_single(visual_obs)
        visual_obs = self._resize_observation(visual_obs)
        return visual_obs, vector_obs

    @staticmethod
    def _preprocess_single(single_visual_obs):
            return (255.0 * single_visual_obs).astype(np.uint8)

    @staticmethod
    def _resize_observation(observation):
        """
        Re-sizes visual observation to 84x84
        """
        obs_image = Image.fromarray(observation)
        obs_image = obs_image.resize((84, 84), Image.NEAREST)
        return np.array(obs_image)



#{0: [0, 0], 1: [0, 1], 2: [0, 2], 3: [1, 0], 4: [1, 1], 5: [1, 2], 6: [2, 0], 7: [2, 1], 8: [2, 2]}
class FilterActionEnv(gym.ActionWrapper):
    """
    An environment wrapper that limits the action space.
    """
    _ACTIONS = (0, 1, 2, 3, 4, 5, 6)

    def __init__(self, env):
        super().__init__(env)
        self.actions = self._ACTIONS
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, act):
        return self.actions[act]