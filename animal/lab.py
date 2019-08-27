import gym
import random
import glob
import os
import numpy as np
from animalai.envs.arena_config import ArenaConfig
from animalai.envs.gym.environment import ActionFlattener
from PIL import Image


class LabAnimal(gym.Wrapper):
    def __init__(self, env, arenas_dir):
        gym.Wrapper.__init__(self, env)
        if os.path.isdir(arenas_dir):
            files = glob.glob("{}/*.yaml".format(arenas_dir))
        else:
            #assume is a pattern
            files = glob.glob(arenas_dir)
            
        print(files)
        
        self.env_list = [(f,ArenaConfig(f)) for f in files]
        self._arena_file = ''

    def step(self, action):
        action = int(action)
        obs, reward, done, info = self.env.step(action)
        if reward < -0.1 and done: #dead
            reward = -20
        info['arena']=self._arena_file
        return obs, reward, done, info 

    def reset(self, **kwargs):
        self._arena_file, arena = random.choice(self.env_list)
        return self.env.reset(arenas_configurations=arena,**kwargs)
        

class RetroEnv(gym.Wrapper):
    def __init__(self,env):
        gym.Wrapper.__init__(self, env)
        self.flattener = ActionFlattener([3,3])
        self.action_space = self.flattener.action_space
        self.observation_space = gym.spaces.Box(0, 255,dtype=np.uint8,shape=(84, 84, 3))

    def step(self, action): 
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
