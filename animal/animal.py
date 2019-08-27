import os
import sys
import gym
from os.path import join
import random
import numpy as np
from gym import error, spaces
from baselines.bench import load_results
from baselines import bench
from gym.spaces.box import Box
import animalai
from animalai.envs.gym.environment import AnimalAIEnv
from animal.lab import LabAnimal, RetroEnv
import time


def make_animal_env(log_dir, allow_early_resets, inference_mode, 
                    frame_skip, greyscale, arenas_dir, info_keywords=()):
    base_port = random.randint(0,100)
    def make_env(rank):
        def _thunk():
            
            if 'DISPLAY' not in os.environ.keys():
                os.environ['DISPLAY'] = ':0'
            exe = os.path.join(os.path.dirname(animalai.__file__),'../../env/AnimalAI')
            env = AnimalAIEnv(environment_filename = exe,
                               retro=False, worker_id=base_port+rank, docker_training=False, 
                               seed = 0, n_arenas = 1, arenas_configurations=None, 
                               greyscale=greyscale, inference=inference_mode,resolution=None)
            env = RetroEnv(env)
            env = LabAnimal(env,arenas_dir)

            obs_shape = env.observation_space.shape

            if frame_skip > 0: 
                env = FrameSkipEnv(env, skip=frame_skip)
                print("Frame skip: ", frame_skip, flush=True)

            if log_dir is not None:
                env = bench.Monitor(env, os.path.join(log_dir, str(rank)),
                                    allow_early_resets=allow_early_resets,
                                    info_keywords=info_keywords)

            # If the input has shape (W,H,3), wrap for PyTorch convolutions
            obs_shape = env.observation_space.shape
            if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
               env = TransposeImage(env, op=[2, 0, 1])

            return env

        return _thunk

    return make_env




class RecordPosition(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.position = np.array([20,0.5,20])

    def step(self, action):
        start = time.time()
        obs, rew, done, info = self.env.step(action)
        step_time_length = time.time() - start

        self.position  += info['brain_info'].vector_observations[0]*step_time_length
        info['Position'] = self.position

        if done:
            info['Time'] = 0
            self.position = 0

        return obs, rew, done, info



class FrameCounter(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.T_frames = 0

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        if done:
            info['Time'] = 0
            self.T_frames = 0
        self.T_frames  += 1

        info['Time'] = self.T_frames

        return obs, rew, done, info



class FrameSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        #max_frame = self._obs_buffer.max(axis=0)
        last_frame = obs

        return last_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, f"Error: Operation, {str(op)}, must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
       return ob.transpose(self.op[0], self.op[1], self.op[2])