import os
import gym
import uuid
import torch
import torch.nn as nn
import random
import animalai
import numpy as np
from ppo.envs import TransposeImage
from animalai.envs.arena_config import ArenaConfig
from animalai.envs.gym.environment import AnimalAIEnv
from animal.animal import RetroEnv, FrameSkipEnv
from animal.wrappers import RetroEnv, Stateful, FilterActionEnv


def create_label(arena):
    """ Multi-target binary label for objects."""
    label = np.zeros(10)

    for i in arena.arenas[0].items:

        if i.name == 'Wall':
            label[0] = 1
        elif i.name == 'WallTransparent':
            label[1] = 1
        elif i.name == 'Ramp':
            label[2] = 1
        elif i.name == 'CylinderTunnel':
            label[3] = 1
        elif i.name == 'CylinderTunnelTransparent':
            label[4] = 1
        elif i.name == 'Cardbox1':
            label[5] = 1
        elif i.name == 'Cardbox2':
            label[6] = 1
        elif i.name == 'UObject':
            label[7] = 1
        elif i.name == 'LObject':
            label[8] = 1
        elif i.name == 'LObject2':
            label[9] = 1

    return label


def compute_error(label, prediction):
    """ Compute batch error as number of objects wrongly classified. """

    error = (label == prediction)
    error = torch.sum(error, dim=0)
    error = torch.mean(error, dim=0)

    return error


class Loss:
    """ Multi-target binary loss. """

    def __init__(self):

        self.loss = nn.BCEWithLogitsLoss(reduce=False, reduction=None)

    def compute(self, logits, prediction):

        loss = nn.BCEWithLogitsLoss(logits, prediction)
        loss = torch.sum(loss, dim=0)
        loss = torch.mean(loss, dim=0)

        return loss


def make_animal_env(list_arenas, list_params, inference_mode, frame_skip, reduced_actions, state):

    base_port = random.randint(0, 100)
    def make_env(rank):
        def _thunk():

            if 'DISPLAY' not in os.environ.keys():
                os.environ['DISPLAY'] = ':0'
            exe = os.path.join(os.path.dirname(animalai.__file__),'../../env/AnimalAI')
            env = AnimalAIEnv(
                environment_filename=exe,
                retro=False, worker_id=base_port + rank,
                docker_training=False,
                seed=0, n_arenas=1, arenas_configurations=None,
                greyscale=False, inference=inference_mode,
                resolution=None)

            env = RetroEnv(env)
            env = LabAnimalCollect(env, list_arenas, list_params)

            if reduced_actions:
                env = FilterActionEnv(env)

            if state:
                env = Stateful(env)

            if frame_skip > 0:
                env = FrameSkipEnv(env, skip=frame_skip)
                print("Frame skip: ", frame_skip, flush=True)

            # If the input has shape (W,H,3), wrap for PyTorch convolutions
            obs_shape = env.observation_space.shape
            if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
                env = TransposeImage(env, op=[2, 0, 1])

            return env

        return _thunk

    return make_env


class LabAnimalCollect(gym.Wrapper):
    def __init__(self, env, list_arenas, list_params):

        gym.Wrapper.__init__(self, env)
        self._num_arenas = len(list_arenas)
        assert self._num_arenas == len(list_params)
        self.list_arenas = list_arenas
        self.list_params = list_params
        self._arena_file = ''
        self._type = None
        self._env_steps = None
        self._label = None

    def step(self, action):
        action = int(action)
        obs, reward, done, info = self.env.step(action)
        self._env_steps += 1
        info['arena'] = self._arena_file
        info['arena_type'] = self._type
        info['label'] = self._label

        return obs, reward, done, info

    def reset(self, **kwargs):
        # Create new arena
        name = str(uuid.uuid4())
        index = random.choice(range(self._num_arenas))
        arena_func = self.list_arenas[index]
        params = self.list_params[index]
        arena_type, _, _ = arena_func("/tmp/", name, **params)
        self._arena_file, arena = ("/tmo/{}.yaml".format(name), ArenaConfig(
            "/tmp/{}.yaml".format(name)))
        os.remove("/tmp/{}.yaml".format(name))

        self._type = arena_type
        self._env_steps = 0
        self._label = create_label(arena)

        return self.env.reset(arenas_configurations=arena, **kwargs)
