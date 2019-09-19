import os
import gym
import uuid
import random
import animalai
import numpy as np
from animalai.envs.arena_config import ArenaConfig
from animalai.envs.gym.environment import AnimalAIEnv
from ppo.envs import TransposeImage
from ..animal import RetroEnv


def make_animal_env():
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
                greyscale=False, inference=False,
                resolution=None)

            env = RetroEnv(env)
            env = LabAnimalCollect(env)

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
        self._agent_pos = None
        self._agent_rot = None

    def step(self, action):
        action = int(action)
        obs, reward, done, info = self.env.step(action)

        self._env_steps += 1
        info['arena'] = self._arena_file
        info['arena_type'] = self._type

        self._agent_pos, self._agent_rot = get_new_position(
            action, info['vector_obs'], self._agent_pos, self._agent_rot)

        info['agent_position'] = self._agent_pos
        info['agent_rotation'] = self._agent_rot

        return obs, reward, done, info

    def reset(self, **kwargs):

        # Create new arena
        name = str(uuid.uuid4())
        index = random.choice(range(self._num_arenas))
        arena_func = self.list_arenas[index]
        params = self.list_params[index]
        arena_type, agent_pos, agent_rot = arena_func("/tmp/", name, **params)
        self._arena_file, arena = ("/tmo/{}.yaml".format(name), ArenaConfig(
            "/tmp/{}.yaml".format(name)))
        os.remove("/tmp/{}.yaml".format(name))

        self._type = arena_type
        self._env_steps = 0
        self._agent_pos = agent_pos
        self._agent_rot = agent_rot

        return self.env.reset(arenas_configurations=arena, **kwargs)


def get_new_position(action, speed, current_pos, current_rot):
    """ Calculates next agent position and rotation """

    mov_act = action[0]
    rotation_act = action[1]

    if mov_act == 1:
        direction = 1
    elif mov_act == 2:
        direction = -1
    else:
        direction = 0

    if rotation_act == 1:
        rotation_sign = 1
    elif rotation_act == 2:
        rotation_sign = -1
    else:
        rotation_sign = 0

    if current_rot + 6 * rotation_sign > 360:
        current_rot = 6 - abs(360 - current_rot)
    if current_rot + 6 * rotation_sign < 0:
        current_rot = 360 - abs(current_rot - 6)
    else:
        current_rot += (6 * rotation_sign)

    speed_magnitude = np.sqrt(speed[0] ** 2 + speed[2] ** 2)

    y_comp = np.cos(np.radians(current_rot))
    x_comp = np.sin(np.radians(current_rot))

    speed[0] = speed_magnitude * x_comp
    speed[2] = speed_magnitude * y_comp
    step = 0.06

    current_pos += (step * speed * direction)

    return current_pos, current_rot
