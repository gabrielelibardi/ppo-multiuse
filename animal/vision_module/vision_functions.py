import os
import gym
import uuid
import math
import torch
import random
import animalai
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from animalai.envs.arena_config import ArenaConfig
from animalai.envs.gym.environment import AnimalAIEnv
from ppo.envs import TransposeImage
from animal.animal import RetroEnv


def make_animal_env(list_arenas, list_params):
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
            env = LabAnimalCollect(env, list_arenas, list_params)

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

        action_ = self.flattener.lookup_action(action)
        self._agent_pos, self._agent_rot = get_new_position(
            action_, info['vector_obs'], self._agent_pos, self._agent_rot)

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


def loss_func(real_pos, real_rot, pred_pos, pred_rot):

    pos_term, pos_error = pos_loss(real_pos, pred_pos)
    rot_term,  rot_error = rot_loss(real_rot, pred_rot.unsqueeze(-1))

    return torch.mean(pos_term + rot_term), pos_error, rot_error


def pos_loss(pos1, pos2):

    pos1 = torch.clamp(pos1, 0, 40)
    pos2 = torch.clamp(pos2, 0, 40)

    unnormalized_loss = (pos1[:, 0] - pos2[:, 0]) ** 2 + (pos1[:, 1] - pos2[:, 1]) ** 2
    loss = unnormalized_loss / 40 ** 2 * 2

    return loss.unsqueeze(-1), torch.sqrt(torch.mean(unnormalized_loss))


def rot_loss(rot1, rot2):

    aaa = abs(rot1 - rot2)
    bbb = torch.min(rot1, rot2) + 360
    unnormalized_loss = torch.min(aaa, abs(bbb - torch.max(rot1, rot2)))
    loss = unnormalized_loss / 180.

    return loss, torch.mean(unnormalized_loss)


def plot_prediction(obs, real_pos, real_rot, pred_pos, pred_rot):

    obs = obs[0, :, :, :].permute(1, 2, 0).cpu().detach().numpy()
    real_pos = torch.clamp(real_pos[0, :], 0, 40).cpu().detach().numpy()
    pred_pos = torch.clamp(pred_pos[0, :], 0, 40).cpu().detach().numpy()

    fig = plt.figure()
    gs = gridspec.GridSpec(1, 2)
    gs.update(wspace=0.1, hspace=0.1, left=0.1, right=0.9, bottom=0.1, top=0.9)

    ax1 = plt.subplot(gs[0, 0])
    plt.ylim(0, 40)
    plt.xlim(0, 40)
    ax1.scatter(real_pos[0], real_pos[1], color='r')
    ax1.scatter(pred_pos[0], pred_pos[1], color='b')
    plt.legend(['real', 'pred'])

    ax2 = plt.subplot(gs[0, 1])
    plt.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        labelbottom=False,
        labelleft=False)
    ax2.imshow(obs / 255.)

    return fig

