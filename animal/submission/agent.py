import sys
sys.path.append('.')
import torch
from ppo.model import Policy
from ppo.model import CNNBase,FixupCNNBase,ImpalaCNNBase,StateCNNBase
from ppo.envs import  VecPyTorch, VecPyTorchFrameStack, FrameSkipEnv, TransposeImage, VecPyTorchStateStack
from ppo.wrappers import RetroEnv,Stateful,FilterActionEnv
from animalai.envs.gym.environment import ActionFlattener
from PIL import Image
from ppo.envs import VecPyTorchFrameStack, TransposeImage, VecPyTorch
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import numpy as np
from gym.spaces import Box
import gym

class FakeAnimalEnv(gym.Env):
 
    def set_step(self,obs,reward,done,info):
        self.obs = obs
        self.reward = reward
        self.done = done
        self.info = info

    def step(self, action_unused):
        self.steps += 1
        self.env_reward += reward
        return self.obs,self.reward,self.done,self.info

    def set_maxtime(self,maxtime):
        self.maxtime = maxtime

    def reset(self):
        self.steps = 0
        self.env_reward = 0
        return np.zeros((84,84,3),dtype=np.float32)


frame_skip = 2
frame_stack = 2
state_stack = 4
#CNN=FixupCNNBase
CNN=StateCNNBase
reduced_actions = True

def make_env():
    env = FakeAnimalEnv()
    env = RetroEnv(env)
    if reduced_actions:
       env = FilterActionEnv(env)
    env = Stateful(env)

    if frame_skip > 0:
        env = FrameSkipEnv(env, skip=frame_skip)
    env = TransposeImage(env, op=[2, 0, 1])
    return env


class Agent(object):

    def __init__(self, device='cpu'):
        """
         Load your agent here and initialize anything needed
        """
        envs = DummyVecEnv([make_env])
        envs = VecPyTorch(envs, device)
        envs = VecPyTorchFrameStack(envs, frame_stack, device)
        if reduced_actions: #TODO: hugly hack
            state_size = 13
        else:
            state_size = 15
        if state_stack > 0:
            envs = VecPyTorchStateStack(envs,state_size,state_stack)
        self.envs = envs
        self.flattener = self.envs.unwrapped.envs[0].flattener
        # Load the configuration and model using *** ABSOLUTE PATHS ***
        self.model_path = '/aaio/data/animal.state_dict'
        base_kwargs={'recurrent': True}
        base_kwargs['fullstate_size'] = envs.state_size*envs.state_stack
        self.policy = Policy(self.envs.observation_space.shape,self.envs.action_space,base=CNN,base_kwargs=base_kwargs)
        self.policy.load_state_dict(torch.load(self.model_path,map_location=device))
        self.recurrent_hidden_states = torch.zeros(1, self.policy.recurrent_hidden_state_size).to(device)
        self.masks = torch.zeros(1, 1).to(device)  # set to zero
        self.device = device

    def reset(self, t=250):
        """
        Reset is called before each episode begins
        Leave blank if nothing needs to happen there
        :param t the number of timesteps in the episode
        """
        self.envs.reset()
        self.envs.unwrapped.envs[0].unwrapped.set_maxtime(t)
        self.recurrent_hidden_states = torch.zeros(1, self.policy.recurrent_hidden_state_size).to(self.device)
        self.masks = torch.zeros(1, 1).to(self.device)

    def step(self, obs, reward, done, info):
        """
        A single step the agent should take based on the current
        :param brain_info:  a single BrainInfo containing the observations and reward for a single step for one agent
        :return:            a list of actions to execute (of size 2)
        """
        self.envs.unwrapped.envs[0].unwrapped.set_step(obs,reward,done,info) #set obs,etc in fakenv
        obs, reward, done, info = self.envs.step( torch.LongTensor([[0]]) ) #apply transformations
        value, action, action_log_prob, self.recurrent_hidden_states, dist_entropy = self.policy.act(
            obs, self.recurrent_hidden_states, self.masks, deterministic=False)
        self.masks.fill_(1.0)
        action = self.flattener.lookup_action(int(action))
        return action


