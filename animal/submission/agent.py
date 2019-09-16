import sys
sys.path.append('.')
import torch
from ppo.model import Policy
from ppo.model import CNNBase,FixupCNNBase,ImpalaCNNBase
from ppo.envs import  VecPyTorch, VecPyTorchFrameStack, FrameSkipEnv, TransposeImage
from animalai.envs.gym.environment import ActionFlattener
from PIL import Image
from ppo.envs import VecPyTorchFrameStack, TransposeImage, VecPyTorch
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import numpy as np
from gym.spaces import Box
import gym

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
        return  self.actions[act]


class FakeEnv(gym.Env):
    #def __init__(self):
    #    self.action_space = self._flattener.action_space
    #    self.observation_space = Box(0, 255,dtype=np.uint8,shape=(84, 84, 3))

    def set_step(self,obs,reward,done,info):
        self.obs = obs
        self.reward = reward
        self.done = done
        self.info = info

    def step(self, action_unused):
        return self.obs,self.reward,self.done,self.info

    #def reset(self, **kwargs):
    #    return self.observation_space.low

frame_skip = 2
frame_stack = 2
#CNN=FixupCNNBase
CNN=ImpalaCNNBase
reduced_actions = True

def make_env():
    env = FakeEnv()
    env = RetroEnv(env)
    if reduced_actions:
       env = FilterActionEnv(env)
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
        self.envs = VecPyTorchFrameStack(envs, frame_stack, device)
        self.flattener = self.envs.unwrapped.envs[0].flattener
        # Load the configuration and model using *** ABSOLUTE PATHS ***
        self.model_path = '/aaio/data/animal.state_dict'
        self.policy = Policy(self.envs.observation_space.shape,self.envs.action_space,base=CNN,base_kwargs={'recurrent': True})
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

