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
import torch
import pickle
import matplotlib.pyplot as plt

def make_animal_env(log_dir, inference_mode, frame_skip, arenas_dir, info_keywords, reduced_actions, seed, state, replay_ratio, record_actions, schedule_ratio, demo_dir):
    base_port = 100*seed  # avoid collisions
    def make_env(rank):
        def _thunk():
            
            if 'DISPLAY' not in os.environ.keys():
                os.environ['DISPLAY'] = ':0'
            exe = os.path.join(os.path.dirname(animalai.__file__),'../../env/AnimalAI')
            env = AnimalAIEnv(environment_filename = exe,
                               retro=False, worker_id=base_port+rank, docker_training=False, 
                               seed = 7, n_arenas = 1, arenas_configurations=None, 
                               greyscale=False, inference=inference_mode,resolution=None)
            env = RetroEnv(env)
            if reduced_actions:
                env = FilterActionEnv(env)
            if record_actions: 
                env = LabAnimalRecordAction(env,arenas_dir,replay_ratio, record_actions)
            else:
                #env = LabAnimalReplayAll(env,arenas_dir,replay_ratio, schedule_ratio, demo_dir)
                env = LabAnimalReplayRecord(env,arenas_dir,replay_ratio, schedule_ratio, demo_dir)
                #env = LabAnimal(env, arenas_dir,replay_ratio)
            #env = RewardShaping(env)
            
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


class ReplayAll():
    def __init__(self,replay_ratio,arenas,schedule_ratio, demo_dir):
        self.rho = replay_ratio
        self.replay = False
        #self.flattener = ActionFlattener([3,3])
        
        self.schedule_ratio = schedule_ratio
        """ for arena  in arenas:
            actions_name =  arena.split('/')[-1].split('.')[0]
            filename = '/workspace7/Unity3D/gabriele/Animal-AI/animal-ppo/RUNS/recorded_reason/{}.npz'.format(actions_name)
            if os.path.isfile(filename):
                self.recordings[arena] = np.load(filename) """
        self.demo_dir = demo_dir
        self.files = glob.glob("{}/*".format(demo_dir))
        #self.files = glob.glob("{}/*".format('/workspace7/Unity3D/gabriele/Animal-AI/animal-ppo/RUNS/recorded_reason2'))
        
        self.recordings = [np.load(filename) for ii, filename in enumerate(self.files) if 100 >= ii ]
        self.num_extra = len(self.recordings) -1
        self.max_N_demonstrations = 100



    def replay_step(self, action):
        print('LEN DEMOS',len(self.recordings) )

        if self.replay == True:
            if  self.num_steps > self.step:
                act = self.acts[self.step]
                obs = self.obs[self.step]
                reward = self.rews[self.step]
                if self.step == (self.num_steps -1):
                    done = True
                else:
                    done = False
                self.step +=1
                return [act, obs, reward, done]
                
            else:
                return [action]
        else:
            return [action]
    
    def reset(self,arena_name, average_performance):
        
        if self.schedule_ratio:
            rho = (1 - average_performance)*self.rho
        else:
            rho = self.rho
        #actions_name =  arena_name.split('/')[-1].split('.')[0]
        #self.filename = '/workspace7/Unity3D/gabriele/Animal-AI/animal-ppo/RUNS/recorded_reason/{}.npz'.format(actions_name)
        #if os.path.isfile( self.filename):
        if len(self.recordings) != 0:
            recording = random.choice(self.recordings)

            if  random.choices([0,1], weights=[rho, 1 - rho])[0] == 0:
                self.replay = True
                
                self.acts = recording['actions']
                self.obs = recording['observations']
                self.rews = recording['rewards']
                self.num_steps = self.acts.shape[0]
                self.step = 0
            else:
                self.replay = False
        else:
            self.replay = False

        return self.replay
    
        
    def add_demo(self,demo):

        self.recordings.insert(1,demo)
        self.num_extra = len(self.recordings) -1

        if self.num_extra >=  self.max_N_demonstrations:
            self.recordings.pop()



class LabAnimalReplayRecord(gym.Wrapper):
    def __init__(self, env, arenas_dir, replay_ratio,schedule_ratio, demo_dir):
        gym.Wrapper.__init__(self, env)
        if os.path.isdir(arenas_dir):
            #files = glob.glob("{}/*/*.yaml".format(arenas_dir)) + glob.glob("{}/*.yaml".format(arenas_dir))
            files = glob.glob("{}/*.yml".format(arenas_dir)) + glob.glob("{}/*.yaml".format(arenas_dir))
        else:
            #assume is a pattern
            files = glob.glob(arenas_dir)
        
        self.env_list = [(f,ArenaConfig(f)) for f in files]
        self._arena_file = ''
        self.replayer = ReplayAll(replay_ratio,files,schedule_ratio, demo_dir)
        self.performance_tracker = np.zeros(1000)
        self.n_arenas = 0
        self.directory = demo_dir
        self.obs_rollouts = []
        self.rews_rollouts = []
        self.actions_rollouts = []
        self.demo = None
    

    def step(self, action_):
        action, demo_in = action_
        out =self.replayer.replay_step(action)
        info = {}
        if len(out) == 1:
            obs, reward, done, info = self.env.step(action)
            self.obs_rollouts.append(obs)
            self.rews_rollouts.append(reward)
            self.actions_rollouts.append(action)
            if (reward > -0.01 ) and (reward < 0):
                reward = 0 # get rid of the time reward
            
            self.env_reward_no_D += reward
            self.len_real +=1
            
            info['action'] = 99
        else:
            action, obs, reward, done = out
            
            if (reward > -0.01 ) and (reward < 0):
                reward = 0 # get rid of the time reward
        
            info['action'] = action
            self.len_real  = 0
             
        self.steps += 1

        self.env_reward += reward
        info['arena']=self._arena_file  #for monitor
        info['max_reward']=self.max_reward
        info['max_time']=self.max_time
        info['ereward'] = self.env_reward
        info['reward_woD'] = self.env_reward_no_D
        info['len_real'] = self.len_real
        if self.demo:
            info['demo_out'] = self.demo
            self.demo = None
        else:
            info['demo_out'] = None

        for demo_in_ in demo_in:
            self.replayer.add_demo(demo_in_)

        return obs, reward, done, info

    def reset(self, **kwargs):
        self.n_arenas += 1
        self.steps = 0
        self.env_reward = 0
        self.env_reward_no_D = 0
        self.len_real = 0
        if (len (self.actions_rollouts) > 0) and (sum(self.rews_rollouts) > 0.5):
            arena_name =  self._arena_file.split('/')[-1].split('.')[0]
            self.filename = '{}/{}_{}'.format(self.directory , arena_name, random.getrandbits(50))

            
            self.demo = {'name':self.filename,'observations':np.array(self.obs_rollouts),
                            'rewards':np.array(self.rews_rollouts),
                            'actions':np.array(self.actions_rollouts) }

            """ np.savez(self.filename,observations=np.array(self.obs_rollouts),
                            rewards=np.array(self.rews_rollouts),
                            actions=np.array(self.actions_rollouts)) """
          
    
           
        self.obs_rollouts = []
        self.rews_rollouts = []
        self.actions_rollouts = []

        if self.demo:
            self.replayer.add_demo(self.demo)

        self._arena_file, arena = random.choice(self.env_list)

        average_performance = np.average(self.performance_tracker)
        replay = self.replayer.reset(self._arena_file, average_performance)
#        self.max_reward = analyze_arena(arena)
        self.max_reward = set_reward_arena(arena, force_new_size=False)
        self.max_time = arena.arenas[0].t
        return self.env.reset(arenas_configurations=arena,**kwargs) 

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


class LabAnimalReplayAll(gym.Wrapper):
    def __init__(self, env, arenas_dir, replay_ratio,schedule_ratio, demo_dir):
        gym.Wrapper.__init__(self, env)
        if os.path.isdir(arenas_dir):
            files = glob.glob("{}/*/*.yaml".format(arenas_dir)) + glob.glob("{}/*.yaml".format(arenas_dir))
            
        else:
            #assume is a pattern
            files = glob.glob(arenas_dir)
        
        self.env_list = [(f,ArenaConfig(f)) for f in files]
        self._arena_file = ''
        self.replayer = ReplayAll(replay_ratio,files,schedule_ratio, demo_dir)
        self.performance_tracker = np.zeros(1000)
        self.n_arenas = 0

    def step(self, action):
        out =self.replayer.replay_step(action)
        info = {}
        if len(out) == 1:
            obs, reward, done, info = self.env.step(action)
            if (reward > -0.01 ) and (reward < 0):
                reward = 0 # get rid of the time reward
            
            self.env_reward_no_D += reward
            info['action'] = 99
        else:
            action, obs, reward, done = out
            
            if (reward > -0.01 ) and (reward < 0):
                reward = 0 # get rid of the time reward
        
            info['action'] = action
                  
        self.steps += 1

        self.env_reward += reward
        info['arena']=self._arena_file  #for monitor
        info['max_reward']=self.max_reward
        info['max_time']=self.max_time
        info['ereward'] = self.env_reward
        info['reward_woD'] = self.env_reward_no_D
        if done:
            self.performance_tracker[self.n_arenas % 1000] = max(self.env_reward_no_D, 0)/self.max_reward

        return obs, reward, done, info        

    def reset(self, **kwargs):
        self.n_arenas += 1
        self.steps = 0
        self.env_reward = 0
        self.env_reward_no_D = 0
        
        """  while True:
            self._arena_file, arena = random.choice(self.env_list)
            replay = self.replayer.reset(self._arena_file)
            if replay:
                break """
        self._arena_file, arena = random.choice(self.env_list)

        average_performance = np.average(self.performance_tracker)
        replay = self.replayer.reset(self._arena_file, average_performance)
#        self.max_reward = analyze_arena(arena)
        self.max_reward = set_reward_arena(arena, force_new_size=False)
        self.max_time = arena.arenas[0].t
        return self.env.reset(arenas_configurations=arena,**kwargs)    

class LabAnimal(gym.Wrapper):
    def __init__(self, env, arenas_dir, replay_ratio):
        gym.Wrapper.__init__(self, env)
        if os.path.isdir(arenas_dir):
            files = glob.glob("{}/*.yaml".format(arenas_dir))
        else:
            #assume is a pattern
            files = glob.glob(arenas_dir)
        
        self.env_list = [(f,ArenaConfig(f)) for f in files]
        self._arena_file = ''
        self.replayer = ReplayActions(replay_ratio)

    def step(self, action):
        action_ =self.replayer.replay_step(action)
        obs, reward, done, info = self.env.step(action_)
        self.steps += 1
        self.env_reward += reward
        info['arena']=self._arena_file  #for monitor
        info['max_reward']=self.max_reward
        info['max_time']=self.max_time
        info['ereward'] = self.env_reward
        info['action'] = action_ 
        return obs, reward, done, info        

    def reset(self, **kwargs):
        self.steps = 0
        self.env_reward = 0
        self._arena_file, arena = random.choice(self.env_list)
        self.replayer.reset(self._arena_file)
#        self.max_reward = analyze_arena(arena)
        self.max_reward = set_reward_arena(arena, force_new_size=False)
        self.max_time = arena.arenas[0].t
        return self.env.reset(arenas_configurations=arena,**kwargs)


class LabAnimalRecordAction(gym.Wrapper):
    def __init__(self, env, arenas_dir, replay_ratio, record_actions):
        gym.Wrapper.__init__(self, env)
        if os.path.isdir(arenas_dir):
            files = glob.glob("{}/*.yaml".format(arenas_dir))
        else:
            #assume is a pattern
            files = glob.glob(arenas_dir)
        
        self.env_list = [(f,ArenaConfig(f)) for f in files]
        self._arena_file = ''
        self.replayer = ReplayActions(replay_ratio)
        self.obs_rollouts = []
        self.rews_rollouts = []
        self.actions_rollouts = []
        self.directory = record_actions
        self.arena_num = 0

    def step(self, action):
        
        action_ =self.replayer.replay_step(action)
        obs, reward, done, info = self.env.step(action)
        self.obs_rollouts.append(obs)
        self.rews_rollouts.append(reward)
        self.actions_rollouts.append(action)
        self.steps += 1
        self.env_reward += reward
        info['arena']=self._arena_file  #for monitor
        info['max_reward']=self.max_reward
        info['max_time']=self.max_time
        info['ereward'] = self.env_reward
        info['action'] = action_ 
        return obs, reward, done, info        

    def reset(self, **kwargs):
        
        if (len (self.actions_rollouts) > 0) and (self.env_reward > 0) :
            arena_name =  self._arena_file.split('/')[-1].split('.')[0]
            self.filename = '{}/{}'.format( self.directory ,arena_name)

            print(os.path.exists('{}.npz'.format(self.filename)))
            print(self.filename)
            if not os.path.exists('{}.npz'.format(self.filename)):
            
                np.savez(self.filename,observations=np.array(self.obs_rollouts),
                            rewards=np.array(self.rews_rollouts),
                            actions=np.array(self.actions_rollouts))
            self.arena_num += 1

        #self._arena_file, arena = random.choice(self.env_list)
        self._arena_file, arena   = self.env_list[self.arena_num % len(self.env_list)]
        self.steps = 0
        self.env_reward = 0
        
        self.replayer.reset(self._arena_file)
#        self.max_reward = analyze_arena(arena)
        self.max_reward = set_reward_arena(arena, force_new_size=False)
        self.max_time = arena.arenas[0].t

        self.obs_rollouts = []
        self.rews_rollouts = []
        self.actions_rollouts = []

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

