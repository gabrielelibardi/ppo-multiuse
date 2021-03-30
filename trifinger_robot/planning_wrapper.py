import os
import gym
import sys
import numpy as np
import random
import torch
import glob
import copy


class MPC_planner(gym.Wrapper):
    def __init__(self, env): 
        gym.Wrapper.__init__(self, env)
        """self.state_to_reset = {'robot_position': np.array([-8.55992742e-03,  1.21459918e+00, -2.25559440e+00,  5.46024435e-04,
        1.21814735e+00, -2.26098951e+00,  5.46024435e-04,  1.21814735e+00,
       -2.26098951e+00]), 'robot_velocity': np.array([-2.13998186, -1.78282227,  3.33341258,  0.13650611, -0.89577924,
        1.9846363 ,  0.13650611, -0.89577924,  1.9846363 ]), 'robot_tip_positions': np.array([[ 0.08574786,  0.06014798,  0.15276312],
       [ 0.00988153, -0.10327163,  0.15277971],
       [-0.09437662,  0.04307816,  0.15277971]]), 'object_position': np.array([0.08385101, 0.00214682, 0.03254566]), 'object_orientation': np.array([ 4.70624660e-04, -5.91465649e-04,  5.06674167e-01,  8.62137296e-01]), 'goal_object_position': np.array([ 0.08309543, -0.09184842,  0.08837619])}
        """  
        self.init_state = self.env.reset()
        self.state_to_reset = self.init_state
        self.H = 5
        self.step_count = 0
        self.n_traj_max = 30
        self.n_traj = 0
        self.best_sum_rewards = -9999999999999999.0
        self.best_action = None
        self.sum_rewards = 0
        self.take_action = False
        self.real_steps = 0

    def step(self, action):

        self.step_count += 1
        # print('STEP N.',self.step_count)
        # print('TRAJECTORY N.',self.n_traj )
        if self.take_action:
            # print('TAKE REAL ACTION----------------------------------------------------------------------------------------')
            action = self.best_action # when all the trajectories have been taken

            print(self.real_steps)                          # algo takes best action so far. The state reached is saved as best state.
            self.real_steps += 1
        obs, reward, done, info = self.env.step(action)
        if self.take_action:
            print('REAL REWARD', reward)
            if self.real_steps >= 3000:
                self.state_to_reset = copy.deepcopy(self.init_state)
                self.real_steps = 0
            else:
                self.state_to_reset = copy.deepcopy(obs)
            #print('---------------------------------------------------------------------------------------------------')
            #print(self.state_to_reset)
            self.take_action = False
            self.step_count = 0
            self.n_traj = 0

        if self.step_count == 1:
            self.first_action = action
            
    
        self.sum_rewards += reward
        
        if self.step_count >= self.H:
            done = True
            self.n_traj += 1
            if self.sum_rewards >= self.best_sum_rewards:
                self.best_sum_rewards = self.sum_rewards
                self.best_action = self.first_action
            
            if self.n_traj >= self.n_traj_max:
                self.take_action = True
                #print('END PLANNING-------------------------------------------------------------------')
                self.n_traj = 0
        

        return obs, reward, done, info

    def reset(self, **kwargs):
        self.step_count = 0
        self.sum_rewards = 0
        #print('-----------------------RESETTING-------------------')
        obbs = self.env.unwrapped.reset(self.state_to_reset)
        #print(obbs)
        return obbs
    

