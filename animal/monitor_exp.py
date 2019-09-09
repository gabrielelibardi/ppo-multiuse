#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:51:35 2019

@author: gianni
"""

from  glob import glob
from time import sleep
from baselines.bench import load_results
from matplotlib import pylab as plt
import numpy as np

exps = glob('RUNS/exp_2*')
print(exps)

while True:
    for i,d in enumerate(exps):
        fig = plt.figure(i,clear=True, figsize=(15,9))
        try:
            df = load_results(d)
            df['f']= df['l'].cumsum()/1000000
            df['perf']= df['r']/(2+df['max_reward'])#correct for reshaping, but I should really have the real ones in info
            df['perf'].where(df['perf']>0,0,inplace=True)
            df['goal'] = df['perf']>0.9  #guess a threadshold
            print(df['goal'].iloc[0:100])

            roll = 5 
            total_time = df['t'].iloc[-1]
            total_steps = df['l'].sum()
            total_episodes = df['r'].size
            
            ax = plt.subplot(2, 2, 1)
            ax.set_title(' {} total time: {:.1f} h FPS {:.1f}'.format(d.upper(),total_time/3600, total_steps/total_time))
            df[['f','r']].rolling(50*roll).mean().iloc[0:-1:40].plot('f','r',  ax=ax,legend=False)
            ax.set_xlabel('N. steps (M)')
            ax.set_ylabel('Reward')
            ax.grid(True)
    
            ax = plt.subplot(2, 2, 2)
            df[['f','perf']].rolling(50*roll).mean().iloc[0:-1:40].plot('f','perf', ax=ax,legend=False)
            ax.set_xlabel('N. steps (M)')
            ax.set_ylabel('Performance')
            ax.grid(True)

            ax = plt.subplot(2, 2, 3)
            df[['f','goal']].rolling(50*roll).mean().iloc[0:-1:40].plot('f','goal', ax=ax,legend=False)
            ax.set_xlabel('N. steps (M)')
            ax.set_ylabel('Performance')
            ax.grid(True)
             
            fig.tight_layout() 
            ax.get_figure().savefig('/webdata/'+d+'.jpg')
        except:
            print('Failed at ',d,flush=True)
        
    sleep(360)



quit()



