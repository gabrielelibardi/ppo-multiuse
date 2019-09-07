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

exps = glob('exp_*')

while True:
    for i,d in enumerate(exps):
        fig = plt.figure(i,clear=True, figsize=(15,9))
        try:
            df = load_results(d)
            df['f']= df['l'].cumsum()/1000000
            roll = 5 
            rdf = df.rolling(roll)
            total_time = df['t'].iloc[-1]
            total_steps = df['l'].sum()
            total_episodes = df['r'].size
            
            ax = plt.subplot(2, 2, 1)
            ax.set_title(' {} total time: {:.1f} h FPS {:.1f}'.format(d.upper(),total_time/3600, total_steps/total_time))

            rdf.max().iloc[0:-1:40].plot('f','floor', style='-',ax=ax,legend=False)
            rdf.min().iloc[0:-1:40].plot('f','floor', style='-',ax=ax,legend=False)
            df.rolling(50*roll).mean().iloc[0:-1:40].plot('f','floor', style='-', ax=ax,legend=False)
            ax.yaxis.set_major_locator(plt.MultipleLocator(1))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(2))
            ax.set_xlabel('N. steps (M)')
            ax.set_ylabel('Floor')
            ax.grid(True)

            ax = plt.subplot(2, 2, 2)
            rdf.max().iloc[0:-1:40].plot('f','r', ax=ax,legend=False)
            rdf.min().iloc[0:-1:40].plot('f','r', ax=ax,legend=False)
            df.rolling(50*roll).mean().iloc[0:-1:40].plot('f','r', ax=ax,legend=False)
            ax.set_xlabel('N. steps (M)')
            ax.set_ylabel('Reward')
            ax.grid(True)
            
            ax = plt.subplot(2, 2, 3)
            ax.set_title('Goal mean: {:.5f}'.format(df['goal'].iloc[-5000:].mean()))
            rdf.max().iloc[0:-1:40].plot('f','goal', style='-',ax=ax,legend=False)
            rdf.min().iloc[0:-1:40].plot('f','goal', style='-',ax=ax,legend=False)
            df.rolling(50*roll).mean().iloc[0:-1:40].plot('f','goal', style='-', ax=ax,legend=False)
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(2))
            ax.set_xlabel('N. steps (M)')
            ax.set_ylabel('Goal')
            ax.grid(True)

            ax = plt.subplot(2, 2, 4)
            rdf.max().iloc[0:-1:40].plot(y='l', ax = ax,legend=False)
            rdf.min().iloc[0:-1:40].plot(y='l', ax = ax,legend=False)
            df.rolling(50*roll).mean().iloc[0:-1:40].plot(y='l', ax = ax,legend=False)
            ax.set_xlabel('N. episodes')
            ax.set_ylabel('N. steps')
            ax.grid(True)

            fig.tight_layout() 
            ax.get_figure().savefig('/webdata/'+d+'.jpg')
        except:
            print('Failed at ',d,flush=True)
        
    sleep(360)



quit()



