# -*- coding: utf-8 -*-
"""
@author : Haoran You

"""
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import os, sys
import numpy as np

datapoints = 2000

loss = np.load('./figures/loss_{}.npy'.format(datapoints), allow_pickle=True)
latency = np.load('./figures/latency_{}.npy'.format(datapoints), allow_pickle=True)
config = np.load('./figures/config_{}.npy'.format(datapoints), allow_pickle=True)

sorted_ind = np.array(loss).argsort()
print(f"| config for lowest  loss ({loss[sorted_ind[0]]}) model: \n {config[sorted_ind[0]]} \n")
print(f"| config for highest loss ({loss[sorted_ind[-1]]}) model: \n {config[sorted_ind[-1]]} \n")

sorted_ind = np.array(latency).argsort()
print(f"| config for lowest  latency ({latency[sorted_ind[0]]}) model: \n {config[sorted_ind[0]]} \n")
print(f"| config for highest latency ({latency[sorted_ind[-1]]}) model: \n  {config[sorted_ind[-1]]} \n")

lw = 3
font_big = 14
font_mid = 12
font_small = 10
font_board = 1.5

fig, ax = plt.subplots(1,1,figsize=(5, 4))

ax.scatter(latency, loss, s=50, c='b', alpha=0.5)
ax.tick_params(axis='both', which='major', labelsize=font_small)
ax.set_xlabel('Latency (ms)', fontweight="bold", fontsize=font_big)
ax.set_ylabel('Valid. Loss', fontweight="bold", fontsize=font_big)
ax.grid(axis='both', color='gray', linestyle='-', linewidth=0.3)

ax.spines['bottom'].set_linewidth(font_board)
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_linewidth(font_board)
ax.spines['left'].set_color('black')
ax.spines['top'].set_linewidth(font_board)
ax.spines['top'].set_color('black')
ax.spines['right'].set_linewidth(font_board)
ax.spines['right'].set_color('black')

plt.tight_layout()

plt.savefig('./figures/loss_vs_lat_{}.pdf'.format(datapoints))
plt.savefig('./figures/loss_vs_lat_{}.png'.format(datapoints))