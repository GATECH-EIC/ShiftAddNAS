# -*- coding: utf-8 -*-
"""
@author : Haoran You

"""
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import os, sys
import numpy as np

pred = np.load('./figures/pred.npy')
real = np.load('./figures/real.npy')

lw = 3
font_big = 14
font_mid = 12
font_small = 10
font_board = 1.5

fig, ax = plt.subplots(1,1,figsize=(5, 4))

ax.scatter(real, pred, s=50, c='b', alpha=0.5)
ax.tick_params(axis='both', which='major', labelsize=font_small)
ax.set_xlabel('Real Latency (s)', fontweight="bold", fontsize=font_big)
ax.set_ylabel('Predicted Latency (s)', fontweight="bold", fontsize=font_big)
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

plt.savefig('./figures/pred_vs_real.pdf')