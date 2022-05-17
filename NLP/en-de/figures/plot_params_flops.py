# -*- coding: utf-8 -*-
"""
@author : Haoran You

"""
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import os, sys
import numpy as np

bleus_searched  = [27.00, 25.92, 26.05, 18.05, 26.27, 26.96, 27.09, 27.95, 27.86, 27.84,
27.73 ,28, 27.75,
26.65, 27.29, 27.35, 27.69, 28.13]
bleus_attention = [27.92, 27.81, 27.46, 27.56, 27.11]
bleus_conv = [27.05, 26.47]

params_searched  = [44270304, 35272172, 30212004, 34141760, 23911604, 42435436, 43217044, 135436632, 89284188, 81668956,
36974940, 39008236, 33103536,
25876768, 30998848, 29949136, 34606180,
42549968]
params_attention = [48336896, 39928832, 34675200, 38479872, 27322368]
params_conv = [41639432, 29747796]

flops_searched  = [2878702302, 2265012678, 1922753454, 2139794088, 1545173934, 2798388702, 2809463244, 8538347742, 5699251398, 5109719226,
2436802752, 2558699712, 2138204352,
1631385768, 2008535220, 1945620660, 2224804020,
2771036352]
flops_attention = [3128586468, 2550989004, 2199275712, 2409868992, 1755033780]
flops_conv = [2712046272, 1891011996]

transformer_params = [176e6, 44e6, 22095360]
transformer_flops = [106e8, 268e7, 1355428068]
transformer_bleus = [28.4, 27.3, 24.7]

hat_params = [49380864, 51204480, 40972800, 35719168, 30465536]
hat_flops = [3087759576, 3172784832, 2543462592, 2191749300, 1873336488]
hat_bleus = [28.5, 28.1, 27.9, 27.6, 25.8]

lw = 3
msize = 5
font_big = 14
font_mid = 12
font_small = 10
font_board = 1.5

def remove_ith(bleus_searched, bleus_attention,
			   params_searched, params_attention,
			   flops_searched, flops_attention,
			   ith):
	bleus_searched = bleus_searched[:ith] + bleus_searched[ith+1:]
	bleus_attention = bleus_attention[:ith] + bleus_attention[ith+1:]

	params_searched = params_searched[:ith] + params_searched[ith+1:]
	params_attention = params_attention[:ith] + params_attention[ith+1:]

	flops_searched = flops_searched[:ith] + flops_searched[ith+1:]
	flops_attention = flops_attention[:ith] + flops_attention[ith+1:]

	return bleus_searched, bleus_attention, params_searched, params_attention, flops_searched, flops_attention

bleus_searched, bleus_attention, params_searched, params_attention, flops_searched, flops_attention = remove_ith(
	bleus_searched, bleus_attention,
	params_searched, params_attention,
	flops_searched, flops_attention,
	ith=3)

fig, ax = plt.subplots(1,2,figsize=(10, 4))

ax[0].scatter(transformer_params, transformer_bleus, s=100, c='purple', alpha=0.5, marker='*', label='transformer')
ax[0].scatter(hat_params, hat_bleus, s=100, c='g', alpha=0.5, marker='^', label='HAT')
ax[0].scatter(params_searched, bleus_searched, s=50, c='b', alpha=0.5, label='searched')
# ax[0].scatter(params_attention, bleus_attention, s=50, c='r', alpha=0.5, label='all attention')
ax[0].scatter(params_conv, bleus_conv, s=50, c='cyan', alpha=0.5, label='all conv')
ax[0].tick_params(axis='both', which='major', labelsize=font_small)
ax[0].set_xlabel('Num. of Params', fontweight="bold", fontsize=font_big)
ax[0].set_ylabel('BLEUs Score', fontweight="bold", fontsize=font_big)
ax[0].grid(axis='x', color='gray', linestyle='-', linewidth=0.3)
ax[0].legend(fontsize=font_big, loc='lower right')
# ax[0].set_yscale('log')

ax[1].scatter(transformer_flops, transformer_bleus, s=100, c='purple', alpha=0.5, marker='*', label='transformer')
ax[1].scatter(hat_flops, hat_bleus, s=100, c='g', alpha=0.5, marker='^', label='HAT')
ax[1].scatter(flops_searched, bleus_searched, s=50, c='b', alpha=0.5, label='searched')
# ax[1].scatter(flops_attention, bleus_attention, s=50, c='r', alpha=0.5, label='all attention')
ax[1].scatter(flops_conv, bleus_conv, s=50, c='cyan', alpha=0.5, label='all conv')
ax[1].tick_params(axis='both', which='major', labelsize=font_small)
ax[1].set_xlabel('FLOPs', fontweight="bold", fontsize=font_big)
ax[1].set_ylabel('BLEUs Score', fontweight="bold", fontsize=font_big)
ax[1].grid(axis='x', color='gray', linestyle='-', linewidth=0.3)
ax[1].legend(fontsize=font_big, loc='lower right')
# ax[1].set_yscale('log')
ax[1].set_xlim([1e9, 4e9])

for i in range(2):
	ax[i].spines['bottom'].set_linewidth(font_board)
	ax[i].spines['bottom'].set_color('black')
	ax[i].spines['left'].set_linewidth(font_board)
	ax[i].spines['left'].set_color('black')
	ax[i].spines['top'].set_linewidth(font_board)
	ax[i].spines['top'].set_color('black')
	ax[i].spines['right'].set_linewidth(font_board)
	ax[i].spines['right'].set_color('black')

plt.tight_layout()

plt.savefig('BLEUs.pdf')
plt.savefig('BLEUs.png')