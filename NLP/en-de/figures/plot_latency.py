# -*- coding: utf-8 -*-
"""
@author : Haoran You

"""
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import os, sys
import numpy as np

transformer_bleus = [28.4, 27.3, 24.7]
transformer_latency = [130.25, 41.82, 21.18]

hat_bleus = [25.8, 26.9, 27.6, 27.8, 28.2, 28.4]
hat_loss = [4.605, 4.299, 4.161, 4.053, 3.954, 3.917]
hat_ppl = [6.96, 5.61, 5.2, 4.89, 4.61, 4.52]
hat_latency = [23.83, 27.82, 33.72, 34.76, 41.70, 45.70]

conv_bleus = [27.05, 26.47]
conv_loss = [3.935, 4.107]
conv_ppl = [4.6, 5.11]
conv_latency = [53.59, 35.10]

nas_bleus = [28.17, 28.46, 27.73, 28, 27.75, 28.13, 26.65, 27.29, 27.35, 27.69]
nas_loss = [3.902, 3.847, 3.974, 3.975, 3.997, 3.964, 4.501, 4.106, 4.103, 4.087]
nas_ppl = [4.46, 4.32, 4.72, 4.72, 4.78, 4.69, 6.78, 5.13, 5.11, 5.08]
nas_latency = [60.70, 68.83, 37.73, 40.06, 33.41, 59.50, 25.17, 22.84, 30.22, 34.51]

nas_quant_bleus = [27.84, 28.25]
nas_quant_loss = [3.93, 3.887]
nas_quant_ppl = [4.54, 4.42]
nas_quant_latency = [15.20, 17.23]

manual_bleus = [28]
manual_loss = [3.957]
manual_ppl = [4.61]
manual_latency = [48.45]

lw = 3
msize = 5
font_big = 14
font_mid = 12
font_small = 10
font_board = 1.5
font_legend = 8

fig, ax = plt.subplots(1,3,figsize=(10, 3))

ax[0].scatter(transformer_latency, transformer_bleus, s=60, c='purple', alpha=0.5, marker='*', label='Transformer')
ax[0].scatter(hat_latency, hat_bleus, s=60, c='g', alpha=0.5, marker='^', label='HAT')
ax[0].scatter(conv_latency, conv_bleus, s=50, c='orange', alpha=0.5, label='Lightweight Conv')
ax[0].scatter(nas_latency, nas_bleus, s=50, c='b', alpha=0.5, label='Searched')
ax[0].scatter(nas_quant_latency, nas_quant_bleus, s=50, c='r', alpha=0.5, label='Searched (Quant)')
ax[0].scatter(manual_latency, manual_bleus, s=50, c='cyan', alpha=0.5, label='Manual')
ax[0].tick_params(axis='both', which='major', labelsize=font_small)
ax[0].set_xlabel('Latency (ms)', fontweight="bold", fontsize=font_big)
ax[0].set_ylabel('BLEUs Score', fontweight="bold", fontsize=font_big)
ax[0].grid(axis='x', color='gray', linestyle='-', linewidth=0.3)
ax[0].legend(fontsize=font_legend, loc='best')
# ax[0].set_yscale('log')

# ax[1].scatter(transformer_flops, transformer_bleus, s=100, c='purple', alpha=0.5, marker='*', label='transformer')
ax[1].scatter(hat_latency, hat_loss, s=60, c='g', alpha=0.5, marker='^', label='HAT')
ax[1].scatter(conv_latency, conv_loss, s=50, c='orange', alpha=0.5, label='Lightweight Conv')
ax[1].scatter(nas_latency, nas_loss, s=50, c='b', alpha=0.5, label='Searched')
ax[1].scatter(nas_quant_latency, nas_quant_loss, s=50, c='r', alpha=0.5, label='Searched (Quant)')
ax[1].scatter(manual_latency, manual_loss, s=50, c='cyan', alpha=0.5, label='Manual')
ax[1].tick_params(axis='both', which='major', labelsize=font_small)
ax[1].set_xlabel('Latency (ms)', fontweight="bold", fontsize=font_big)
ax[1].set_ylabel('Validation Loss', fontweight="bold", fontsize=font_big)
ax[1].grid(axis='x', color='gray', linestyle='-', linewidth=0.3)
ax[1].legend(fontsize=font_legend, loc='best')
# ax[1].set_yscale('log')

# ax[2].scatter(transformer_flops, transformer_bleus, s=100, c='purple', alpha=0.5, marker='*', label='transformer')
ax[2].scatter(hat_latency, hat_ppl, s=60, c='g', alpha=0.5, marker='^', label='HAT')
ax[2].scatter(conv_latency, conv_ppl, s=50, c='orange', alpha=0.5, label='Lightweight Conv')
ax[2].scatter(nas_latency, nas_ppl, s=50, c='b', alpha=0.5, label='Searched')
ax[2].scatter(nas_quant_latency, nas_quant_ppl, s=50, c='r', alpha=0.5, label='Searched (Quant)')
ax[2].scatter(manual_latency, manual_ppl, s=50, c='cyan', alpha=0.5, label='Manual')
ax[2].tick_params(axis='both', which='major', labelsize=font_small)
ax[2].set_xlabel('Latency (ms)', fontweight="bold", fontsize=font_big)
ax[2].set_ylabel('Validation PPL', fontweight="bold", fontsize=font_big)
ax[2].grid(axis='x', color='gray', linestyle='-', linewidth=0.3)
ax[2].legend(fontsize=font_legend, loc='best')
# ax[2].set_yscale('log')

for i in range(3):
	ax[i].spines['bottom'].set_linewidth(font_board)
	ax[i].spines['bottom'].set_color('black')
	ax[i].spines['left'].set_linewidth(font_board)
	ax[i].spines['left'].set_color('black')
	ax[i].spines['top'].set_linewidth(font_board)
	ax[i].spines['top'].set_color('black')
	ax[i].spines['right'].set_linewidth(font_board)
	ax[i].spines['right'].set_color('black')

plt.tight_layout()

plt.savefig('latency.pdf')
plt.savefig('latency.png')