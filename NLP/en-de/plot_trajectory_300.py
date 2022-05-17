import matplotlib.pyplot as plt
import os, sys
import numpy as np

path_300ms = "./checkpoints/wmt14.en-de/subtransformer/wmt14ende_gpu_V100_shiftadd_v3@300ms_Jun.29/"
path_300ms_quant = "./checkpoints/wmt14.en-de/subtransformer/wmt14ende_gpu_V100_shiftadd_v3@300ms_Jun.29_quant/"
path_300ms_quant_cyclic = "./checkpoints/wmt14.en-de/subtransformer/wmt14ende_gpu_V100_shiftadd_v3@300ms_Jun.29_quant_cyclic/"

line_300ms = path_300ms + 'record.txt'
line_300ms_quant = path_300ms_quant + 'record.txt'
line_300ms_quant_cyclic = path_300ms_quant_cyclic + 'record.txt'

def get_loss_ppl_bleus(baseline):
    with open(baseline, 'r') as log:
        content = log.readlines()
        losses = []
        train_ppl = []
        test_bleus = []
        for line in content:
            loss = float(line.split(',')[0][:-2])
            ppl = float(line.split(',')[1][:-2])
            bleus = float(line.split(',')[2][:-2])
            losses.append(loss)
            train_ppl.append(ppl)
            test_bleus.append(bleus)
        log.close()
    return losses, train_ppl, test_bleus

losses_300, ppl_300, bleus_300 = get_loss_ppl_bleus(line_300ms)
losses_300_quant, ppl_300_quant, bleus_300_quant = get_loss_ppl_bleus(line_300ms_quant)
losses_300_quant_cyclic, ppl_300_quant_cyclic, bleus_300_quant_cyclic = get_loss_ppl_bleus(line_300ms_quant_cyclic)

lw = 3
font_big = 14
font_mid = 12
font_small = 10
font_board = 2

total_epochs = 100
axis = [i for i in range(1, total_epochs+1)]
line_type = [['r', 'g'], ['c', 'y'], ['violet', 'pink']]

fig, ax = plt.subplots(1,3,figsize=(15, 3.5))

ax[0].plot(axis[:total_epochs], losses_300[:total_epochs], line_type[0][0], label="300ms", lw=lw)
ax[0].plot(axis[:total_epochs], losses_300_quant[:total_epochs], line_type[0][1], label="300ms FW-8/BW-8", lw=lw)
ax[0].plot(axis[:total_epochs], losses_300_quant_cyclic[:total_epochs], line_type[1][0], label="300ms CPT FW-(4,8)/BW-8", lw=lw)
leg = ax[0].legend(fontsize=font_mid)
leg.get_frame().set_edgecolor("black")
leg.get_frame().set_linewidth(1.5)
ax[0].tick_params(axis='both', which='major', labelsize=font_small)
ax[0].set_xlabel('Epochs', fontweight="bold", fontsize=font_big)
ax[0].set_ylabel('Training Losses', fontweight="bold", fontsize=font_big)
ax[0].grid(axis='both', color='gray', linestyle='-', linewidth=0.3)
# ax[0].set_xscale('log')

ax[1].plot(axis[:total_epochs], ppl_300[:total_epochs], line_type[0][0], label="300ms", lw=lw)
ax[1].plot(axis[:total_epochs], ppl_300_quant[:total_epochs], line_type[0][1], label="300ms FW-8/BW-8", lw=lw)
ax[1].plot(axis[:total_epochs], ppl_300_quant_cyclic[:total_epochs], line_type[1][0], label="300ms CPT FW-(4,8)/BW-8", lw=lw)
leg = ax[1].legend(fontsize=font_mid)
leg.get_frame().set_edgecolor("black")
leg.get_frame().set_linewidth(1.5)
ax[1].tick_params(axis='both', which='major', labelsize=font_small)
ax[1].set_xlabel('Epochs', fontweight="bold", fontsize=font_big)
ax[1].set_ylabel('Training PPL', fontweight="bold", fontsize=font_big)
ax[1].grid(axis='both', color='gray', linestyle='-', linewidth=0.3)
ax[1].set_yscale('log')
ax[1].set_xscale('log')

ax[2].plot(axis[:total_epochs], bleus_300[:total_epochs], line_type[0][0], label="300ms", lw=lw)
ax[2].plot(axis[:total_epochs], bleus_300_quant[:total_epochs], line_type[0][1], label="300ms FW-8/BW-8", lw=lw)
ax[2].plot(axis[:total_epochs], bleus_300_quant_cyclic[:total_epochs], line_type[1][0], label="300ms CPT FW-(4,8)/BW-8", lw=lw)
leg = ax[2].legend(fontsize=font_mid)
leg.get_frame().set_edgecolor("black")
leg.get_frame().set_linewidth(1.5)
ax[2].set_yticks((10, 20, 25, 30))
ax[2].tick_params(axis='both', which='major', labelsize=font_small)
ax[2].set_xlabel('Epochs', fontweight="bold", fontsize=font_big)
ax[2].set_ylabel('Testing BLEUs', fontweight="bold", fontsize=font_big)
ax[2].grid(axis='both', color='gray', linestyle='-', linewidth=0.3)
# ax[2].set_xscale('log')

plt.tight_layout()
plt.savefig('./figures/trajectory_300.pdf')
plt.savefig('./figures/trajectory_300.svg')