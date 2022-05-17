import matplotlib.pyplot as plt
import os, sys
import numpy as np

path_120ms = "./checkpoints/wmt14.en-de/subtransformer/wmt14ende_gpu_V100_shiftadd_v3@120ms_update/"
path_150ms = "./checkpoints/wmt14.en-de/subtransformer/wmt14ende_gpu_V100_shiftadd_v3@150ms_update/"
path_180ms = "./checkpoints/wmt14.en-de/subtransformer/wmt14ende_gpu_V100_shiftadd_v3@180ms_update/"
path_200ms = "./checkpoints/wmt14.en-de/subtransformer/wmt14ende_gpu_V100_shiftadd_v3@200ms_update/"
path_250ms = "./checkpoints/wmt14.en-de/subtransformer/wmt14ende_gpu_V100_shiftadd_v3@250ms_update/"
path_300ms = "./checkpoints/wmt14.en-de/subtransformer/wmt14ende_gpu_V100_shiftadd_v3@300ms_update/"
path_350ms = "./checkpoints/wmt14.en-de/subtransformer/wmt14ende_gpu_V100_shiftadd_v3@350ms_update/"
# path_350ms = "./checkpoints/wmt14.en-de/subtransformer/wmt14ende_gpu_V100_shiftadd_v3@350ms_update/"

line_120ms = path_120ms + 'record.txt'
line_150ms = path_150ms + 'record.txt'
line_180ms = path_180ms + 'record.txt'
line_200ms = path_200ms + 'record.txt'
line_250ms = path_250ms + 'record.txt'
line_300ms = path_300ms + 'record.txt'
line_350ms = path_350ms + 'record.txt'

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

losses_120, ppl_120, bleus_120 = get_loss_ppl_bleus(line_120ms)
losses_150, ppl_150, bleus_150 = get_loss_ppl_bleus(line_150ms)
losses_180, ppl_180, bleus_180 = get_loss_ppl_bleus(line_180ms)
losses_200, ppl_200, bleus_200 = get_loss_ppl_bleus(line_200ms)
losses_250, ppl_250, bleus_250 = get_loss_ppl_bleus(line_250ms)
losses_300, ppl_300, bleus_300 = get_loss_ppl_bleus(line_300ms)
losses_350, ppl_350, bleus_350 = get_loss_ppl_bleus(line_350ms)

lw = 3
font_big = 14
font_mid = 12
font_small = 10
font_board = 2

total_epochs = 140
axis = [i for i in range(1, total_epochs+1)]
line_type = [['r', 'g'], ['c', 'y'], ['violet', 'pink'], ['b']]

fig, ax = plt.subplots(1,3,figsize=(12, 3.5))

ax[0].plot(axis[:total_epochs], losses_120[:total_epochs], line_type[0][0], label="120ms", lw=lw)
ax[0].plot(axis[:total_epochs], losses_150[:total_epochs], line_type[0][1], label="150ms", lw=lw)
ax[0].plot(axis[:total_epochs], losses_180[:total_epochs], line_type[1][0], label="180ms", lw=lw)
ax[0].plot(axis[:total_epochs], losses_200[:total_epochs], line_type[1][1], label="200ms", lw=lw)
ax[0].plot(axis[:total_epochs], losses_250[:total_epochs], line_type[2][0], label="250ms", lw=lw)
ax[0].plot(axis[:total_epochs], losses_300[:total_epochs], line_type[2][1], label="300ms", lw=lw)
ax[0].plot(axis[:total_epochs], losses_350[:total_epochs], line_type[3][0], label="350ms", lw=lw)
leg = ax[0].legend(fontsize=font_mid)
leg.get_frame().set_edgecolor("black")
leg.get_frame().set_linewidth(1.5)
ax[0].tick_params(axis='both', which='major', labelsize=font_small)
ax[0].set_xlabel('Epochs', fontweight="bold", fontsize=font_big)
ax[0].set_ylabel('Training Losses', fontweight="bold", fontsize=font_big)
ax[0].grid(axis='both', color='gray', linestyle='-', linewidth=0.3)
# ax[0].set_yscale('log')

ax[1].plot(axis[:total_epochs], ppl_120[:total_epochs], line_type[0][0], label="120ms", lw=lw)
ax[1].plot(axis[:total_epochs], ppl_150[:total_epochs], line_type[0][1], label="150ms", lw=lw)
ax[1].plot(axis[:total_epochs], ppl_180[:total_epochs], line_type[1][0], label="180ms", lw=lw)
ax[1].plot(axis[:total_epochs], ppl_200[:total_epochs], line_type[1][1], label="200ms", lw=lw)
ax[1].plot(axis[:total_epochs], ppl_250[:total_epochs], line_type[2][0], label="250ms", lw=lw)
ax[1].plot(axis[:total_epochs], ppl_300[:total_epochs], line_type[2][1], label="300ms", lw=lw)
ax[1].plot(axis[:total_epochs], ppl_350[:total_epochs], line_type[3][0], label="350ms", lw=lw)
leg = ax[1].legend(fontsize=font_mid)
leg.get_frame().set_edgecolor("black")
leg.get_frame().set_linewidth(1.5)
ax[1].tick_params(axis='both', which='major', labelsize=font_small)
ax[1].set_xlabel('Epochs', fontweight="bold", fontsize=font_big)
ax[1].set_ylabel('Training PPL', fontweight="bold", fontsize=font_big)
ax[1].grid(axis='both', color='gray', linestyle='-', linewidth=0.3)
ax[1].set_yscale('log')
# ax[1].set_xscale('log')

ax[2].plot(axis[:total_epochs], bleus_120[:total_epochs], line_type[0][0], label="120ms", lw=lw)
ax[2].plot(axis[:total_epochs], bleus_150[:total_epochs], line_type[0][1], label="150ms", lw=lw)
ax[2].plot(axis[:total_epochs], bleus_180[:total_epochs], line_type[1][0], label="180ms", lw=lw)
ax[2].plot(axis[:total_epochs], bleus_200[:total_epochs], line_type[1][1], label="200ms", lw=lw)
ax[2].plot(axis[:total_epochs], bleus_250[:total_epochs], line_type[2][0], label="250ms", lw=lw)
ax[2].plot(axis[:total_epochs], bleus_300[:total_epochs], line_type[2][1], label="300ms", lw=lw)
ax[2].plot(axis[:total_epochs], bleus_350[:total_epochs], line_type[3][0], label="350ms", lw=lw)
leg = ax[2].legend(fontsize=font_mid, loc='upper left')
leg.get_frame().set_edgecolor("black")
leg.get_frame().set_linewidth(1.5)
# ax[2].set_yticks((10, 20, 25, 30))
ax[2].tick_params(axis='both', which='major', labelsize=font_small)
ax[2].set_xlabel('Epochs', fontweight="bold", fontsize=font_big)
ax[2].set_ylabel('Testing BLEUs', fontweight="bold", fontsize=font_big)
ax[2].grid(axis='both', color='gray', linestyle='-', linewidth=0.3)
# ax[2].set_xscale('log')
ax[2].set_ylim([23, 29])

plt.tight_layout()
plt.savefig('./figures/trajectory_comp.pdf')
plt.savefig('./figures/trajectory_comp.png')