import torch
import random
import numpy as np
from matplotlib import pyplot as plt

from scipy import stats

path_cifar    = '../searching_v1/work_dirs/cifar100/Supernet_v1/arch_ranking.pth.tar'
path_cifar_ws = '../searching_v2/work_dirs/cifar100/Supernet_v2/arch_ranking.pth.tar'

info_cifar = torch.load(path_cifar)
info_cifar_ws = torch.load(path_cifar_ws)

max_length = min(len(info_cifar), len(info_cifar_ws))

cifar_params = []
cifar_train_acc_1 = []
cifar_train_acc_5 = []
cifar_test_acc_1 = []
cifar_test_acc_5 = []

cifar_ws_params = []
cifar_ws_train_acc_1 = []
cifar_ws_train_acc_5 = []
cifar_ws_test_acc_1 = []
cifar_ws_test_acc_5 = []

for i in range(max_length):
    _cifar_params = info_cifar[i]['params']
    _cifar_train_acc_1 = info_cifar[i]['train_acc_1']
    _cifar_train_acc_5 = info_cifar[i]['train_acc_5']
    _cifar_test_acc_1 = info_cifar[i]['test_acc_1']
    _cifar_test_acc_5 = info_cifar[i]['test_acc_5']

    _cifar_ws_params = info_cifar_ws[i]['params']
    _cifar_ws_train_acc_1 = info_cifar_ws[i]['train_acc_1']
    _cifar_ws_train_acc_5 = info_cifar_ws[i]['train_acc_5']
    _cifar_ws_test_acc_1 = info_cifar_ws[i]['test_acc_1']
    _cifar_ws_test_acc_5 = info_cifar_ws[i]['test_acc_5']

    cifar_params.append(_cifar_params)
    cifar_train_acc_1.append(_cifar_train_acc_1)
    cifar_train_acc_5.append(_cifar_train_acc_5)
    cifar_test_acc_1.append(_cifar_test_acc_1)
    cifar_test_acc_5.append(_cifar_test_acc_5)

    cifar_ws_params.append(_cifar_ws_params)
    cifar_ws_train_acc_1.append(_cifar_ws_train_acc_1)
    cifar_ws_train_acc_5.append(_cifar_ws_train_acc_5)
    cifar_ws_test_acc_1.append(_cifar_ws_test_acc_1)
    cifar_ws_test_acc_5.append(_cifar_ws_test_acc_5)


print('---'*10)
print('Ranking correlation in terms of top-1 training accuracy: ')
print('kendalltau: ', stats.kendalltau(cifar_train_acc_1, cifar_ws_train_acc_1))
print('pearsonr: ', stats.pearsonr(cifar_train_acc_1, cifar_ws_train_acc_1))
print('spearmanr: ', stats.spearmanr(cifar_train_acc_1, cifar_ws_train_acc_1))
print('---'*10)

print('---'*10)
print('Ranking correlation in terms of top-1 testing accuracy: ')
print('kendalltau: ', stats.kendalltau(cifar_test_acc_1, cifar_ws_test_acc_1))
print('pearsonr: ', stats.pearsonr(cifar_test_acc_1, cifar_ws_test_acc_1))
print('spearmanr: ', stats.spearmanr(cifar_test_acc_1, cifar_ws_test_acc_1))
print('---'*10)


lw = 3
font_big = 14
font_mid = 12
font_small = 10
font_board = 2

axis = [i for i in range(1, max_length+1)]
line_type = [['r', 'g'], ['c', 'y'], ['violet', 'pink']]

fig, ax = plt.subplots(1,3,figsize=(15, 3.5))

p1 = np.sort(cifar_test_acc_1)
p2 = np.sort(cifar_ws_test_acc_1)

ax[0].scatter(axis, p1, s=100, c='red', alpha=0.5, marker='o', label='CIFAR w/o sharing')
ax[0].scatter(axis, p2, s=100, c='blue', alpha=0.5, marker='o', label='CIFAR w/ sharing')
ax[0].tick_params(axis='both', which='major', labelsize=font_small)
ax[0].set_xlabel('No. Arch.', fontweight="bold", fontsize=font_big)
ax[0].set_ylabel('Top-1 Accuracy (%)', fontweight="bold", fontsize=font_big)
ax[0].grid(axis='x', color='gray', linestyle='-', linewidth=0.3)
ax[0].legend(fontsize=font_big, loc='lower right')

ax[1].scatter(cifar_train_acc_1, cifar_ws_train_acc_1, s=100, c='red', alpha=0.5, marker='o', label="Acc. Corr.")
leg = ax[1].legend(fontsize=font_mid)
leg.get_frame().set_edgecolor("black")
leg.get_frame().set_linewidth(1.5)
ax[1].tick_params(axis='both', which='major', labelsize=font_small)
ax[1].set_xlabel('CIFAR w/o sharing', fontweight="bold", fontsize=font_big)
ax[1].set_ylabel('CIFAR w/ sharing', fontweight="bold", fontsize=font_big)
ax[1].grid(axis='both', color='gray', linestyle='-', linewidth=0.3)
ax[1].set_yscale('log')
# ax[1].set_xscale('log')

ax[2].scatter(cifar_test_acc_1, cifar_ws_test_acc_1, s=100, c='blue', alpha=0.5, marker='o', label="Acc. Corr.")
leg = ax[2].legend(fontsize=font_mid)
leg.get_frame().set_edgecolor("black")
leg.get_frame().set_linewidth(1.5)
ax[2].tick_params(axis='both', which='major', labelsize=font_small)
ax[2].set_xlabel('CIFAR w/o sharing', fontweight="bold", fontsize=font_big)
ax[2].set_ylabel('CIFAR w/ sharing', fontweight="bold", fontsize=font_big)
ax[2].grid(axis='both', color='gray', linestyle='-', linewidth=0.3)
ax[2].set_yscale('log')
# ax[2].set_xscale('log')

plt.tight_layout()
plt.savefig('./work_dirs/corr_kl.pdf')
plt.savefig('./work_dirs/corr_kl.svg')
