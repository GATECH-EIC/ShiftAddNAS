
# import argparse
# import datetime
# import json
# import numpy as np
# import os
# import time
# import torch
# import torch.backends.cudnn as cudnn
# from pathlib import Path
# from timm.data import Mixup
# from timm.data.distributed_sampler import OrderedDistributedSampler
# from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
# from timm.models import create_model
# from timm.optim import create_optimizer
# from timm.scheduler import create_scheduler
# from timm.utils import NativeScaler, get_state_dict, ModelEma

# import utils
# from bossnas.models.supernets.hytra_supernet import Supernet_v1
# from bossnas.models.supernets.hytra_supernet_ws import Supernet_v2
# from datasets import build_dataset
# from engine import train_one_epoch, evaluate
# from samplers import RASampler

# from matplotlib import pyplot as plt
# from bossnas.models.operations.adder.adder import Adder2D
# from bossnas.models.operations.deepshift.modules import Conv2dShift
# import torch.nn as nn



# plots histogram of weights
def plot_fig_weights(flat_weights, title_name):

    NUM_PLOT_BINS = 30
    font_board = 2

    ax = fig.add_subplot(1, 2, title_name)
    # ax.set_title("Pruning " + save_name[title_name-1] + '%')
    ax.set_title("")
    ax.hist(flat_weights, NUM_PLOT_BINS, color='green', alpha=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['bottom'].set_linewidth(font_board)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_linewidth(font_board)
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_linewidth(font_board)
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_linewidth(font_board)
    ax.spines['right'].set_color('black')


# plot laplace
from torch.distributions.laplace import Laplace
from matplotlib import pyplot as plt

m = Laplace(0, 4)
dist_1 = m.sample((3000,))
dist_2 = m.sample((7000,))

print(dist_1[:100])
print(max(dist_1), min(dist_1))

fig = plt.figure(figsize=(4.5,2.5))
plot_fig_weights(dist_1.reshape(1, -1), 1)
plot_fig_weights(dist_2.reshape(1, -1), 2)
plt.tight_layout()
plt.savefig('laplace.pdf')

# plot add
# fig = plt.figure(figsize=(9,2.5))
# layer_count = 0
# sub_fig_count = 0
# for name, m in model.named_modules():
#     # if isinstance(m, Adder2D):
#     if 'depth' in name:
#         shared_weight = m.shared_weight
#         for key, item in shared_weight.items():
#             weight = item.data.cpu().reshape(-1)
#             weight = weight[weight.nonzero()]
#             print(layer_count, '-th layer: ', weight.shape)
#             layer_count += 1
#             # plot
#             if layer_count >= 34 and layer_count < 38:
#                 sub_fig_count += 1
#                 plot_fig_weights(weight.reshape(1, -1), sub_fig_count)
# print('total adder layers: ', layer_count)
# plt.tight_layout()
# plt.savefig(args.output_dir+'/add.pdf')
