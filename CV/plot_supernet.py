import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

def get_acc(path):
    if os.path.exists(path):
        f = open(path, 'r')
        content = f.readlines()
        train_lrs = []
        train_losses = []
        test_losses = []
        top_1_accs = []
        top_5_accs = []
        for line in content:
            train_lr = np.round(float(yaml.safe_load(line)["train_lr"]), 5)
            train_loss = yaml.safe_load(line)["train_loss"]
            test_loss = yaml.safe_load(line)["test_loss"]
            top_1_acc = yaml.safe_load(line)["test_acc1"]
            top_5_acc = yaml.safe_load(line)["test_acc5"]
            train_lrs.append(train_lr)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            top_1_accs.append(top_1_acc)
            top_5_accs.append(top_5_acc)
        f.close()

    return train_losses, test_losses, top_1_accs, top_5_accs, train_lrs

path_1 = 'searching_v1/work_dirs/cifar100/Supernet_v1-20210818-133106/log.txt'
path_2 = 'searching_v2/work_dirs/cifar100/Supernet_v2-20210820-220728/log.txt'
path_3 = 'searching_v3/work_dirs/cifar100/Supernet_v3-20210820-231347/log.txt'
path_4 = 'searching_v4/work_dirs/cifar100/Supernet_v4-20210826-193803/log.txt'
path_5 = 'searching_v5/work_dirs/cifar100/Supernet_v5-20210902-220001/log.txt'

train_losses_1, test_losses_1, top_1_accs_1, top_5_accs_1, train_lrs = get_acc(path_1)
train_losses_2, test_losses_2, top_1_accs_2, top_5_accs_2, train_lrs = get_acc(path_2)
train_losses_3, test_losses_3, top_1_accs_3, top_5_accs_3, train_lrs = get_acc(path_3)
train_losses_4, test_losses_4, top_1_accs_4, top_5_accs_4, train_lrs = get_acc(path_4)
train_losses_5, test_losses_5, top_1_accs_5, top_5_accs_5, train_lrs = get_acc(path_5)

lw = 3
font_big = 14
font_mid = 12
font_small = 10
font_board = 2

total_epochs = min(len(top_1_accs_1), len(top_1_accs_2), len(top_1_accs_3), len(top_1_accs_4))
axis = [i for i in range(1, total_epochs+1)]
line_type = [['r', 'g'], ['c', 'y'], ['violet', 'pink']]

fig, ax = plt.subplots(1,2,figsize=(9, 3.5))
plt.subplots_adjust(wspace=0.1, hspace=0.2)

ax[0].plot(axis[:total_epochs], train_losses_1[:total_epochs], line_type[0][0], label="Supernet v1", lw=lw)
ax[0].plot(axis[:total_epochs], train_losses_2[:total_epochs], line_type[0][1], label="Supernet v2", lw=lw)
ax[0].plot(axis[:total_epochs], train_losses_3[:total_epochs], line_type[1][0], label="Supernet v3", lw=lw)
ax[0].plot(axis[:total_epochs], train_losses_4[:total_epochs], line_type[1][1], label="Supernet v4", lw=lw)
ax[0].plot(axis[:total_epochs], train_losses_5[:total_epochs], line_type[2][0], label="Supernet v5", lw=lw)
leg = ax[0].legend(fontsize=font_mid)
leg.get_frame().set_edgecolor("black")
leg.get_frame().set_linewidth(1.5)
ax[0].tick_params(axis='both', which='major', labelsize=font_small)
ax[0].set_xlabel('Epochs', fontweight="bold", fontsize=font_big)
ax[0].set_ylabel('Train Loss', fontweight="bold", fontsize=font_big)
ax[0].grid(axis='both', color='gray', linestyle='-', linewidth=0.3)
ax[0].set_xlim([0, 300])

ax[1].plot(axis[:total_epochs], top_1_accs_1[:total_epochs], line_type[0][0], label="Supernet v1", lw=lw)
ax[1].plot(axis[:total_epochs], top_1_accs_2[:total_epochs], line_type[0][1], label="Supernet v2", lw=lw)
ax[1].plot(axis[:total_epochs], top_1_accs_3[:total_epochs], line_type[1][0], label="Supernet v3", lw=lw)
ax[1].plot(axis[:total_epochs], top_1_accs_4[:total_epochs], line_type[1][1], label="Supernet v4", lw=lw)
ax[1].plot(axis[:total_epochs], top_1_accs_5[:total_epochs], line_type[2][0], label="Supernet v5", lw=lw)
leg = ax[1].legend(fontsize=font_mid)
leg.get_frame().set_edgecolor("black")
leg.get_frame().set_linewidth(1.5)
ax[1].tick_params(axis='both', which='major', labelsize=font_small)
ax[1].set_xlabel('Epochs', fontweight="bold", fontsize=font_big)
ax[1].set_ylabel('Top-1 Acc. (%)', fontweight="bold", fontsize=font_big)
ax[1].grid(axis='both', color='gray', linestyle='-', linewidth=0.3)
ax[1].set_xlim([0, 300])

# ax[2].plot(axis[:total_epochs], train_lrs[:total_epochs], line_type[0][0], label="LR", lw=lw)
# leg = ax[2].legend(fontsize=font_mid)
# leg.get_frame().set_edgecolor("black")
# leg.get_frame().set_linewidth(1.5)
# ax[2].tick_params(axis='both', which='major', labelsize=font_small)
# ax[2].set_xlabel('Epochs', fontweight="bold", fontsize=font_big)
# ax[2].set_ylabel('Learning Rate', fontweight="bold", fontsize=font_big)
# ax[2].grid(axis='both', color='gray', linestyle='-', linewidth=0.3)
# ax[2].set_xlim([0, 300])

plt.tight_layout()
plt.savefig('figures/Supernet.pdf')
plt.savefig('figures/Supernet.png', dpi=300)