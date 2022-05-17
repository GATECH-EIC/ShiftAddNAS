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

path = 'log.txt'
train_losses, test_losses, top_1_accs, top_5_accs, train_lrs = get_acc(path)

lw = 3
font_big = 14
font_mid = 12
font_small = 10
font_board = 2

total_epochs = len(top_1_accs)
axis = [i for i in range(1, total_epochs+1)]
line_type = [['r', 'g'], ['c', 'y'], ['violet', 'pink']]

fig, ax = plt.subplots(1,2,figsize=(9, 3.5))
plt.subplots_adjust(wspace=0.1, hspace=0.2)

ax[0].plot(axis[:total_epochs], train_losses[:total_epochs], line_type[0][0], label="Train Loss", lw=lw)
ax[0].plot(axis[:total_epochs], test_losses[:total_epochs], line_type[0][1], label="Test Loss", lw=lw)
leg = ax[0].legend(fontsize=font_mid)
leg.get_frame().set_edgecolor("black")
leg.get_frame().set_linewidth(1.5)
ax[0].tick_params(axis='both', which='major', labelsize=font_small)
ax[0].set_xlabel('Epochs', fontweight="bold", fontsize=font_big)
ax[0].set_ylabel('Losses', fontweight="bold", fontsize=font_big)
ax[0].grid(axis='both', color='gray', linestyle='-', linewidth=0.3)
ax[0].set_xlim([0, 300])

ax[1].plot(axis[:total_epochs], top_1_accs[:total_epochs], line_type[0][0], label="Top-1 Acc.", lw=lw)
ax[1].plot(axis[:total_epochs], top_5_accs[:total_epochs], line_type[0][1], label="Top-5 Acc.", lw=lw)
leg = ax[1].legend(fontsize=font_mid)
leg.get_frame().set_edgecolor("black")
leg.get_frame().set_linewidth(1.5)
ax[1].tick_params(axis='both', which='major', labelsize=font_small)
ax[1].set_xlabel('Epochs', fontweight="bold", fontsize=font_big)
ax[1].set_ylabel('Test Accuracy (%)', fontweight="bold", fontsize=font_big)
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
plt.savefig('trajectory.pdf')
plt.savefig('trajectory.svg')