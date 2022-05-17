
import argparse
import datetime
import json
import numpy as np
import os
import time
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
from timm.data import Mixup
from timm.data.distributed_sampler import OrderedDistributedSampler
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models import create_model
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler, get_state_dict, ModelEma

import utils
from bossnas.models.supernets.hytra_supernet import Supernet_v1
from bossnas.models.supernets.hytra_supernet_ws import Supernet_v2
from bossnas.models.supernets.hytra_supernet_ws_dist import Supernet_v3
from datasets import build_dataset
from engine import train_one_epoch, evaluate
from samplers import RASampler

from matplotlib import pyplot as plt
from bossnas.models.operations.adder.adder import Adder2D
from bossnas.models.operations.deepshift.modules import Conv2dShift
from bossnas.models.operations.hytra_ops_ws_dist import ResAdd_ws_dist
import torch.nn as nn

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=False)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01_101/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--local_rank', default=0, type=int)
    return parser

def main(args):
    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True


    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
    )

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        sampler_val = OrderedDistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        shuffle=False, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)


    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    def my_load_state_dict(state_dict, names):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k not in names:
                print('loading {} ...'.format(k))
                new_state_dict[k] = v
                continue
            else:
                print('skipping {} ...'.format(k))
        return new_state_dict

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        # model_without_ddp.load_state_dict(checkpoint['model'])
        names = ["_blocks.0._block_layers.0._mix_ops.11.downsample.0.weight", "_blocks.0._block_layers.0._mix_ops.11.conv1.weight", "_blocks.0._block_layers.0._mix_ops.11.conv2.weight", "_blocks.0._block_layers.0._mix_ops.11.conv3.weight", "_blocks.0._block_layers.1._mix_ops.8.downsample_d.0.weight", "_blocks.0._block_layers.1._mix_ops.8.conv1_d.weight", "_blocks.0._block_layers.1._mix_ops.8.conv2.weight", "_blocks.0._block_layers.1._mix_ops.8.conv3.weight", "_blocks.0._block_layers.1._mix_ops.11.conv1.weight", "_blocks.0._block_layers.1._mix_ops.11.conv2.weight", "_blocks.0._block_layers.1._mix_ops.11.conv3.weight", "_blocks.0._block_layers.2._mix_ops.5.downsample_d.0.weight", "_blocks.0._block_layers.2._mix_ops.5.conv1_d.weight", "_blocks.0._block_layers.2._mix_ops.5.conv2.weight", "_blocks.0._block_layers.2._mix_ops.5.conv3.weight", "_blocks.0._block_layers.2._mix_ops.8.downsample_d.0.weight", "_blocks.0._block_layers.2._mix_ops.8.conv1_d.weight", "_blocks.0._block_layers.2._mix_ops.8.conv1.weight", "_blocks.0._block_layers.2._mix_ops.8.conv2.weight", "_blocks.0._block_layers.2._mix_ops.8.conv3.weight", "_blocks.0._block_layers.2._mix_ops.11.conv1.weight", "_blocks.0._block_layers.2._mix_ops.11.conv2.weight", "_blocks.0._block_layers.2._mix_ops.11.conv3.weight", "_blocks.0._block_layers.3._mix_ops.5.downsample_d.0.weight", "_blocks.0._block_layers.3._mix_ops.5.conv1_d.weight", "_blocks.0._block_layers.3._mix_ops.5.conv1.weight", "_blocks.0._block_layers.3._mix_ops.5.conv2.weight", "_blocks.0._block_layers.3._mix_ops.5.conv3.weight", "_blocks.0._block_layers.3._mix_ops.8.downsample_d.0.weight", "_blocks.0._block_layers.3._mix_ops.8.conv1_d.weight", "_blocks.0._block_layers.3._mix_ops.8.conv1.weight", "_blocks.0._block_layers.3._mix_ops.8.conv2.weight", "_blocks.0._block_layers.3._mix_ops.8.conv3.weight", "_blocks.0._block_layers.3._mix_ops.11.conv1.weight", "_blocks.0._block_layers.3._mix_ops.11.conv2.weight", "_blocks.0._block_layers.3._mix_ops.11.conv3.weight", "_blocks.1._block_layers.0._mix_ops.5.downsample_d.0.weight", "_blocks.1._block_layers.0._mix_ops.5.conv1_d.weight", "_blocks.1._block_layers.0._mix_ops.5.conv1.weight", "_blocks.1._block_layers.0._mix_ops.5.conv2.weight", "_blocks.1._block_layers.0._mix_ops.5.conv3.weight", "_blocks.1._block_layers.0._mix_ops.8.downsample_d.0.weight", "_blocks.1._block_layers.0._mix_ops.8.conv1_d.weight", "_blocks.1._block_layers.0._mix_ops.8.conv1.weight", "_blocks.1._block_layers.0._mix_ops.8.conv2.weight", "_blocks.1._block_layers.0._mix_ops.8.conv3.weight", "_blocks.1._block_layers.1._mix_ops.5.downsample_d.0.weight", "_blocks.1._block_layers.1._mix_ops.5.conv1_d.weight", "_blocks.1._block_layers.1._mix_ops.5.conv1.weight", "_blocks.1._block_layers.1._mix_ops.5.conv2.weight", "_blocks.1._block_layers.1._mix_ops.5.conv3.weight", "_blocks.1._block_layers.1._mix_ops.8.conv1.weight", "_blocks.1._block_layers.1._mix_ops.8.conv2.weight", "_blocks.1._block_layers.1._mix_ops.8.conv3.weight", "_blocks.1._block_layers.2._mix_ops.5.downsample_d.0.weight", "_blocks.1._block_layers.2._mix_ops.5.conv1_d.weight", "_blocks.1._block_layers.2._mix_ops.5.conv1.weight", "_blocks.1._block_layers.2._mix_ops.5.conv2.weight", "_blocks.1._block_layers.2._mix_ops.5.conv3.weight", "_blocks.1._block_layers.2._mix_ops.8.conv1.weight", "_blocks.1._block_layers.2._mix_ops.8.conv2.weight", "_blocks.1._block_layers.2._mix_ops.8.conv3.weight", "_blocks.1._block_layers.3._mix_ops.1.downsample_d.0.weight", "_blocks.1._block_layers.3._mix_ops.1.conv1_d.weight", "_blocks.1._block_layers.3._mix_ops.1.conv2.weight", "_blocks.1._block_layers.3._mix_ops.1.conv3.weight", "_blocks.1._block_layers.3._mix_ops.5.downsample_d.0.weight", "_blocks.1._block_layers.3._mix_ops.5.conv1_d.weight", "_blocks.1._block_layers.3._mix_ops.5.conv1.weight", "_blocks.1._block_layers.3._mix_ops.5.conv2.weight", "_blocks.1._block_layers.3._mix_ops.5.conv3.weight", "_blocks.1._block_layers.3._mix_ops.8.conv1.weight", "_blocks.1._block_layers.3._mix_ops.8.conv2.weight", "_blocks.1._block_layers.3._mix_ops.8.conv3.weight", "_blocks.2._block_layers.0._mix_ops.1.downsample_d.0.weight", "_blocks.2._block_layers.0._mix_ops.1.conv1_d.weight", "_blocks.2._block_layers.0._mix_ops.1.conv1.weight", "_blocks.2._block_layers.0._mix_ops.1.conv2.weight", "_blocks.2._block_layers.0._mix_ops.1.conv3.weight", "_blocks.2._block_layers.0._mix_ops.5.downsample_d.0.weight", "_blocks.2._block_layers.0._mix_ops.5.conv1_d.weight", "_blocks.2._block_layers.0._mix_ops.5.conv1.weight", "_blocks.2._block_layers.0._mix_ops.5.conv2.weight", "_blocks.2._block_layers.0._mix_ops.5.conv3.weight", "_blocks.2._block_layers.1._mix_ops.1.downsample_d.0.weight", "_blocks.2._block_layers.1._mix_ops.1.conv1_d.weight", "_blocks.2._block_layers.1._mix_ops.1.conv1.weight", "_blocks.2._block_layers.1._mix_ops.1.conv2.weight", "_blocks.2._block_layers.1._mix_ops.1.conv3.weight", "_blocks.2._block_layers.1._mix_ops.5.conv1.weight", "_blocks.2._block_layers.1._mix_ops.5.conv2.weight", "_blocks.2._block_layers.1._mix_ops.5.conv3.weight", "_blocks.2._block_layers.2._mix_ops.1.downsample_d.0.weight", "_blocks.2._block_layers.2._mix_ops.1.conv1_d.weight", "_blocks.2._block_layers.2._mix_ops.1.conv1.weight", "_blocks.2._block_layers.2._mix_ops.1.conv2.weight", "_blocks.2._block_layers.2._mix_ops.1.conv3.weight", "_blocks.2._block_layers.2._mix_ops.5.conv1.weight", "_blocks.2._block_layers.2._mix_ops.5.conv2.weight", "_blocks.2._block_layers.2._mix_ops.5.conv3.weight", "_blocks.2._block_layers.3._mix_ops.1.downsample_d.0.weight", "_blocks.2._block_layers.3._mix_ops.1.conv1_d.weight", "_blocks.2._block_layers.3._mix_ops.1.conv1.weight", "_blocks.2._block_layers.3._mix_ops.1.conv2.weight", "_blocks.2._block_layers.3._mix_ops.1.conv3.weight", "_blocks.2._block_layers.3._mix_ops.5.conv1.weight", "_blocks.2._block_layers.3._mix_ops.5.conv2.weight", "_blocks.2._block_layers.3._mix_ops.5.conv3.weight", "_blocks.3._block_layers.0._mix_ops.1.downsample_d.0.weight", "_blocks.3._block_layers.0._mix_ops.1.conv1_d.weight", "_blocks.3._block_layers.0._mix_ops.1.conv1.weight", "_blocks.3._block_layers.0._mix_ops.1.conv2.weight", "_blocks.3._block_layers.0._mix_ops.1.conv3.weight", "_blocks.3._block_layers.1._mix_ops.1.conv1.weight", "_blocks.3._block_layers.1._mix_ops.1.conv2.weight", "_blocks.3._block_layers.1._mix_ops.1.conv3.weight", "_blocks.3._block_layers.2._mix_ops.1.conv1.weight", "_blocks.3._block_layers.2._mix_ops.1.conv2.weight", "_blocks.3._block_layers.2._mix_ops.1.conv3.weight", "_blocks.3._block_layers.3._mix_ops.1.conv1.weight", "_blocks.3._block_layers.3._mix_ops.1.conv2.weight", "_blocks.3._block_layers.3._mix_ops.1.conv3.weight"]
        ckpt = my_load_state_dict(checkpoint['model'], names)
        model_without_ddp.load_state_dict(ckpt)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

    # for name, param in model.named_parameters():
    #     print(name)

    # for name, m in model.named_modules():
    #     print(name)

    # exit()

    # plots histogram of weights
    def plot_fig_weights(flat_weights, title_name):

        NUM_PLOT_BINS = 30
        font_board = 2

        ax = fig.add_subplot(1, 4, title_name)
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


    # plot add
    fig = plt.figure(figsize=(9,2.5))
    layer_count = 0
    module_count = 0
    sub_fig_count = 0

    shared_weight = None
    for name, m in model.named_modules():
        # if isinstance(m, Adder2D):
        if 'depth' in name:
            shared_weight = m.shared_weight
            for key, item in shared_weight.items():
                weight = item.data.cpu().reshape(-1)
                # weight = weight[weight.nonzero()]
                print(layer_count, '-th layer: ', name, weight.shape)
                layer_count += 1
                # plot
                if layer_count >= 34 and layer_count < 38:
                    sub_fig_count += 1
                    plot_fig_weights(weight.reshape(1, -1), sub_fig_count)

        # if isinstance(m, ResAdd_ws_dist):
        #     print(name)
        #     module_count += 1

        #     stage = 30

        #     conv3_affine = m.conv3_affine

        #     weight_shape = shared_weight['conv3'].shape
        #     weight = shared_weight['conv3'].reshape(-1)
        #     weight_min = weight.min()
        #     weight_max = weight.max()
        #     for i in range(stage):
        #         # temp = weight.detach() * self.conv3_affine[i]
        #         temp = weight * conv3_affine[i]
        #         index = (weight >= (weight_min + i * (weight_max - weight_min) / stage)) & (weight < (weight_min + (i+1) * (weight_max - weight_min) / stage))
        #         temp =  temp * index

        #         if i == 0:
        #             new_conv3_weight = temp
        #         else:
        #             new_conv3_weight = new_conv3_weight + temp
        #             del temp

        #     # plot
        #     if module_count >= 34 and module_count < 38:
        #         sub_fig_count += 1
        #         plot_fig_weights(weight.reshape(1, -1), sub_fig_count)

    print('total conv layers: ', layer_count)
    print('total ResAdd_ws_dist modules: ', module_count)
    plt.tight_layout()
    plt.savefig(args.output_dir+'/conv.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    args.output_dir = os.path.join(args.output_dir, '-'.join([
        args.model,
    ]))

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
