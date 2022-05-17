import argparse
import datetime
import json
import numpy as np
import os
import time
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable

import torch.backends.cudnn as cudnn
from pathlib import Path
from timm.data import Mixup
from timm.data.distributed_sampler import OrderedDistributedSampler
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models import create_model
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler, get_state_dict, ModelEma, ApexScaler

import utils
import cal_params_flops.boss_models
import cal_params_flops.boss_models_Q
from datasets import build_dataset
from engine import train_one_epoch, evaluate
from samplers import RASampler

from cal_params_flops.boss_candidates.adder import adder
from cal_params_flops.boss_candidates.deepshift import modules, modules_q

def get_args_parser():
    parser = argparse.ArgumentParser('Calculating FLOPs scripts', add_help=False)
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

    return parser


def cal_flops(model=None, input_res=224, multiply_adds=True):
    prods = {}
    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)
        return hook_per

    list_1=[]
    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))
    list_2={}
    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)


    list_conv=[]
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        # flops = (kernel_ops * (2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width * batch_size

        num_weight_params = (self.weight.data != 0).float().sum()
        flops = (num_weight_params * (2 if multiply_adds else 1) + bias_ops * output_channels) * output_height * output_width * batch_size

        list_conv.append(flops)

    list_add=[]
    def add_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size * (self.input_channel / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        # flops = (kernel_ops * (2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width * batch_size

        num_weight_params = (self.adder.data != 0).float().sum()
        flops = (num_weight_params * (2 if multiply_adds else 1) + bias_ops * output_channels) * output_height * output_width * batch_size

        list_add.append(flops)

    list_shift=[]
    def shift_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        # flops = (kernel_ops * (2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width * batch_size

        num_weight_params = (self.shift.data != 0).float().sum()
        flops = (num_weight_params * (2 if multiply_adds else 1) + bias_ops * output_channels) * output_height * output_width * batch_size

        list_shift.append(flops)

    list_shift_q=[]
    def shift_q_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        # flops = (kernel_ops * (2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width * batch_size

        num_weight_params = (self.weight.data != 0).float().sum()
        flops = (num_weight_params * (2 if multiply_adds else 1) + bias_ops * output_channels) * output_height * output_width * batch_size

        list_shift_q.append(flops)

    list_linear=[]
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn=[]
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement() * 2)

    list_relu=[]
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling=[]
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = 0
        flops = (kernel_ops + bias_ops) * output_channels * output_height * output_width * batch_size

        list_pooling.append(flops)

    list_upsample=[]

    # For bilinear upsample
    def upsample_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        flops = output_height * output_width * output_channels * batch_size * 12
        list_upsample.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            if isinstance(net, torch.nn.Upsample):
                net.register_forward_hook(upsample_hook)
            if isinstance(net, adder.Adder2D):
                net.register_forward_hook(add_hook)
            if isinstance(net, modules.Conv2dShift):
                net.register_forward_hook(shift_hook)
            if isinstance(net, modules_q.Conv2dShiftQ):
                net.register_forward_hook(shift_q_hook)
            return
        for c in childrens:
            foo(c)

    if model == None:
        model = torchvision.models.alexnet()
    foo(model)
    input = Variable(torch.rand(3,input_res,input_res).unsqueeze(0), requires_grad = True).cuda()
    out = model(input)

    del input

    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling) + sum(list_upsample) + sum(list_add) + sum(list_shift) + sum(list_shift_q))
    mult_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling) + sum(list_upsample))
    add_flops = sum(list_add)
    shift_flops = sum(list_shift) + sum(list_shift_q)

    print('  + Number of FLOPs: %.2fG (Mult: %.2fG / Add: %.2fG / Shift: %.2fG)' % (total_flops / 1e9, mult_flops / 1e9, add_flops / 1e9, shift_flops / 1e9))
    print('  + Number of MACs : %.2fG (Mult: %.2fG / Add: %.2fG / Shift: %.2fG)' % (total_flops / 2 / 1e9, mult_flops / 2 / 1e9, add_flops / 2 / 1e9, shift_flops / 2 / 1e9))

    return total_flops.item()

def print_model_param_nums(model=None):
    if model == None:
        model = torchvision.models.alexnet()
    total = sum([param.nelement() if param.requires_grad else 0 for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))

def main(args):

    print(f"Creating model: {args.model}")

    args.nb_classes = 1000

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
    )

    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('number of params:', n_parameters)

    print_model_param_nums(model)
    n_flops = cal_flops(model, args.input_size)
    # print('number of flops: ', n_flops.data)

    return

def encoder_config(encoding):
    config = [[], [], [], []]
    mapping = {0: [11, 12, 13], 1: [8, 9, 10], 2: [4, 5, 6, 7], 3: [0, 1]}
    for i, item in enumerate(encoding):
        if item in mapping[0]:
            index = 0
            types = item - 10
        elif item in mapping[1]:
            index = 1
            types = item - 7
        elif item in mapping[2]:
            index = 2
            types = item - 4
        elif item in mapping[3]:
            index = 3
            types = item
        else:
            print('error! please check your encoding ...')
            exit()

        config[index].append(types)

    print(config)

    return config

def get_customized_model(encoding):
    config = encoder_config(encoding)

    model = create_model(
        'customized_model',
        pretrained=False,
        encoding=config,
        num_classes=1000,
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_block_rate=None,
    )

    model = model.cuda()

    n_params = sum([param.nelement() if param.requires_grad else 0 for param in model.parameters()])
    n_flops = cal_flops(model, 224)

    del model

    return n_params, n_flops

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Calculating FLOPs scripts', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
