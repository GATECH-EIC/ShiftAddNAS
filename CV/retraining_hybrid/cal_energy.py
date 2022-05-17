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
import boss_models
import boss_models_Q
from datasets import build_dataset
from engine import train_one_epoch, evaluate
from samplers import RASampler

from boss_candidates.adder import adder
from boss_candidates.deepshift import modules, modules_q
from boss_candidates.bot_op import MatMul

import csv
from hw_utils import get_OPs_HW_metric

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


def cal_flops(model=None, model_name='random', input_res=224, multiply_adds=True):
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


    list_conv = []
    list_conv_ops = []
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        # flops = (kernel_ops * (2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width * batch_size

        num_weight_params = (self.weight.data != 0).float().sum()
        flops = (num_weight_params * (2 if multiply_adds else 1) + bias_ops * output_channels) * output_height * output_width * batch_size

        conv_op = {
            "idx": len(list_conv_ops)+1,
            "type": "Conv",
            "batch": batch_size,
            "kernel_size": self.kernel_size[0],
            "stride": self.stride if type(self.stride) is int else self.stride[0],
            "padding": self.padding if type(self.padding) is int else self.padding[0],
            "input_H": input_height,
            "input_W": input_width,
            "input_C": input_channels,
            "output_E": output_height,
            "output_F": output_width,
            "output_M": output_channels
        }

        list_conv.append(flops)
        list_conv_ops.append(conv_op)

    list_add = []
    list_add_ops = []
    def add_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size * (self.input_channel / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        # flops = (kernel_ops * (2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width * batch_size

        num_weight_params = (self.adder.data != 0).float().sum()
        flops = (num_weight_params * (2 if multiply_adds else 1) + bias_ops * output_channels) * output_height * output_width * batch_size

        add_op = {
            "idx": len(list_add_ops)+1,
            "type": "Add",
            "batch": batch_size,
            "kernel_size": self.kernel_size,
            "stride": self.stride if type(self.stride) is int else self.stride[0],
            "padding": self.padding if type(self.padding) is int else self.padding[0],
            "input_H": input_height,
            "input_W": input_width,
            "input_C": input_channels,
            "output_E": output_height,
            "output_F": output_width,
            "output_M": output_channels
        }

        list_add.append(flops)
        list_add_ops.append(add_op)

    list_shift = []
    list_shift_ops = []
    def shift_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        # flops = (kernel_ops * (2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width * batch_size

        num_weight_params = (self.shift.data != 0).float().sum()
        flops = (num_weight_params * (2 if multiply_adds else 1) + bias_ops * output_channels) * output_height * output_width * batch_size

        shift_op = {
            "idx": len(list_shift_ops)+1,
            "type": "Shift",
            "batch": batch_size,
            "kernel_size": self.kernel_size[0],
            "stride": self.stride if type(self.stride) is int else self.stride[0],
            "padding": self.padding if type(self.padding) is int else self.padding[0],
            "input_H": input_height,
            "input_W": input_width,
            "input_C": input_channels,
            "output_E": output_height,
            "output_F": output_width,
            "output_M": output_channels
        }

        list_shift.append(flops)
        list_shift_ops.append(shift_op)

    list_shift_q = []
    list_shift_q_ops = []
    def shift_q_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        # flops = (kernel_ops * (2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width * batch_size

        num_weight_params = (self.weight.data != 0).float().sum()
        flops = (num_weight_params * (2 if multiply_adds else 1) + bias_ops * output_channels) * output_height * output_width * batch_size

        shift_q_op = {
            "idx": len(list_shift_q_ops)+1,
            "type": "Shift",
            "batch": batch_size,
            "kernel_size": self.kernel_size[0],
            "stride": self.stride if type(self.stride) is int else self.stride[0],
            "padding": self.padding if type(self.padding) is int else self.padding[0],
            "input_H": input_height,
            "input_W": input_width,
            "input_C": input_channels,
            "output_E": output_height,
            "output_F": output_width,
            "output_M": output_channels
        }

        list_shift_q.append(flops)
        list_shift_q_ops.append(shift_q_op)

    list_linear = []
    list_linear_ops = []
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)

        linear_q_op = {
            "idx": len(list_linear_ops)+1,
            "type": "FC",
            "batch": batch_size,
            "kernel_size": 1,
            "stride": 1,
            "padding": 0,
            "input_H": 1,
            "input_W": 1,
            "input_C": self.weight.shape[1],
            "output_E": 1,
            "output_F": 1,
            "output_M": self.weight.shape[0]
        }

        list_linear.append(flops)
        list_linear_ops.append(linear_q_op)

    list_matmul = []
    list_matmul_ops = []
    def matmul_hook(self, input, output):
        # print(input[0].size(), input[1].size())

        batch_size = input[0].size(2)
        ops = input[1].nelement() * (2 if multiply_adds else 1)
        flops = batch_size * ops

        matmul_op = {
            "idx": len(list_matmul_ops)+1,
            "type": "FC",
            "batch": input[0].size(1) * input[0].size(2),
            "kernel_size": 1,
            "stride": 1,
            "padding": 0,
            "input_H": 1,
            "input_W": 1,
            "input_C": input[0].size(3),
            "output_E": 1,
            "output_F": 1,
            "output_M": input[1].size(3)
        }

        list_matmul.append(flops)
        list_matmul_ops.append(matmul_op)

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
            if isinstance(net, MatMul):
                net.register_forward_hook(matmul_hook)
            return
        for c in childrens:
            foo(c)

    if model == None:
        model = torchvision.models.alexnet()
    foo(model)
    input = Variable(torch.rand(3,input_res,input_res).unsqueeze(0), requires_grad = True)#.cuda()
    out = model(input)


    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling) + sum(list_upsample) + sum(list_add) + sum(list_shift) + sum(list_shift_q)) + sum(list_matmul)
    mult_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling) + sum(list_upsample)) + sum(list_matmul)
    add_flops = sum(list_add)
    shift_flops = sum(list_shift) + sum(list_shift_q)

    print('  + Number of FLOPs: %.2fG (Mult: %.2fG / Add: %.2fG / Shift: %.2fG)' % (total_flops / 1e9, mult_flops / 1e9, add_flops / 1e9, shift_flops / 1e9))
    print('  + Number of MACs : %.2fG (Mult: %.2fG / Add: %.2fG / Shift: %.2fG)' % (total_flops / 2 / 1e9, mult_flops / 2 / 1e9, add_flops / 2 / 1e9, shift_flops / 2 / 1e9))


    print('conv ops: \t', len(list_conv_ops))
    print('add ops: \t', len(list_add_ops))
    print('shift ops: \t', len(list_shift_ops))
    print('shift_q ops: \t', len(list_shift_q_ops))
    print('linear ops: \t', len(list_linear_ops))
    print('matmul ops: \t', len(list_matmul_ops))

    save_dir = './hw_record/{}-{}/'.format(model_name, str(input_res))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    conv_energy, conv_latency = prediction(list_conv_ops, save_name=save_dir+'conv.csv', num_bits=32)
    add_energy, add_latency = prediction(list_add_ops, save_name=save_dir+'add.csv', num_bits=32)
    shift_energy, shift_latency = prediction(list_shift_ops, save_name=save_dir+'shift.csv', num_bits=32)
    shift_q_energy, shift_q_latency = prediction(list_shift_q_ops, save_name=save_dir+'shfit_q.csv', num_bits=32)
    linear_energy, linear_latency = prediction(list_linear_ops, save_name=save_dir+'linear.csv', num_bits=32)
    matmul_energy, matmul_latency = prediction(list_matmul_ops, save_name=save_dir+'matmul.csv', num_bits=32)

    mult_energy = conv_energy + linear_energy + matmul_energy
    mult_latency = conv_latency + linear_latency + matmul_latency
    total_energy = conv_energy + add_energy + shift_energy + shift_q_energy + linear_energy + matmul_energy
    total_latency = conv_latency + add_latency + shift_latency + shift_q_latency + linear_latency + matmul_latency


    print('  + Energy (mJ): %.2f (Mult: %.2f / Add: %.2f / Shift: %.2f)' % (total_energy, mult_energy, add_energy, shift_energy+shift_q_energy))
    print('  + Latency (ms): %.2f (Mult: %.2f / Add: %.2f / Shift: %.2f)' % (total_latency, mult_latency, add_latency, shift_latency+shift_q_latency))

    return total_flops, total_energy, total_latency

def prediction(OPs_list, save_name, num_bits=32):
    if os.path.exists(save_name):
        os.remove(save_name)

    if len(OPs_list) == 0:
        return 0, 0

    total_energy = 0
    total_latency = 0
    for item in OPs_list:
        idx = item["idx"]
        print("====>processing {} ops: {}".format(idx, item))
        energy, latency, breakdown, min_energy, min_latency = get_OPs_HW_metric(OPs_list[idx-1], v_stats=False, v_show_optimal=False, v_align=True, num_bits=num_bits)
        OPs_list[idx-1]["energy"] = energy
        OPs_list[idx-1]["latency"]= latency
        print("============================>{}st OPs, energy: {} (min: {}) mJ, latency: {} (min: {}) ms".format(idx, energy, min_energy, latency, min_latency))
        # print("                               >energy breakdown: computation: {} mJ; DRAM: {} mJ; DRAM-GB: {} mJ; GB: {} mJ; NoC: {} mJ; RF: {} mJ".format(breakdown[0], breakdown[1], breakdown[2], breakdown[3], breakdown[4], breakdown[5]))
        # print("                               >energy breakdown: input: {} mJ; weight: {} mJ; output: {} mJ".format(breakdown[6], breakdown[7], breakdown[8]))
        data = [idx, breakdown[0]*1e9, energy, latency, breakdown[6], breakdown[7], breakdown[8], breakdown[0], breakdown[1], breakdown[2], breakdown[3], breakdown[4], breakdown[5] ]
        with open(save_name, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(data)

        total_energy += energy
        total_latency +=  latency

    return total_energy, total_latency*1000


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

    # model = model.cuda()

    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('number of params:', n_parameters)

    print_model_param_nums(model)
    n_flops, n_energy, n_latency = cal_flops(model, args.model, args.input_size)
    # print('number of flops: ', n_flops.data)

    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Calculating FLOPs scripts', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
