import random
import math
import numpy as np
import torch
import torch.nn as nn

from bossnas.models.utils.hytra_paths import all_path, get_all_config_v1, get_all_config_v2
from bossnas.models.operations.operation_dict import OPS
from bossnas.models.operations.operation_dict import reset
from openselfsup.models.registry import BACKBONES

from timm.models.layers import create_classifier, DropPath
from timm.models.registry import register_model
from timm.models.resnet import drop_blocks
from timm.models.vision_transformer import _cfg
from itertools import chain

PRIMITIVES = [
    'ResAtt',
    'ResConv_ws',
    'ResAdd_ws',
    'ResShift_ws'
    # 'ResAtt_SiLU',
    # 'ResConv_SESiLU'
]


def uniform_random_op_encoding(num_of_ops, layers):
    return np.random.randint(0, num_of_ops, layers)


def fair_random_op_encoding(num_of_ops, layers):
    # return alist
    encodings = np.zeros((layers, num_of_ops), dtype=np.int8)
    for i in range(layers):
        encodings[:][i] = np.random.choice(np.arange(0, num_of_ops),
                                           size=num_of_ops,
                                           replace=False)
    return encodings.T.tolist()


def mix_random_op_encoding(num_of_ops, start_block, layers):
    encodings = np.zeros((layers, num_of_ops), dtype=np.int8)
    for i in range(layers):
        encodings[:][i] = np.random.choice(np.arange(0, num_of_ops),
                                           size=num_of_ops,
                                           replace=False)
    # min_layer can't have ship_conn
    min_layers = layers // 2
    for i in range(min_layers):
        encodings[:][i] = np.random.choice(np.arange(1, num_of_ops),
                                           size=num_of_ops, )
    return encodings.T.tolist()


def get_path(str, num):
    if (num == 1):
        for x in str:
            yield x
    else:
        for x in str:
            for y in get_path(str, num - 1):
                yield x + y


def all_op_encoding(num_of_ops, layers):
    # return alist
    encodings = []
    strKey = ""
    for x in range(num_of_ops):
        strKey += str(x)
    for path in get_path(strKey, layers):
        encodings.append([int(op) for op in path])
    return encodings


def _get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


class shared_parameters(nn.Module):
    def __init__(self, inplanes, planes, fmap_size, up_inplanes, stride=1, cardinality=1, base_width=64, reduce_first=1, dilation=1, expansion=4):
        super(shared_parameters, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * expansion

        self.shared_weight = {}
        if up_inplanes is not None:
            self.shared_weight['downsample_d.0'] = nn.Parameter(torch.randn(outplanes, up_inplanes, 3, 3)).cuda()
        if inplanes != outplanes:
            self.shared_weight['downsample.0'] = nn.Parameter(torch.randn(outplanes, inplanes, 1, 1)).cuda()
        if up_inplanes is not None:
            self.shared_weight['conv1_d'] = nn.Parameter(torch.randn(first_planes, up_inplanes, 1, 1)).cuda()

        self.shared_weight['conv1'] = nn.Parameter(torch.randn(first_planes, inplanes, 1, 1)).cuda()
        self.shared_weight['conv2'] = nn.Parameter(torch.randn(width, first_planes // cardinality, 3, 3)).cuda()
        self.shared_weight['conv3'] = nn.Parameter(torch.randn(outplanes, width, 1, 1)).cuda()

class MixOps(nn.Module):
    def __init__(self, inc, outc, stage, to_dispatch=False, init_op_index=None, ):
        assert to_dispatch == (init_op_index is not None)
        super(MixOps, self).__init__()
        self.stage = stage
        self._mix_ops = nn.ModuleList()
        # self.stage_depths = [4, 3, 2, 2]
        self.stage_depths = [4, 4, 4, 4]
        self.layer_fmap = [7, 14, 28, 56]
        # self.layer_c = [2048, 1024, 512, 256]
        self.layer_c = [1024, 512, 256, 128]
        self.downsamples = nn.ModuleDict()
        self.inc = inc
        bot_expansion = 4

        stage_depth = self.stage_depths[self.stage]

        if self.inc == 64:
            in_depth = 3
            for out_depth in range(in_depth + 1):
                outc = self.layer_c[out_depth] // 4
                stride = 2 ** (in_depth - out_depth)
                downsample_cfg = str(inc) + '_' + str(outc)
                self.downsamples[downsample_cfg] = nn.Sequential(
                    # nn.Conv2d(inc, outc, kernel_size=stride, stride=stride),
                    nn.Conv2d(inc, inc, kernel_size=stride, stride=stride, groups=inc, bias=False),
                    nn.Conv2d(inc, outc, kernel_size=1, bias=False),
                    nn.BatchNorm2d(outc))

        for depth in range(stage_depth):
            outc = self.layer_c[depth]
            if self.inc == 64:
                inc = outc // 4
            else:
                inc = outc
            if depth < stage_depth - 1:
                up_inc = self.layer_c[depth + 1]
            else:
                up_inc = None
            fmap_size = self.layer_fmap[depth]

            setattr(self, 'depth_{}_paramters'.format(depth), shared_parameters(inc, outc // 4, fmap_size, up_inc))

            for prim in PRIMITIVES:
                if prim.startswith('ResAtt'):
                    if depth == 0 or depth == 1:
                        head = outc // bot_expansion // 64
                        self._mix_ops.append(OPS[prim](inc, outc, fmap_size, head, up_inc))
                elif prim.startswith('ResConv_ws'):

                    conv = OPS[prim](inc, outc // 4, fmap_size, up_inc, getattr(self, 'depth_{}_paramters'.format(depth)).shared_weight)
                    # for name, m in conv.named_modules():
                    #     if isinstance(m, nn.Conv2d):
                    #         weight_shape = m.weight.shape
                    #         # bias_shape = m.bias.shape
                    #         # prev_weight = m.weight.data.cpu()
                    #         weight = m.weight.reshape(-1)
                    #         # bias = m.bias.data.cpu().reshape(-1)
                    #         # assert torch.all(torch.eq(prev_weight, weight.reshape(shape[0], shape[1], shape[2], shape[3])))
                    #         # print('conv layer:', name, weight.shape, shape)
                    #         # self.shared_conv_weight[name] = {'weight': weight, 'weight_shape': weight_shape, 'bias': bias, 'bias_shape': bias_shape}
                    #         self.shared_conv_weight[name] = {'weight': weight, 'weight_shape': weight_shape}
                    self._mix_ops.append(conv)
                elif prim.startswith('ResAdd_ws'):
                    add = OPS[prim](inc, outc // 4, fmap_size, up_inc, getattr(self, 'depth_{}_paramters'.format(depth)).shared_weight)
                    self._mix_ops.append(add)
                elif prim.startswith('ResShift_ws'):
                    shift = OPS[prim](inc, outc // 4, fmap_size, up_inc, getattr(self, 'depth_{}_paramters'.format(depth)).shared_weight)
                    self._mix_ops.append(shift)
                # elif prim.startswith('ResShift'):
                #     shift = OPS[prim](inc, outc // 4, fmap_size, up_inc)
                #     self._mix_ops.append(shift)
                else:
                    print('wrong PRIMITIVES!')
                    exit()

    def forward(self, x, forward_index):
        # print('forward_index', forward_index)
        # print('stage: ', self.stage)

        # print('mix op input dim (prev): ', x.shape)

        if self.inc == 64:
            inc = x.shape[1]
            outc = self._mix_ops[forward_index].inc
            downsample_cfg = str(inc) + '_' + str(outc)
            x = self.downsamples[downsample_cfg](x)

        # print('mix op input dim (next): ', x.shape)

        return self._mix_ops[forward_index](x)


class Block(nn.Module):
    def __init__(self, inc, hidden_outc, outc, layers, stage, to_dispatch=False, init_op_list=None):
        super(Block, self).__init__()
        init_op_list = init_op_list if init_op_list is not None else [None] * layers  # to_dispatch
        self._block_layers = nn.ModuleList()
        # TODO:
        if layers == 1:
            self._block_layers.append(
                MixOps(inc, outc, to_dispatch, init_op_list[0]))
        else:
            for i in range(layers):
                if i == 0:
                    self._block_layers.append(
                        MixOps(inc, hidden_outc, stage, to_dispatch, init_op_list[i], ))
                elif i == layers - 1:
                    self._block_layers.append(
                        MixOps(hidden_outc, outc, stage, to_dispatch, init_op_list[i], ))
                else:
                    self._block_layers.append(
                        MixOps(hidden_outc, hidden_outc, stage, to_dispatch, init_op_list[i], ))

    def forward(self, x, forwad_list=None):
        assert len(forwad_list) == len(self._block_layers)
        for i, layer in enumerate(self._block_layers):
            x = layer(x, forward_index=forwad_list[i])
        return x

    def reset_params(self):
        self.apply(reset)


@BACKBONES.register_module
class SupernetHyTra_ws(nn.Module):

    def __init__(self, to_dispatch=False, init_op_list=None, block_layers_num=None, num_classes=100):
        super(SupernetHyTra_ws, self).__init__()
        # [origin_outc, outc, num_layers]
        # self.block_cfgs = [[256, 256, 4],
        #                    [512, 512, 4],
        #                    [1024, 1024, 4],
        #                    [2048, 2048, 4],
        #                    ]
        self.block_cfgs = [[128, 128, 4],
                           [256, 256, 4],
                           [512, 512, 4],
                           [1024, 1024, 4],
                           ]
        self.stage_depths = [4, 3, 2, 2]  # restrict candidate scales for each stage
        if block_layers_num is not None:
            for i in range(len(self.block_cfgs)):
                self.block_cfgs[i][3] = block_layers_num[i]

        self._to_dis = to_dispatch
        self._op_layers_list = [cfg[-1] for cfg in self.block_cfgs]
        self._init_op_list = init_op_list if init_op_list is not None else [None] * sum(
            self._op_layers_list)  # dispatch params

        # print(self._op_layers_list)

        in_chans = 3
        stem_width = 64
        deep_stem = False
        norm_layer = nn.BatchNorm2d
        self.inplanes = stem_width * 2 if deep_stem else 64
        if deep_stem:
            self.conv1 = nn.Sequential(*[
                nn.Conv2d(in_chans, stem_width, 3, stride=2, padding=1, bias=False),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, stem_width, 3, stride=1, padding=1, bias=False),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, self.inplanes, 3, stride=1, padding=1, bias=False)])
        else:
            self.conv1 = nn.Conv2d(in_chans, stem_width, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self._stem = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool)
        feat_dim = self._make_block(self.block_cfgs)
        self._num_of_ops = len(PRIMITIVES)

        self.num_features = feat_dim
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes,
                                                      pool_type='avg')

        # self.all_configs = get_all_config_v2()
        self.all_configs = np.load('all_configs.npy').tolist()

    def init_weights(self):
        pass  # FIXME: Using pytorch default weight init

    def _make_block(self, block_cfgs, inc=64):
        self._blocks = nn.ModuleList()
        block_layer_index = 0
        for i, sing_cfg in enumerate(block_cfgs):
            hidden_outc, outc, layers = sing_cfg[0], sing_cfg[1], sing_cfg[2]
            self._blocks.append(
                Block(inc, hidden_outc, outc, layers, stage=i, to_dispatch=self._to_dis,
                      init_op_list=self._init_op_list[block_layer_index:block_layer_index + layers]))
            inc = outc
            block_layer_index += layers

        return inc

    def forward_feature(self, x, start_block, forward_op=None, block_op=True):
        # outs = []
        if start_block == 0:
            x = self._stem(x)

        # print('after stem: ', x.shape)
        for i, block in enumerate(self._blocks):
            if i < start_block:  # if start_block = 1 , forward will skip the block 0 and 1
                continue
            if block_op:
                x = block(x, forward_op)
                # print('after block op: ', x.shape)
            else:
                x = block(x, forward_op[sum(self._op_layers_list[:i]):sum(
                    self._op_layers_list[:(i + 1)])])
                # print('after block op (else): ', x.shape)
            # break
        # self.visualize_feature = x
        # outs.append(x)
        # return tuple(outs)
        return x

    def forward(self, x):
        # example
        # forward_op = [[5,5,5,5], [4,4,4,4], [2,2,2,2], [0,0,0,0]]
        # forward_op = list(chain(*forward_op))

        # random select one
        forward_op = random.sample(self.all_configs, 1)[0]

        # print(forward_op)

        x = self.forward_feature(x, start_block=0, forward_op=forward_op, block_op=False)
        x = self.global_pool(x)
        x = self.fc(x)
        return x


    def set_forward_cfg(self, method='random', start_block=0):  # support method: uniform/fair
        # TODO: support fair
        if self._to_dis:  # stand-alone must be zeros
            forward_op = np.zeros(sum(self._op_layers_list), dtype=int)
        elif method == 'uni':
            forward_op = uniform_random_op_encoding(num_of_ops=self._num_of_ops,
                                                    layers=sum(self._op_layers_list))
        elif method == 'fair':
            forward_op = fair_random_op_encoding(num_of_ops=self._num_of_ops,
                                                 layers=sum(self._op_layers_list))
        elif method == 'mix':
            forward_op = mix_random_op_encoding(num_of_ops=self._num_of_ops,
                                                start_block=start_block,
                                                layers=self._op_layers_list[start_block])
        elif method == 'random':
            print('start_block: ', start_block)
            print('all_path: ', self.get_all_path(start_block))
            forward_op = random.sample(self.get_all_path(start_block), 4)
        else:
            raise NotImplementedError
        return forward_op

    def get_all_path(self, start_block):
        return all_path[self.stage_depths[start_block]]

    def reset_params(self):
        self.apply(reset)

    @classmethod
    def dispatch(cls, init_op_list, block_layers_num=None):
        return cls(True, init_op_list, block_layers_num)

    def step_start_trigger(self):
        '''generate fair choices'''
        pass

    def get_layers(self, block):
        '''get num layers of a block'''
        return self.block_cfgs[block][3]

    def get_block(self, block_num):
        '''get block module to train separately'''
        return self._blocks[block_num]


@register_model
def Supernet_v2(pretrained=False, **kwargs):
    model = SupernetHyTra_ws()
    model.default_cfg = _cfg()
    return model


def get_model_parameters_number(model):
    params_num = sum(p.numel() for p in model.parameters())  # if p.requires_grad)
    return params_num

