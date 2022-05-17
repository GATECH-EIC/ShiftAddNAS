import math
import torch
from timm.models.layers import create_attn
from torch import nn
import time

from bossnas.models.operations.adder import adder
from bossnas.models.operations.deepshift import modules, modules_q

from torch.distributions.laplace import Laplace
from torch.distributions.normal import Normal
import torch.nn.functional as F

from matplotlib import pyplot as plt

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=128):
        super(Attention, self).__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.to_qkv(x).view(b, 3 * self.heads, self.dim_head, h * w).chunk(3, dim=1)

        q *= self.scale
        attn = q.transpose(-2, -1) @ k
        attn = attn.softmax(dim=-1)

        out = (v @ attn).reshape(b, -1, h, w)
        return out


class PEG(nn.Module):
    """
    modified from PEG in https://github.com/Meituan-AutoML/CPVT
    """
    def __init__(self, dim, stride):
        super(PEG, self).__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=stride, padding=1, groups=dim)

    def forward(self, x):
        # x = x + self.conv(x)
        return self.conv(x)


class PEG_Add(nn.Module):
    """
    modified from PEG in https://github.com/Meituan-AutoML/CPVT
    """
    def __init__(self, dim, stride):
        super(PEG_Add, self).__init__()
        # self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=stride, padding=1, groups=dim)
        self.conv = adder.Adder2D(dim, dim, kernel_size=3, stride=stride, padding=1, groups=dim)

    def forward(self, x):
        # x = x + self.conv(x)
        return self.conv(x)


class PEG_Shift(nn.Module):
    """
    modified from PEG in https://github.com/Meituan-AutoML/CPVT
    """
    def __init__(self, dim, stride):
        super(PEG_Shift, self).__init__()
        # self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=stride, padding=1, groups=dim)
        self.conv = modules.Conv2dShift(dim, dim, kernel_size=3, stride=stride, padding=1, groups=dim, bias=False, weight_bits=5, quant_bits=16)

    def forward(self, x):
        # x = x + self.conv(x)
        return self.conv(x)


class ResAtt(nn.Module):
    """
    modified from BoTNet block in https://github.com/lucidrains/bottleneck-transformer-pytorch
    which is translated from tensorflow code
    https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2
    """
    def __init__(self, dim, fmap_size, dim_out, proj_factor, heads=4, dim_head=128,
                 act_layer=nn.ReLU, up_dim=None):
        super().__init__()
        self.fmap_size = fmap_size
        self.inc = dim
        # shortcut
        if up_dim is not None:
            self.shortcut_d = nn.Sequential(
                nn.Conv2d(up_dim, dim_out, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(dim_out),
                act_layer(inplace=True))
        if dim != dim_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(dim, dim_out, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(dim_out),
                act_layer(inplace=True))
        else:
            self.shortcut = nn.Identity()

        # contraction and expansion
        attn_dim_in = dim_out // proj_factor
        attn_dim_out = heads * dim_head
        if up_dim is not None:
            self.proj_d = nn.Sequential(nn.Conv2d(up_dim, attn_dim_in, 1, bias=False),
                                        PEG(attn_dim_in, stride=2),
                                        nn.BatchNorm2d(attn_dim_in))
        self.proj = nn.Sequential(nn.Conv2d(dim, attn_dim_in, 1, bias=False),
                                  PEG(attn_dim_in, stride=1),
                                  nn.BatchNorm2d(attn_dim_in))

        self.net = nn.Sequential(
            act_layer(inplace=True),
            Attention(
                dim=attn_dim_in,
                heads=heads,
                dim_head=dim_head),
            nn.BatchNorm2d(attn_dim_out),
            act_layer(inplace=True),
            nn.Conv2d(attn_dim_out, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out))

        # init last batch norm gamma to zero
        nn.init.zeros_(self.net[-1].weight)

        # final activation
        self.activation = act_layer(inplace=True)

    def zero_init_last_bn(self):
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, x):
        fmap_size = x.shape[-1]
        if fmap_size > self.fmap_size:
            shortcut = self.shortcut_d(x)
            x = self.proj_d(x)
        else:
            shortcut = self.shortcut(x)
            x = self.proj(x)
        x = self.net(x)
        x += shortcut
        return self.activation(x)


class ResConv_ws_dist(nn.Module):
    """
    modified from ResNet block in timm/models/resnet.py
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, cardinality=1, base_width=64, reduce_first=1, dilation=1,
                 first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, attn_layer=None,
                 aa_layer=None, drop_block=None, drop_path=None, fmap_size=56, up_inplanes=None, shared_weight=None):
        super(ResConv_ws_dist, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)
        self.fmap_size = fmap_size

        self.shared_weight = shared_weight

        if up_inplanes is not None:
            self.downsample_d = nn.Sequential(
                nn.Conv2d(up_inplanes, outplanes, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(outplanes),
                act_layer(inplace=True))

            del self.downsample_d[0].weight

        if inplanes != outplanes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, outplanes, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(outplanes),
                act_layer(inplace=True))

            del self.downsample[0].weight

        else:
            self.downsample = nn.Identity()

        if up_inplanes is not None:
            self.conv1_d = nn.Conv2d(up_inplanes, first_planes, kernel_size=1, bias=False)

            del self.conv1_d.weight

            self.peg_d = PEG(first_planes, stride=2)
            self.bn1_d = norm_layer(first_planes)
        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)

        del self.conv1.weight

        self.peg = PEG(first_planes, stride=1)
        self.bn1 = norm_layer(first_planes)

        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
            padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)

        del self.conv2.weight

        self.bn2 = norm_layer(width)
        self.act2 = act_layer(inplace=True)
        self.aa = aa_layer(channels=width, stride=stride) if use_aa else None

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)

        del self.conv3.weight

        self.bn3 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path
        self.inc = inplanes
        self.zero_init_last_bn()

        self.normal = Normal(0, 1)

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, plot=False):
        fmap_size = x.shape[-1]

        kl_loss = 0

        # print(x.shape)
        # print('fmap_size: ', fmap_size)
        # print('fmap_size (self): ', self.fmap_size)

        if fmap_size > self.fmap_size:

            self.downsample_d[0].weight = self.shared_weight.downsample_d
            self.conv1_d.weight = self.shared_weight.conv1_d

            kl_loss += F.kl_div(F.log_softmax(self.shared_weight.downsample_d.reshape(-1), 0),
                                F.softmax(self.normal.sample(self.shared_weight.downsample_d.reshape(-1).shape).cuda(), 0), reduction="none").mean()
            kl_loss += F.kl_div(F.log_softmax(self.shared_weight.conv1_d.reshape(-1), 0),
                                F.softmax(self.normal.sample(self.shared_weight.conv1_d.reshape(-1).shape).cuda(), 0), reduction="none").mean()

            residual = self.downsample_d(x)
            x = self.conv1_d(x)
            x = self.peg_d(x)
            x = self.bn1_d(x)
        else:

            if not isinstance(self.downsample, nn.Identity):
                self.downsample[0].weight = self.shared_weight.downsample
                kl_loss += F.kl_div(F.log_softmax(self.shared_weight.downsample.reshape(-1), 0),
                                    F.softmax(self.normal.sample(self.shared_weight.downsample.reshape(-1).shape).cuda(), 0), reduction="none").mean()

            self.conv1.weight = self.shared_weight.conv1
            kl_loss += F.kl_div(F.log_softmax(self.shared_weight.conv1.reshape(-1), 0),
                                    F.softmax(self.normal.sample(self.shared_weight.conv1.reshape(-1).shape).cuda(), 0), reduction="none").mean()

            residual = self.downsample(x)
            x = self.conv1(x)
            x = self.peg(x)
            x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        self.conv2.weight = self.shared_weight.conv2
        kl_loss += F.kl_div(F.log_softmax(self.shared_weight.conv2.reshape(-1), 0),
                            F.softmax(self.normal.sample(self.shared_weight.conv2.reshape(-1).shape).cuda(), 0), reduction="none").mean()

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)
        if self.aa is not None:
            x = self.aa(x)

        self.conv3.weight = self.shared_weight.conv3
        kl_loss += F.kl_div(F.log_softmax(self.shared_weight.conv3.reshape(-1), 0),
                            F.softmax(self.normal.sample(self.shared_weight.conv3.reshape(-1).shape).cuda(), 0), reduction="none").mean()

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x += residual
        x = self.act3(x)

        # print('old_weight: ', self.shared_weight.conv3.reshape(-1))

        return x, 0

    def del_redundancy(self):
        if self.downsample_d is not None:
            del self.downsample_d[0].weight
        if self.downsample is not None:
            del self.downsample[0].weight
        if self.conv1_d is not None:
            del self.conv1_d.weight
        if self.conv1 is not None:
            del self.conv1.weight
        del self.conv2.weight
        del self.conv3.weight


class ResAdd_ws_dist(nn.Module):
    """
    modified from ResNet block in timm/models/resnet.py
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, cardinality=1, base_width=64, reduce_first=1, dilation=1,
                 first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, attn_layer=None,
                 aa_layer=None, drop_block=None, drop_path=None, fmap_size=56, up_inplanes=None, shared_weight=None):
        super(ResAdd_ws_dist, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)
        self.fmap_size = fmap_size

        self.shared_weight = shared_weight

        if up_inplanes is not None:
            self.downsample_d = nn.Sequential(
                # nn.Conv2d(up_inplanes, outplanes, 3, stride=2, padding=1, bias=False),
                adder.Adder2D(up_inplanes, outplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(outplanes),
                act_layer(inplace=True))

            del self.downsample_d[0].adder

        if inplanes != outplanes:
            self.downsample = nn.Sequential(
                # nn.Conv2d(inplanes, outplanes, 1, stride=1, padding=0, bias=False),
                adder.Adder2D(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(outplanes),
                act_layer(inplace=True))

            del self.downsample[0].adder

        else:
            self.downsample = nn.Identity()

        if up_inplanes is not None:
            # self.conv1_d = nn.Conv2d(up_inplanes, first_planes, kernel_size=1, bias=False)
            self.conv1_d = adder.Adder2D(up_inplanes, first_planes, kernel_size=1, bias=False)

            del self.conv1_d.adder

            # self.peg_d = PEG(first_planes, stride=2)
            self.peg_d = PEG_Add(first_planes, stride=2)
            self.bn1_d = norm_layer(first_planes)
        # self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.conv1 = adder.Adder2D(inplanes, first_planes, kernel_size=1, bias=False)

        del self.conv1.adder

        # self.peg = PEG(first_planes, stride=1)
        self.peg = PEG_Add(first_planes, stride=1)
        self.bn1 = norm_layer(first_planes)

        self.act1 = act_layer(inplace=True)

        # self.conv2 = nn.Conv2d(
        #     first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
        #     padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        self.conv2 = adder.Adder2D(
            first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
            padding=first_dilation, groups=cardinality, bias=False) ### assume dilation is always 1 !

        del self.conv2.adder

        self.bn2 = norm_layer(width)
        self.act2 = act_layer(inplace=True)
        self.aa = aa_layer(channels=width, stride=stride) if use_aa else None

        # self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.conv3 = adder.Adder2D(width, outplanes, kernel_size=1, bias=False)

        del self.conv3.adder

        self.bn3 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path
        self.inc = inplanes
        self.zero_init_last_bn()


        # piece-wise affine
        self.stage = 100
        self.downsample_d_affine = nn.Parameter(torch.randn(self.stage))
        self.downsample_affine = nn.Parameter(torch.randn(self.stage))
        self.conv1_d_affine = nn.Parameter(torch.randn(self.stage))
        self.conv1_affine = nn.Parameter(torch.randn(self.stage))
        self.conv2_affine = nn.Parameter(torch.randn(self.stage))
        self.conv3_affine = nn.Parameter(torch.randn(self.stage))

        self.laplace = Laplace(0, 2)

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, plot=False):

        kl_loss = 0

        fmap_size = x.shape[-1]

        if fmap_size > self.fmap_size:

            weight_shape = self.shared_weight.downsample_d.shape
            weight = self.shared_weight.downsample_d.reshape(-1)
            weight_min = weight.min()
            weight_max = weight.max()
            for i in range(self.stage):
                # temp = weight.detach() * self.downsample_d_affine[i]
                temp = weight * self.downsample_d_affine[i]
                index = (weight >= (weight_min + i * (weight_max - weight_min) / self.stage)) & (weight < (weight_min + (i+1) * (weight_max - weight_min) / self.stage))
                temp =  temp * index.detach()

                if i == 0:
                    new_downsample_d_weight = temp
                else:
                    new_downsample_d_weight = new_downsample_d_weight + temp
                    del temp

            # self.downsample_d[0].adder = self.shared_weight.downsample_d
            self.downsample_d[0].adder = new_downsample_d_weight.reshape(weight_shape)
            kl_loss += F.kl_div(F.log_softmax(new_downsample_d_weight, 0), F.softmax(self.laplace.sample(new_downsample_d_weight.shape).cuda(), 0), reduction="none").mean()


            weight_shape = self.shared_weight.conv1_d.shape
            weight = self.shared_weight.conv1_d.reshape(-1)
            weight_min = weight.min()
            weight_max = weight.max()
            for i in range(self.stage):
                # temp = weight.detach() * self.conv1_d_affine[i]
                temp = weight * self.conv1_d_affine[i]
                index = (weight >= (weight_min + i * (weight_max - weight_min) / self.stage)) & (weight < (weight_min + (i+1) * (weight_max - weight_min) / self.stage))
                temp =  temp * index.detach()

                if i == 0:
                    new_conv1_d_weight = temp
                else:
                    new_conv1_d_weight = new_conv1_d_weight + temp
                    del temp


            # self.conv1_d.adder = self.shared_weight.conv1_d
            self.conv1_d.adder = new_conv1_d_weight.reshape(weight_shape)
            kl_loss += F.kl_div(F.log_softmax(new_conv1_d_weight, 0), F.softmax(self.laplace.sample(new_conv1_d_weight.shape).cuda(), 0), reduction="none").mean()

            residual = self.downsample_d(x)
            x = self.conv1_d(x)
            x = self.peg_d(x)
            x = self.bn1_d(x)
        else:

            if not isinstance(self.downsample, nn.Identity):

                weight_shape = self.shared_weight.downsample.shape
                weight = self.shared_weight.downsample.reshape(-1)
                weight_min = weight.min()
                weight_max = weight.max()
                for i in range(self.stage):
                    # temp = weight.detach() * self.downsample_affine[i]
                    temp = weight * self.downsample_affine[i]
                    index = (weight >= (weight_min + i * (weight_max - weight_min) / self.stage)) & (weight < (weight_min + (i+1) * (weight_max - weight_min) / self.stage))
                    temp =  temp * index.detach()

                    if i == 0:
                        new_downsample_weight = temp
                    else:
                        new_downsample_weight = new_downsample_weight + temp
                        del temp

                # self.downsample[0].adder = self.shared_weight.downsample
                self.downsample[0].adder = new_downsample_weight.reshape(weight_shape)
                kl_loss += F.kl_div(F.log_softmax(new_downsample_weight, 0), F.softmax(self.laplace.sample(new_downsample_weight.shape).cuda(), 0), reduction="none").mean()

            weight_shape = self.shared_weight.conv1.shape
            weight = self.shared_weight.conv1.reshape(-1)
            weight_min = weight.min()
            weight_max = weight.max()
            for i in range(self.stage):
                # temp = weight.detach() * self.conv1_affine[i]
                temp = weight * self.conv1_affine[i]
                index = (weight >= (weight_min + i * (weight_max - weight_min) / self.stage)) & (weight < (weight_min + (i+1) * (weight_max - weight_min) / self.stage))
                temp =  temp * index.detach()

                if i == 0:
                    new_conv1_weight = temp
                else:
                    new_conv1_weight = new_conv1_weight + temp
                    del temp


            # self.conv1.adder = self.shared_weight.conv1
            self.conv1.adder = new_conv1_weight.reshape(weight_shape)
            kl_loss += F.kl_div(F.log_softmax(new_conv1_weight, 0), F.softmax(self.laplace.sample(new_conv1_weight.shape).cuda(), 0), reduction="none").mean()

            residual = self.downsample(x)
            x = self.conv1(x)
            x = self.peg(x)
            x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)


        # start_time = time.time()


        weight_shape = self.shared_weight.conv2.shape
        weight = self.shared_weight.conv2.reshape(-1)
        weight_min = weight.min()
        weight_max = weight.max()
        for i in range(self.stage):
            # temp = weight.detach() * self.conv2_affine[i]
            temp = weight * self.conv2_affine[i]
            index = (weight >= (weight_min + i * (weight_max - weight_min) / self.stage)) & (weight < (weight_min + (i+1) * (weight_max - weight_min) / self.stage))
            temp =  temp * index.detach()

            if i == 0:
                new_conv2_weight = temp
            else:
                new_conv2_weight = new_conv2_weight + temp
                del temp

        # end_time = time.time()
        # print('time cost: ', end_time - start_time)

        # self.conv2.adder = self.shared_weight.conv2
        self.conv2.adder = new_conv2_weight.reshape(weight_shape)
        kl_loss += F.kl_div(F.log_softmax(new_conv2_weight, 0), F.softmax(self.laplace.sample(new_conv2_weight.shape).cuda(), 0), reduction="none").mean()


        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)
        if self.aa is not None:
            x = self.aa(x)


        weight_shape = self.shared_weight.conv3.shape
        weight = self.shared_weight.conv3.reshape(-1)
        weight_min = weight.min()
        weight_max = weight.max()
        for i in range(self.stage):
            # temp = weight.detach() * self.conv3_affine[i]
            temp = weight * self.conv3_affine[i]
            index = (weight >= (weight_min + i * (weight_max - weight_min) / self.stage)) & (weight < (weight_min + (i+1) * (weight_max - weight_min) / self.stage))
            temp =  temp * index.detach()

            if i == 0:
                new_conv3_weight = temp
            else:
                new_conv3_weight = new_conv3_weight + temp
                del temp

        # self.conv3.adder = self.shared_weight.conv3
        self.conv3.adder = new_conv3_weight.reshape(weight_shape)
        kl_loss += F.kl_div(F.log_softmax(new_conv3_weight, 0), F.softmax(self.laplace.sample(new_conv3_weight.shape).cuda(), 0), reduction="none").mean()


        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x += residual
        x = self.act3(x)

        # print('new_weight: ', new_conv3_weight)
        # print('old_weight: ', self.shared_weight.conv3.reshape(-1))

        # plot
        if plot:


            if fmap_size > self.fmap_size:
                flat_weights = new_downsample_d_weight.data.cpu().reshape(-1)
                flat_weights_2 = new_conv1_d_weight.data.cpu().reshape(-1)
                flat_weights_3 = self.shared_weight.downsample_d.data.cpu().reshape(-1)
                flat_weights_4 = self.shared_weight.conv1_d.data.cpu().reshape(-1)

                fig = plt.figure(figsize=(9,2.5))

                NUM_PLOT_BINS = 30
                font_board = 2


                ax = fig.add_subplot(1, 4, 1)
                # ax.set_title("Pruning " + save_name[title_name-1] + '%')
                ax.set_title("")
                ax.hist(flat_weights.reshape(1, -1), NUM_PLOT_BINS, color='green', alpha=0.8)
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

                ax = fig.add_subplot(1, 4, 2)
                # ax.set_title("Pruning " + save_name[title_name-1] + '%')
                ax.set_title("")
                ax.hist(flat_weights_2.reshape(1, -1), NUM_PLOT_BINS, color='green', alpha=0.8)
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

                ax = fig.add_subplot(1, 4, 3)
                # ax.set_title("Pruning " + save_name[title_name-1] + '%')
                ax.set_title("")
                ax.hist(flat_weights_3.reshape(1, -1), NUM_PLOT_BINS, color='green', alpha=0.8)
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

                ax = fig.add_subplot(1, 4, 4)
                # ax.set_title("Pruning " + save_name[title_name-1] + '%')
                ax.set_title("")
                ax.hist(flat_weights_4.reshape(1, -1), NUM_PLOT_BINS, color='green', alpha=0.8)
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

                plt.tight_layout()
                plt.savefig('add_xxx.pdf')
                plt.close('all')
        # exit()

        return x, kl_loss

    def del_redundancy(self):
        if self.downsample_d is not None:
            del self.downsample_d[0].adder
        if self.downsample is not None:
            del self.downsample[0].adder
        if self.conv1_d is not None:
            del self.conv1_d.adder
        if self.conv1 is not None:
            del self.conv1.adder
        # del self.conv2.adder
        # del self.conv3.adder


class ResShift_ws_dist(nn.Module):
    """
    modified from ResNet block in timm/models/resnet.py
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, cardinality=1, base_width=64, reduce_first=1, dilation=1,
                 first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, attn_layer=None,
                 aa_layer=None, drop_block=None, drop_path=None, fmap_size=56, up_inplanes=None, shared_weight=None):
        super(ResShift_ws_dist, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)
        self.fmap_size = fmap_size

        self.shared_weight = shared_weight

        if up_inplanes is not None:
            self.downsample_d = nn.Sequential(
                # nn.Conv2d(up_inplanes, outplanes, 3, stride=2, padding=1, bias=False),
                modules.Conv2dShift(up_inplanes, outplanes, kernel_size=3, stride=2, padding=1, bias=False, weight_bits=5, quant_bits=16),
                nn.BatchNorm2d(outplanes),
                act_layer(inplace=True))

            # del self.downsample_d[0].weight
            del self.downsample_d[0].shift
            del self.downsample_d[0].sign

        if inplanes != outplanes:
            self.downsample = nn.Sequential(
                # nn.Conv2d(inplanes, outplanes, 1, stride=1, padding=0, bias=False),
                modules.Conv2dShift(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=False, weight_bits=5, quant_bits=16),
                nn.BatchNorm2d(outplanes),
                act_layer(inplace=True))

            # del self.downsample[0].weight
            del self.downsample[0].shift
            del self.downsample[0].sign

        else:
            self.downsample = nn.Identity()

        if up_inplanes is not None:
            # self.conv1_d = nn.Conv2d(up_inplanes, first_planes, kernel_size=1, bias=False)
            self.conv1_d = modules.Conv2dShift(up_inplanes, first_planes, kernel_size=1, bias=False, weight_bits=5, quant_bits=16)

            # del self.conv1_d.weight
            del self.conv1_d.shift
            del self.conv1_d.sign

            # self.peg_d = PEG(first_planes, stride=2)
            self.peg_d = PEG_Shift(first_planes, stride=2)
            self.bn1_d = norm_layer(first_planes)
        # self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.conv1 = modules.Conv2dShift(inplanes, first_planes, kernel_size=1, bias=False, weight_bits=5, quant_bits=16)

        # del self.conv1.weight
        del self.conv1.shift
        del self.conv1.sign

        # self.peg = PEG(first_planes, stride=1)
        self.peg = PEG_Shift(first_planes, stride=1)
        self.bn1 = norm_layer(first_planes)

        self.act1 = act_layer(inplace=True)

        # self.conv2 = nn.Conv2d(
        #     first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
        #     padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        self.conv2 = modules.Conv2dShift(
            first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
            padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False, weight_bits=5, quant_bits=16)

        # del self.conv2.weight
        del self.conv2.shift
        del self.conv2.sign


        self.bn2 = norm_layer(width)
        self.act2 = act_layer(inplace=True)
        self.aa = aa_layer(channels=width, stride=stride) if use_aa else None

        # self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.conv3 = modules.Conv2dShift(width, outplanes, kernel_size=1, bias=False, weight_bits=5, quant_bits=16)

        # del self.conv3.weight
        del self.conv3.shift
        del self.conv3.sign

        self.bn3 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path
        self.inc = inplanes
        self.zero_init_last_bn()

        # piece-wise affine
        self.stage = 100
        self.downsample_d_shift_affine = nn.Parameter(torch.randn(self.stage))
        self.downsample_shift_affine = nn.Parameter(torch.randn(self.stage))
        self.conv1_d_shift_affine = nn.Parameter(torch.randn(self.stage))
        self.conv1_shift_affine = nn.Parameter(torch.randn(self.stage))
        self.conv2_shift_affine = nn.Parameter(torch.randn(self.stage))
        self.conv3_shift_affine = nn.Parameter(torch.randn(self.stage))

        self.downsample_d_sign_affine = nn.Parameter(torch.randn(self.stage))
        self.downsample_sign_affine = nn.Parameter(torch.randn(self.stage))
        self.conv1_d_sign_affine = nn.Parameter(torch.randn(self.stage))
        self.conv1_sign_affine = nn.Parameter(torch.randn(self.stage))
        self.conv2_sign_affine = nn.Parameter(torch.randn(self.stage))
        self.conv3_sign_affine = nn.Parameter(torch.randn(self.stage))

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, plot=False):
        fmap_size = x.shape[-1]

        if fmap_size > self.fmap_size:

            # self.downsample_d[0].weight = self.shared_weight.downsample_d

            weight_shape = self.shared_weight.downsample_d.shape
            weight = self.shared_weight.downsample_d.reshape(-1)
            weight_min = weight.min()
            weight_max = weight.max()
            for i in range(self.stage):
                temp_shift = weight.detach() * self.downsample_d_shift_affine[i]
                temp_sign = weight.detach() * self.downsample_d_sign_affine[i]
                index = (weight >= (weight_min + i * (weight_max - weight_min) / self.stage)) & (weight < (weight_min + (i+1) * (weight_max - weight_min) / self.stage))
                temp_shift = temp_shift * index
                temp_sign = temp_sign * index

                if i == 0:
                    new_downsample_d_shift = temp_shift
                    new_downsample_d_sign = temp_sign
                else:
                    new_downsample_d_shift = new_downsample_d_shift + temp_shift
                    new_downsample_d_sign = new_downsample_d_sign + temp_sign
                    del temp_shift
                    del temp_sign

            # self.downsample_d[0].shift = self.shared_weight.downsample_d
            # self.downsample_d[0].sign = self.shared_weight.downsample_d

            self.downsample_d[0].shift = new_downsample_d_shift.reshape(weight_shape)
            self.downsample_d[0].sign = new_downsample_d_sign.reshape(weight_shape)

            # self.conv1_d.weight = self.shared_weight.conv1_d

            weight_shape = self.shared_weight.conv1_d.shape
            weight = self.shared_weight.conv1_d.reshape(-1)
            weight_min = weight.min()
            weight_max = weight.max()
            for i in range(self.stage):
                temp_shift = weight.detach() * self.conv1_d_shift_affine[i]
                temp_sign = weight.detach() * self.conv1_d_sign_affine[i]
                index = (weight >= (weight_min + i * (weight_max - weight_min) / self.stage)) & (weight < (weight_min + (i+1) * (weight_max - weight_min) / self.stage))
                temp_shift = temp_shift * index
                temp_sign = temp_sign * index

                if i == 0:
                    conv1_d_shift = temp_shift
                    conv1_d_sign = temp_sign
                else:
                    conv1_d_shift = conv1_d_shift + temp_shift
                    conv1_d_sign = conv1_d_sign + temp_sign
                    del temp_shift
                    del temp_sign

            # self.conv1_d.shift = self.shared_weight.conv1_d
            # self.conv1_d.sign = self.shared_weight.conv1_d

            self.conv1_d.shift = conv1_d_shift.reshape(weight_shape)
            self.conv1_d.sign = conv1_d_sign.reshape(weight_shape)

            residual = self.downsample_d(x)
            x = self.conv1_d(x)
            x = self.peg_d(x)
            x = self.bn1_d(x)
        else:

            if not isinstance(self.downsample, nn.Identity):
                # self.downsample[0].weight = self.shared_weight.downsample

                weight_shape = self.shared_weight.downsample.shape
                weight = self.shared_weight.downsample.reshape(-1)
                weight_min = weight.min()
                weight_max = weight.max()
                for i in range(self.stage):
                    temp_shift = weight.detach() * self.downsample_shift_affine[i]
                    temp_sign = weight.detach() * self.downsample_sign_affine[i]
                    index = (weight >= (weight_min + i * (weight_max - weight_min) / self.stage)) & (weight < (weight_min + (i+1) * (weight_max - weight_min) / self.stage))
                    temp_shift = temp_shift * index
                    temp_sign = temp_sign * index

                    if i == 0:
                        downsample_shift = temp_shift
                        downsample_sign = temp_sign
                    else:
                        downsample_shift = downsample_shift + temp_shift
                        downsample_sign = downsample_sign + temp_sign
                        del temp_shift
                        del temp_sign

                # self.downsample[0].shift = self.shared_weight.downsample
                # self.downsample[0].sign = self.shared_weight.downsample

                self.downsample[0].shift = downsample_shift.reshape(weight_shape)
                self.downsample[0].sign = downsample_sign.reshape(weight_shape)

            # self.conv1.weight = self.shared_weight.conv1

            weight_shape = self.shared_weight.conv1.shape
            weight = self.shared_weight.conv1.reshape(-1)
            weight_min = weight.min()
            weight_max = weight.max()
            for i in range(self.stage):
                temp_shift = weight.detach() * self.conv1_shift_affine[i]
                temp_sign = weight.detach() * self.conv1_sign_affine[i]
                index = (weight >= (weight_min + i * (weight_max - weight_min) / self.stage)) & (weight < (weight_min + (i+1) * (weight_max - weight_min) / self.stage))
                temp_shift = temp_shift * index
                temp_sign = temp_sign * index

                if i == 0:
                    conv1_shift = temp_shift
                    conv1_sign = temp_sign
                else:
                    conv1_shift = conv1_shift + temp_shift
                    conv1_sign = conv1_sign + temp_sign
                    del temp_shift
                    del temp_sign

            # self.conv1.shift = self.shared_weight.conv1
            # self.conv1.sign = self.shared_weight.conv1

            self.conv1.shift = conv1_shift.reshape(weight_shape)
            self.conv1.sign = conv1_sign.reshape(weight_shape)

            residual = self.downsample(x)
            x = self.conv1(x)
            x = self.peg(x)
            x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        # self.conv2.weight = self.shared_weight.conv2

        weight_shape = self.shared_weight.conv2.shape
        weight = self.shared_weight.conv2.reshape(-1)
        weight_min = weight.min()
        weight_max = weight.max()
        for i in range(self.stage):
            temp_shift = weight.detach() * self.conv2_shift_affine[i]
            temp_sign = weight.detach() * self.conv2_sign_affine[i]
            index = (weight >= (weight_min + i * (weight_max - weight_min) / self.stage)) & (weight < (weight_min + (i+1) * (weight_max - weight_min) / self.stage))
            temp_shift = temp_shift * index
            temp_sign = temp_sign * index

            if i == 0:
                conv2_shift = temp_shift
                conv2_sign = temp_sign
            else:
                conv2_shift = conv2_shift + temp_shift
                conv2_sign = conv2_sign + temp_sign
                del temp_shift
                del temp_sign

        # self.conv2.shift = self.shared_weight.conv2
        # self.conv2.sign = self.shared_weight.conv2

        self.conv2.shift = conv2_shift.reshape(weight_shape)
        self.conv2.sign = conv2_sign.reshape(weight_shape)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)
        if self.aa is not None:
            x = self.aa(x)

        # self.conv3.weight = self.shared_weight.conv3

        weight_shape = self.shared_weight.conv3.shape
        weight = self.shared_weight.conv3.reshape(-1)
        weight_min = weight.min()
        weight_max = weight.max()
        for i in range(self.stage):
            temp_shift = weight.detach() * self.conv3_shift_affine[i]
            temp_sign = weight.detach() * self.conv3_sign_affine[i]
            index = (weight >= (weight_min + i * (weight_max - weight_min) / self.stage)) & (weight < (weight_min + (i+1) * (weight_max - weight_min) / self.stage))
            temp_shift = temp_shift * index
            temp_sign = temp_sign * index

            if i == 0:
                conv3_shift = temp_shift
                conv3_sign = temp_sign
            else:
                conv3_shift = conv3_shift + temp_shift
                conv3_sign = conv3_sign + temp_sign
                del temp_shift
                del temp_sign

        # self.conv3.shift = self.shared_weight.conv3
        # self.conv3.sign = self.shared_weight.conv3

        self.conv3.shift = conv3_shift.reshape(weight_shape)
        self.conv3.sign = conv3_sign.reshape(weight_shape)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x += residual
        x = self.act3(x)

        return x, 0