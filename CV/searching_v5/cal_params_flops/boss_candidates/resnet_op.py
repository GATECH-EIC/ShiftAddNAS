# modified from timm/models/resnet.py
import math
import torch.nn as nn
from timm.models.layers import create_attn, AvgPool2dSame

from .bot_op import PEG

from cal_params_flops.boss_candidates.adder import adder
from cal_params_flops.boss_candidates.deepshift import modules, modules_q

class ResConv(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 attn_layer=None, aa_layer=None, drop_block=None, drop_path=None, avg_down=False):
        super(ResConv, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        # act_layer = nn.SiLU

        if avg_down and (stride == 2 or inplanes != outplanes):
            avg_stride = stride if dilation == 1 else 1
            if stride == 1 and dilation == 1:
                pool = nn.Identity()
            else:
                avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
                pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

            self.downsample = nn.Sequential(*[
                pool,
                nn.Conv2d(inplanes, outplanes, 1, stride=1, padding=0, bias=False),
                norm_layer(outplanes)
            ])
        else:
            if stride == 2:
                self.downsample = nn.Sequential(
                    nn.Conv2d(inplanes, outplanes, 3, stride=2, padding=1, bias=False),
                    norm_layer(outplanes),
                    act_layer(inplace=True)
                )
            elif inplanes != outplanes:
                self.downsample = nn.Sequential(
                    nn.Conv2d(inplanes, outplanes, 1, stride=1, padding=0, bias=False),
                    norm_layer(outplanes),
                    act_layer(inplace=True)
                )
            else:
                self.downsample = nn.Identity()

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.peg = PEG(first_planes, stride=stride)
        self.bn1 = norm_layer(first_planes)

        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            first_planes, width, kernel_size=3, stride=1,
            padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        self.bn2 = norm_layer(width)
        self.act2 = act_layer(inplace=True)
        self.aa = aa_layer(channels=width, stride=stride) if use_aa else None

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        if attn_layer == 'se':
            self.se = create_attn(attn_layer, outplanes)
        else:
            self.se = None

        self.act3 = act_layer(inplace=True)
        # self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path
        self.inc = inplanes

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = self.downsample(x)
        x = self.conv1(x)
        x = self.peg(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)
        if self.aa is not None:
            x = self.aa(x)

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

        return x


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
        self.conv = modules.Conv2dShift(dim, dim, kernel_size=3, stride=stride, padding=1, groups=dim, weight_bits=5, threshold=None, quant_bits=16)

    def forward(self, x):
        # x = x + self.conv(x)
        return self.conv(x)


class PEG_Shift_Q(nn.Module):
    """
    modified from PEG in https://github.com/Meituan-AutoML/CPVT
    """
    def __init__(self, dim, stride):
        super(PEG_Shift_Q, self).__init__()
        # self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=stride, padding=1, groups=dim)
        self.conv = modules_q.Conv2dShiftQ(dim, dim, kernel_size=3, stride=stride, padding=1, groups=dim, weight_bits=5, threshold=None, quant_bits=16)

    def forward(self, x):
        # x = x + self.conv(x)
        return self.conv(x)



class ResAdd(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 attn_layer=None, aa_layer=None, drop_block=None, drop_path=None, avg_down=False):
        super(ResAdd, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        # act_layer = nn.SiLU

        if avg_down and (stride == 2 or inplanes != outplanes):
            avg_stride = stride if dilation == 1 else 1
            if stride == 1 and dilation == 1:
                pool = nn.Identity()
            else:
                avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
                pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

            self.downsample = nn.Sequential(*[
                pool,
                # nn.Conv2d(inplanes, outplanes, 1, stride=1, padding=0, bias=False),
                adder.Adder2D(inplanes, outplanes, 1, stride=1, padding=0, bias=False),
                norm_layer(outplanes)
            ])
        else:
            if stride == 2:
                self.downsample = nn.Sequential(
                    # nn.Conv2d(inplanes, outplanes, 3, stride=2, padding=1, bias=False),
                    adder.Adder2D(inplanes, outplanes, 3, stride=2, padding=1, bias=False),
                    norm_layer(outplanes),
                    act_layer(inplace=True)
                )
            elif inplanes != outplanes:
                self.downsample = nn.Sequential(
                    # nn.Conv2d(inplanes, outplanes, 1, stride=1, padding=0, bias=False),
                    adder.Adder2D(inplanes, outplanes, 1, stride=1, padding=0, bias=False),
                    norm_layer(outplanes),
                    act_layer(inplace=True)
                )
            else:
                self.downsample = nn.Identity()

        # self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.conv1 = adder.Adder2D(inplanes, first_planes, kernel_size=1, bias=False)
        # self.peg = PEG(first_planes, stride=stride)
        self.peg = PEG_Add(first_planes, stride=stride)
        self.bn1 = norm_layer(first_planes)

        self.act1 = act_layer(inplace=True)

        # self.conv2 = nn.Conv2d(
        #     first_planes, width, kernel_size=3, stride=1,
        #     padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        self.conv2 = adder.Adder2D(
            first_planes, width, kernel_size=3, stride=1,
            padding=first_dilation, groups=cardinality, bias=False) ### assume dilation is always 1 !
        self.bn2 = norm_layer(width)
        self.act2 = act_layer(inplace=True)
        self.aa = aa_layer(channels=width, stride=stride) if use_aa else None

        # self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.conv3 = adder.Adder2D(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        if attn_layer == 'se':
            self.se = create_attn(attn_layer, outplanes)
        else:
            self.se = None

        self.act3 = act_layer(inplace=True)
        # self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path
        self.inc = inplanes

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = self.downsample(x)
        x = self.conv1(x)
        x = self.peg(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)
        if self.aa is not None:
            x = self.aa(x)

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

        return x


class ResShift(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 attn_layer=None, aa_layer=None, drop_block=None, drop_path=None, avg_down=False):
        super(ResShift, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        # act_layer = nn.SiLU

        if avg_down and (stride == 2 or inplanes != outplanes):
            avg_stride = stride if dilation == 1 else 1
            if stride == 1 and dilation == 1:
                pool = nn.Identity()
            else:
                avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
                pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

            self.downsample = nn.Sequential(*[
                pool,
                # nn.Conv2d(inplanes, outplanes, 1, stride=1, padding=0, bias=False),
                modules.Conv2dShift(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=False, weight_bits=5, threshold=None, quant_bits=16),
                norm_layer(outplanes)
            ])
        else:
            if stride == 2:
                self.downsample = nn.Sequential(
                    # nn.Conv2d(inplanes, outplanes, 3, stride=2, padding=1, bias=False),
                    modules.Conv2dShift(inplanes, outplanes, 3, stride=2, padding=1, bias=False, weight_bits=5, threshold=None, quant_bits=16),
                    norm_layer(outplanes),
                    act_layer(inplace=True)
                )
            elif inplanes != outplanes:
                self.downsample = nn.Sequential(
                    # nn.Conv2d(inplanes, outplanes, 1, stride=1, padding=0, bias=False),
                    modules.Conv2dShift(inplanes, outplanes, 1, stride=1, padding=0, bias=False, weight_bits=5, threshold=None, quant_bits=16),
                    norm_layer(outplanes),
                    act_layer(inplace=True)
                )
            else:
                self.downsample = nn.Identity()

        # self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.conv1 = modules.Conv2dShift(inplanes, first_planes, kernel_size=1, bias=False, weight_bits=5, threshold=None, quant_bits=16)
        # self.peg = PEG(first_planes, stride=stride)
        self.peg = PEG_Shift(first_planes, stride=stride)
        self.bn1 = norm_layer(first_planes)

        self.act1 = act_layer(inplace=True)

        # self.conv2 = nn.Conv2d(
        #     first_planes, width, kernel_size=3, stride=1,
        #     padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        self.conv2 = modules.Conv2dShift(
            first_planes, width, kernel_size=3, stride=1,
            padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False, weight_bits=5, threshold=None, quant_bits=16)
        self.bn2 = norm_layer(width)
        self.act2 = act_layer(inplace=True)
        self.aa = aa_layer(channels=width, stride=stride) if use_aa else None

        # self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.conv3 = modules.Conv2dShift(width, outplanes, kernel_size=1, bias=False, weight_bits=5, threshold=None, quant_bits=16)
        self.bn3 = norm_layer(outplanes)

        if attn_layer == 'se':
            self.se = create_attn(attn_layer, outplanes)
        else:
            self.se = None

        self.act3 = act_layer(inplace=True)
        # self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path
        self.inc = inplanes

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = self.downsample(x)
        x = self.conv1(x)
        x = self.peg(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)
        if self.aa is not None:
            x = self.aa(x)

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

        return x



class ResShift_Q(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 attn_layer=None, aa_layer=None, drop_block=None, drop_path=None, avg_down=False):
        super(ResShift_Q, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        # act_layer = nn.SiLU

        if avg_down and (stride == 2 or inplanes != outplanes):
            avg_stride = stride if dilation == 1 else 1
            if stride == 1 and dilation == 1:
                pool = nn.Identity()
            else:
                avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
                pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

            self.downsample = nn.Sequential(*[
                pool,
                # nn.Conv2d(inplanes, outplanes, 1, stride=1, padding=0, bias=False),
                modules_q.Conv2dShiftQ(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=False, weight_bits=5, threshold=None, quant_bits=16),
                norm_layer(outplanes)
            ])
        else:
            if stride == 2:
                self.downsample = nn.Sequential(
                    # nn.Conv2d(inplanes, outplanes, 3, stride=2, padding=1, bias=False),
                    modules_q.Conv2dShiftQ(inplanes, outplanes, 3, stride=2, padding=1, bias=False, weight_bits=5, threshold=None, quant_bits=16),
                    norm_layer(outplanes),
                    act_layer(inplace=True)
                )
            elif inplanes != outplanes:
                self.downsample = nn.Sequential(
                    # nn.Conv2d(inplanes, outplanes, 1, stride=1, padding=0, bias=False),
                    modules_q.Conv2dShiftQ(inplanes, outplanes, 1, stride=1, padding=0, bias=False, weight_bits=5, threshold=None, quant_bits=16),
                    norm_layer(outplanes),
                    act_layer(inplace=True)
                )
            else:
                self.downsample = nn.Identity()

        # self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.conv1 = modules_q.Conv2dShiftQ(inplanes, first_planes, kernel_size=1, bias=False, weight_bits=5, threshold=None, quant_bits=16)
        # self.peg = PEG(first_planes, stride=stride)
        self.peg = PEG_Shift_Q(first_planes, stride=stride)
        self.bn1 = norm_layer(first_planes)

        self.act1 = act_layer(inplace=True)

        # self.conv2 = nn.Conv2d(
        #     first_planes, width, kernel_size=3, stride=1,
        #     padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        self.conv2 = modules_q.Conv2dShiftQ(
            first_planes, width, kernel_size=3, stride=1,
            padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False, weight_bits=5, threshold=None, quant_bits=16)
        self.bn2 = norm_layer(width)
        self.act2 = act_layer(inplace=True)
        self.aa = aa_layer(channels=width, stride=stride) if use_aa else None

        # self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.conv3 = modules_q.Conv2dShiftQ(width, outplanes, kernel_size=1, bias=False, weight_bits=5, threshold=None, quant_bits=16)
        self.bn3 = norm_layer(outplanes)

        if attn_layer == 'se':
            self.se = create_attn(attn_layer, outplanes)
        else:
            self.se = None

        self.act3 = act_layer(inplace=True)
        # self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path
        self.inc = inplanes

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = self.downsample(x)
        x = self.conv1(x)
        x = self.peg(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)
        if self.aa is not None:
            x = self.aa(x)

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

        return x
