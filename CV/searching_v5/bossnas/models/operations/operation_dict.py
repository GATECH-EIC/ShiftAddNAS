# modified from https://github.com/changlin31/DNA/blob/master/searching/dna/operations.py
import torch.nn as nn
from .mbconv_ops import InvertedResidual
from timm.models.efficientnet_blocks import ConvBnAct, DepthwiseSeparableConv
from .hytra_ops import ResAtt, ResConv, ResAdd, ResShift
from .hytra_ops_ws import ResConv_ws, ResAdd_ws, ResShift_ws
from .hytra_ops_ws_dist import ResConv_ws_dist, ResAdd_ws_dist, ResShift_ws_dist

# GLOBAL PARAMS
BN_MOMENTUM_PT_DEFAULT = 0.1
BN_EPS_PT_DEFAULT = 1e-5
BN_ARGS_PT = dict(momentum=BN_MOMENTUM_PT_DEFAULT, eps=BN_EPS_PT_DEFAULT)

BN_MOMENTUM_TF_DEFAULT = 1 - 0.99
BN_EPS_TF_DEFAULT = 1e-3
BN_ARGS_TF = dict(momentum=BN_MOMENTUM_TF_DEFAULT, eps=BN_EPS_TF_DEFAULT)

BN_ARGS = BN_ARGS_PT  # Chose bn args here

#  Some params take the default, refer to the original class.
OPS = {
    # MBConv
    'MB6_3x3_se0.25': lambda in_channels, out_channels, stride, downsample:
    InvertedResidual(in_channels, out_channels, 3, stride, act_layer=nn.SiLU, downsample=downsample,
                     exp_ratio=6.0, se_ratio=0.25, norm_kwargs=BN_ARGS),
    'MB6_5x5_se0.25': lambda in_channels, out_channels, stride, downsample:
    InvertedResidual(in_channels, out_channels, 5, stride, act_layer=nn.SiLU, downsample=downsample,
                     exp_ratio=6.0, se_ratio=0.25, norm_kwargs=BN_ARGS),
    'MB3_3x3_se0.25': lambda in_channels, out_channels, stride, downsample:
    InvertedResidual(in_channels, out_channels, 3, stride, act_layer=nn.SiLU, downsample=downsample,
                     exp_ratio=3.0, se_ratio=0.25, norm_kwargs=BN_ARGS),
    'MB3_5x5_se0.25': lambda in_channels, out_channels, stride, downsample:
    InvertedResidual(in_channels, out_channels, 5, stride, act_layer=nn.SiLU, downsample=downsample,
                     exp_ratio=3.0, se_ratio=0.25, norm_kwargs=BN_ARGS),

    # HyTra
    'ResAtt': lambda in_channels, out_channels, fmap_size, head, up_inc:
        ResAtt(dim=in_channels, fmap_size=fmap_size, dim_out=out_channels, proj_factor=4,
               dim_head=64, heads=head, up_dim=up_inc),
    'ResConv': lambda in_channels, out_channels, fmap_size, up_inc:
        ResConv(inplanes=in_channels, planes=out_channels, fmap_size=fmap_size, up_inplanes=up_inc),
    'ResAdd': lambda in_channels, out_channels, fmap_size, up_inc:
        ResAdd(inplanes=in_channels, planes=out_channels, fmap_size=fmap_size, up_inplanes=up_inc),
    'ResShift': lambda in_channels, out_channels, fmap_size, up_inc:
        ResShift(inplanes=in_channels, planes=out_channels, fmap_size=fmap_size, up_inplanes=up_inc),
    'ResAtt_SiLU': lambda in_channels, out_channels, fmap_size, head, up_inc:
        ResAtt(dim=in_channels, fmap_size=fmap_size, dim_out=out_channels, proj_factor=4,
               dim_head=64, heads=head, up_dim=up_inc, act_layer=nn.SiLU(inplace=True)),
    'ResConv_SESiLU': lambda in_channels, out_channels, fmap_size, up_inc:
        ResConv(inplanes=in_channels, planes=out_channels, fmap_size=fmap_size,
                up_inplanes=up_inc, act_layer=nn.SiLU, attn_layer='se'),

    # HyTra weight sharing
    'ResConv_ws': lambda in_channels, out_channels, fmap_size, up_inc, shared_weight:
        ResConv_ws(inplanes=in_channels, planes=out_channels, fmap_size=fmap_size, up_inplanes=up_inc, shared_weight=shared_weight),
    'ResAdd_ws': lambda in_channels, out_channels, fmap_size, up_inc, shared_weight:
        ResAdd_ws(inplanes=in_channels, planes=out_channels, fmap_size=fmap_size, up_inplanes=up_inc, shared_weight=shared_weight),
    'ResShift_ws': lambda in_channels, out_channels, fmap_size, up_inc, shared_weight:
        ResShift_ws(inplanes=in_channels, planes=out_channels, fmap_size=fmap_size, up_inplanes=up_inc, shared_weight=shared_weight),


    # HyTra weight sharing (dost)
    'ResConv_ws_dist': lambda in_channels, out_channels, fmap_size, up_inc, shared_weight:
        ResConv_ws_dist(inplanes=in_channels, planes=out_channels, fmap_size=fmap_size, up_inplanes=up_inc, shared_weight=shared_weight),
    'ResAdd_ws_dist': lambda in_channels, out_channels, fmap_size, up_inc, shared_weight:
        ResAdd_ws_dist(inplanes=in_channels, planes=out_channels, fmap_size=fmap_size, up_inplanes=up_inc, shared_weight=shared_weight),
    'ResShift_ws_dist': lambda in_channels, out_channels, fmap_size, up_inc, shared_weight:
        ResShift_ws_dist(inplanes=in_channels, planes=out_channels, fmap_size=fmap_size, up_inplanes=up_inc, shared_weight=shared_weight),

    # stem
    'Conv3x3_BN_swish': lambda in_channels, out_channels, stride: \
        ConvBnAct(in_channels, out_channels, 3, stride=stride, act_layer=nn.SiLU,
                  norm_kwargs=BN_ARGS),
    'MB1_3x3_se0.25': lambda in_channels, out_channels, stride: \
        DepthwiseSeparableConv(in_channels, out_channels, 3, stride=stride, act_layer=nn.SiLU,
                               se_ratio=0.25, norm_kwargs=BN_ARGS),
}


def reset(m):
    # reset conv2d/linear/BN in Block
    if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
        m.reset_parameters()
