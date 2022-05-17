# HAT: Hardware-Aware Transformers for Efficient Natural Language Processing
# Hanrui Wang, Zhanghao Wu, Zhijian Liu, Han Cai, Ligeng Zhu, Chuang Gan and Song Han
# The 58th Annual Meeting of the Association for Computational Linguistics (ACL), 2020.
# Paper: https://arxiv.org/abs/2005.14187
# Project page: https://hanruiwang.me/project_pages/hat/

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    SuperFairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    MultiheadAttentionSuper,
    EmbeddingSuper,
    LinearSuper,
    LayerNormSuper,
    LightweightConv,
    LightweightAdd,
    LightweightShiftadd,
    MultiBranch,
)

import fairseq.init as init

from deepshift.modules import LinearShift, LinearShift_super

def shift_kernel_super(in_features, out_features):
    shift_linear = LinearShift_super(
        in_features,
        out_features,
        bias=True,
        freeze_sign=False,
        use_kernel=False,
        use_cuda=True,
        rounding='deterministic',
        weight_bits=5)
    return shift_linear

import random


DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model('transformersuper_shiftadd_v2')
class TransformerSuperModel_shiftadd_v2(SuperFairseqEncoderDecoderModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    @classmethod
    def hub_models(cls):
        # fmt: off
        return {
            'transformer.wmt14.en-fr': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2',
            'transformer.wmt16.en-de': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2',
            'transformer.wmt18.en-de': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt18.en-de.ensemble.tar.gz',
            'transformer.wmt19.en-de': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.ensemble.tar.gz',
            'transformer.wmt19.en-ru': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.ensemble.tar.gz',
            'transformer.wmt19.de-en': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.ensemble.tar.gz',
            'transformer.wmt19.ru-en': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.ensemble.tar.gz',
            'transformer.wmt19.en-de.single_model': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.single_model.tar.gz',
            'transformer.wmt19.en-ru.single_model': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.single_model.tar.gz',
            'transformer.wmt19.de-en.single_model': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.single_model.tar.gz',
            'transformer.wmt19.ru-en.single_model': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.single_model.tar.gz',
        }
        # fmt: on

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--get-attn', action='store_true', default=False)
        # fmt: on
        parser.add_argument('--no-decoder-final-norm', action='store_true',
                            help='don\'t add an extra layernorm after the last decoder block')

        # SuperTransformer
        # embedding dim
        parser.add_argument('--encoder-embed-choice', nargs='+', default=[512, 256, 128], type=int)
        parser.add_argument('--decoder-embed-choice', nargs='+', default=[512, 256, 128], type=int)

        # number of layers
        parser.add_argument('--encoder-layer-num-choice', nargs='+', default=[7, 6, 5, 4, 3, 2], type=int)
        parser.add_argument('--decoder-layer-num-choice', nargs='+', default=[7, 6, 5, 4, 3, 2], type=int)

        # FFN inner size
        parser.add_argument('--encoder-ffn-embed-dim-choice', nargs='+', default=[4096, 3072, 2048, 1024], type=int)
        parser.add_argument('--decoder-ffn-embed-dim-choice', nargs='+', default=[4096, 3072, 2048, 1024], type=int)

        # number of heads
        parser.add_argument('--encoder-self-attention-heads-choice', nargs='+', default=[16, 8, 4, 2, 1], type=int)
        parser.add_argument('--decoder-self-attention-heads-choice', nargs='+', default=[16, 8, 4, 2, 1], type=int)
        parser.add_argument('--decoder-ende-attention-heads-choice', nargs='+', default=[16, 8, 4, 2, 1], type=int)

        # types of blocks
        # parser.add_argument('--block-types', nargs='+', default=['lightweight_conv', 'lightweight_add', 'lightweight_shiftadd',
        #                             'self_attention', 'self_attention+lightweight_conv', 'self_attention+lightweight_add', 'self_attention+lightweight_shiftadd'])
        parser.add_argument('--encoder-block-types', nargs='+', default=['lightweight_conv', 'lightweight_add', 'lightweight_shiftadd',
                                    'self_attention', 'self_attention+lightweight_conv', 'self_attention+lightweight_add', 'self_attention+lightweight_shiftadd'])
        parser.add_argument('--decoder-block-types', nargs='+', default=[
                                    'self_attention', 'self_attention+lightweight_conv', 'self_attention+lightweight_add', 'self_attention+lightweight_shiftadd'])

        # qkv dim
        parser.add_argument('--qkv-dim', type=int, default=None)

        # arbitrary-ende-attn
        parser.add_argument('--decoder-arbitrary-ende-attn-choice', nargs='+', default=[-1, 1, 2], type=int, help='-1 means only attend to the last layer; 1 means attend to last two layers, 2 means attend to last three layers')

        parser.add_argument('--vocab-original-scaling', action='store_true', default=False)


        # for SubTransformer
        parser.add_argument('--encoder-embed-dim-subtransformer', type=int, help='subtransformer encoder embedding dimension',
                            default=None)
        parser.add_argument('--decoder-embed-dim-subtransformer', type=int, help='subtransformer decoder embedding dimension',
                            default=None)

        parser.add_argument('--encoder-ffn-embed-dim-all-subtransformer', nargs='+', default=None, type=int)
        parser.add_argument('--decoder-ffn-embed-dim-all-subtransformer', nargs='+', default=None, type=int)

        parser.add_argument('--encoder-layer-num-subtransformer', type=int, help='subtransformer num encoder layers')
        parser.add_argument('--decoder-layer-num-subtransformer', type=int, help='subtransformer num decoder layers')

        parser.add_argument('--encoder-self-attention-heads-all-subtransformer', nargs='+', default=None, type=int)
        parser.add_argument('--decoder-self-attention-heads-all-subtransformer', nargs='+', default=None, type=int)
        parser.add_argument('--decoder-ende-attention-heads-all-subtransformer', nargs='+', default=None, type=int)

        parser.add_argument('--decoder-arbitrary-ende-attn-all-subtransformer', nargs='+', default=None, type=int)

        parser.add_argument('--encoder-block-types-all-subtransformer', nargs='+', default=None)
        parser.add_argument('--decoder-block-types-all-subtransformer', nargs='+', default=None)

        """LightConv and DynamicConv arguments"""
        # parser.add_argument(
        #     "--encoder-kernel-size-list",
        #     type=lambda x: utils.eval_str_list(x, int),
        #     help='list of kernel size (default: "[3,7,15,31,31,31,31]")',
        # )
        # parser.add_argument(
        #     "--decoder-kernel-size-list",
        #     type=lambda x: utils.eval_str_list(x, int),
        #     help='list of kernel size (default: "[3,7,15,31,31,31]")',
        # )
        parser.add_argument('--decoder-kernel-size-list', nargs='*', default=[3, 7, 15, 31, 31, 31, 31], type=int)
        parser.add_argument('--encoder-kernel-size-list', nargs='*', default=[3, 7, 15, 31, 31, 31, 31], type=int)
        parser.add_argument('--encoder_ffn_types', nargs='*', default=['mlp_ffn', 'shift_ffn', 'shift_ffn', 'shift_ffn', 'shift_ffn', 'mlp_ffn'], type=str)
        parser.add_argument('--decoder_ffn_types', nargs='*', default=['mlp_ffn', 'shift_ffn', 'shift_ffn', 'shift_ffn', 'shift_ffn', 'mlp_ffn'], type=str)
        parser.add_argument(
            "--encoder-glu", type=utils.eval_bool, help="glu after in proj"
        )
        parser.add_argument(
            "--decoder-glu", type=utils.eval_bool, help="glu after in proj"
        )
        # parser.add_argument(
        #     "--encoder-conv-type",
        #     default="dynamic",
        #     type=str,
        #     choices=["dynamic", "lightweight"],
        #     help="type of convolution",
        # )
        # parser.add_argument(
        #     "--decoder-conv-type",
        #     default="dynamic",
        #     type=str,
        #     choices=["dynamic", "lightweight"],
        #     help="type of convolution",
        # )
        parser.add_argument("--weight-softmax", default=True, type=utils.eval_bool)
        parser.add_argument(
            "--weight-dropout",
            type=float,
            metavar="D",
            help="dropout probability for conv weights",
        )
        # multi branches
        parser.add_argument('--encoder-branch-type', nargs='+', default=None, type=str,
                            help='type of branches type:kernel:dim:head')
        parser.add_argument('--decoder-branch-type', nargs='+', default=None, type=str,
                            help='type of branches type:kernel:dim:head')
        parser.add_argument('--conv-linear', default=False, action='store_true')

        parser.add_argument('--act_path', default=1, type=int)

        # mode
        parser.add_argument('--my_mode', default='test', type=str)

        # quantize
        # parser.add_argument('--num_bits', default=8, type=int)
        # parser.add_argument('--num_bits_grad', default=8, type=int)


    def profile(self, mode=True):
        for module in self.modules():
            if hasattr(module, 'profile') and self != module:
                module.profile(mode)

    def get_sampled_params_numel(self, config):
        self.set_sample_config(config)
        numels = []

        # codebook
        profile_encoder_block_types = config['encoder']['encoder_block_types']
        profile_decoder_block_types = config['decoder']['decoder_block_types']
        profile_encoder_block_types = [profile_encoder_block_types[i] if not isinstance(profile_encoder_block_types[i], list) else profile_encoder_block_types[i][0] for i in range(len(profile_encoder_block_types))]
        profile_decoder_block_types = [profile_decoder_block_types[i] if not isinstance(profile_decoder_block_types[i], list) else profile_decoder_block_types[i][0] for i in range(len(profile_decoder_block_types))]
        print('profile_encoder_block_types: ', profile_encoder_block_types)
        print('profile_decoder_block_types: ', profile_decoder_block_types)

        profile_blocklist = {
                                'lightweight_conv': ['self_attn', 'self_attention', 'lightweight_add', 'lightweight_shiftadd'],
                                'lightweight_add':  ['self_attn', 'self_attention', 'lightweight_conv', 'lightweight_shiftadd'],
                                'lightweight_shiftadd': ['self_attn', 'self_attention', 'lightweight_conv', 'lightweight_add'],
                                'self_attention': ['lightweight'],
                                'self_attention+lightweight_conv': ['self_attention', 'lightweight_add', 'lightweight_shiftadd', '.lightweight'],
                                'self_attention+lightweight_add':  ['self_attention', 'lightweight_conv', 'lightweight_shiftadd', '.lightweight'],
                                'self_attention+lightweight_shiftadd': ['self_attention', 'lightweight_conv', 'lightweight_add', '.lightweight']
                            }

        def if_block(index, name):
            blocklist = profile_blocklist[index]
            for i, item in enumerate(blocklist):
                if item in name:
                    return True
            return False

        for name, module in self.named_modules():
            if hasattr(module, 'calc_sampled_param_num'):
                # a hacky way to skip the layers that exceed encoder-layer-num or decoder-layer-num
                if name.split('.')[0] == 'encoder' and eval(name.split('.')[2]) >= config['encoder']['encoder_layer_num']:
                    continue
                if name.split('.')[0] == 'decoder' and eval(name.split('.')[2]) >= config['decoder']['decoder_layer_num']:
                    continue

                # a hacky way to skip the layers that are not in subTransformer
                if name.split('.')[0] == 'encoder' and if_block(profile_encoder_block_types[eval(name.split('.')[2])], name):
                    continue
                if name.split('.')[0] == 'decoder' and if_block(profile_decoder_block_types[eval(name.split('.')[2])], name):
                    continue

                print(name)

                numels.append(module.calc_sampled_param_num())
        return sum(numels)


    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        init.build_init(args)

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        return TransformerSuperModel_shiftadd_v2(encoder, decoder)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoder_shiftadd_v2(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder_shiftadd_v2(args, tgt_dict, embed_tokens)


class TransformerEncoder_shiftadd_v2(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        # the configs of super arch
        self.super_embed_dim = args.encoder_embed_dim
        self.super_ffn_embed_dim = [args.encoder_ffn_embed_dim] * args.encoder_layers
        self.super_layer_num = args.encoder_layers
        self.super_self_attention_heads = [args.encoder_attention_heads] * args.encoder_layers

        # shift_ffn
        # if args.shift_ffn:
        #     self.ffn_types = []
        #     for i in range(self.super_layer_num):
        #         if i == 0 or i == (self.super_layer_num - 1):
        #             self.ffn_types.append('mlp_ffn')
        #         else:
        #             self.ffn_types.append('shift_ffn')
        # else:
        #     self.ffn_types = []
        #     for i in range(self.super_layer_num):
        #         self.ffn_types.append('mlp_ffn')

        if args.shift_ffn:
            self.ffn_types = args.encoder_ffn_types
        else:
            self.ffn_types = []
            for i in range(self.super_layer_num):
                self.ffn_types.append('mlp_ffn')

        print('FFN types (Encoder): ', self.ffn_types)

        # for shiftadd
        # self.block_types = [args.block_types] * args.encoder_layers
        self.block_types = [args.encoder_block_types] * args.encoder_layers

        self.super_dropout = args.dropout
        self.super_activation_dropout = getattr(args, 'activation_dropout', 0)

        self.super_embed_scale = math.sqrt(self.super_embed_dim)

        self.encoder_kernel_size_list = args.encoder_kernel_size_list

        # the configs of current sampled arch
        self.sample_embed_dim = None
        self.sample_ffn_embed_dim = None
        self.sample_layer_num = None
        self.sample_self_attention_heads = None

        self.sample_dropout = None
        self.sample_activation_dropout = None

        self.sample_embed_scale = None

        self.register_buffer('version', torch.Tensor([3]))

        # self.dropout = args.dropout

        # embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        # self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, self.super_embed_dim, self.padding_idx,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer_shiftadd_v2(args, layer_idx=i, kernel_size=self.encoder_kernel_size_list[i], ffn_type=self.ffn_types[i])
            for i in range(self.super_layer_num)
        ])

        if args.encoder_normalize_before:
            self.layer_norm = LayerNormSuper(self.super_embed_dim)
        else:
            self.layer_norm = None

        self.vocab_original_scaling = args.vocab_original_scaling

        # self.sample_scale = self.embed_scale


    def set_sample_config(self, config:dict):

        # print(config)

        self.sample_embed_dim = config['encoder']['encoder_embed_dim']

        # Caution: this is a list for all layers
        self.sample_ffn_embed_dim = config['encoder']['encoder_ffn_embed_dim']

        self.sample_layer_num = config['encoder']['encoder_layer_num']

        # Caution: this is a list for all layers
        self.sample_self_attention_heads = config['encoder']['encoder_self_attention_heads']

        # shiftadd
        self.block_types = config['encoder']['encoder_block_types']

        # print(self.block_types)

        self.sample_dropout = calc_dropout(self.super_dropout, self.sample_embed_dim, self.super_embed_dim)
        self.sample_activation_dropout = calc_dropout(self.super_activation_dropout, self.sample_embed_dim, self.super_embed_dim)

        self.sample_embed_scale = math.sqrt(self.sample_embed_dim) if not self.vocab_original_scaling else self.super_embed_scale

        self.embed_tokens.set_sample_config(sample_embed_dim=self.sample_embed_dim, part='encoder')

        if self.layer_norm is not None:
            self.layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)

        for i, layer in enumerate(self.layers):
            # not exceed sample layer number
            if i < self.sample_layer_num:
                layer.set_sample_config(is_identity_layer=False,
                                        sample_embed_dim=self.sample_embed_dim,
                                        sample_ffn_embed_dim_this_layer=self.sample_ffn_embed_dim[i],
                                        sample_self_attention_heads_this_layer=self.sample_self_attention_heads[i],
                                        sample_dropout=self.sample_dropout,
                                        sample_activation_dropout=self.sample_activation_dropout,
                                        block_types=self.block_types[i] if not isinstance(self.block_types[i], list) else self.block_types[i][0])
            # exceeds sample layer number
            else:
                layer.set_sample_config(is_identity_layer=True)


    def forward(self, src_tokens, src_lengths, num_bits=-1, num_bits_grad=-1):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        x = self.sample_embed_scale * self.embed_tokens(src_tokens, part='encoder')
        if self.embed_positions is not None:
            positions = self.embed_positions(src_tokens)

            # sample the positional embedding and add
            x += positions[..., :self.sample_embed_dim]
        x = F.dropout(x, p=self.sample_dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        all_x = []
        # encoder layers
        for layer in self.layers:
            # print('\n   input: ', x.shape)
            x = layer(x, encoder_padding_mask, num_bits=num_bits, num_bits_grad=num_bits_grad)
            all_x.append(x)


        if self.layer_norm:
            x = self.layer_norm(x)

        return {
                'encoder_out': x,
                'encoder_out_all' : all_x,
                'encoder_padding_mask': encoder_padding_mask,
        }


    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)

        # need to reorder each layer of output
        if 'encoder_out_all' in encoder_out.keys():
            new_encoder_out_all = []
            for encoder_out_one_layer in encoder_out['encoder_out_all']:
                new_encoder_out_all.append(encoder_out_one_layer.index_select(1, new_order))
            encoder_out['encoder_out_all'] = new_encoder_out_all

        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)
        for i in range(len(self.layers)):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(state_dict, "{}.layers.{}".format(name, i))

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class TransformerDecoder_shiftadd_v2(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(dictionary)

        # the configs of super arch
        self.super_embed_dim = args.decoder_embed_dim
        self.super_ffn_embed_dim = [args.decoder_ffn_embed_dim] * args.decoder_layers
        self.super_layer_num = args.decoder_layers
        self.super_self_attention_heads = [args.decoder_attention_heads] * args.decoder_layers
        self.super_ende_attention_heads = [args.decoder_attention_heads] * args.decoder_layers
        self.super_arbitrary_ende_attn = [-1] * args.decoder_layers


        # if args.shift_ffn:
        #     self.ffn_types = []
        #     for i in range(self.super_layer_num):
        #         if i == 0 or i == (self.super_layer_num - 1):
        #             self.ffn_types.append('mlp_ffn')
        #         else:
        #             self.ffn_types.append('shift_ffn')
        # else:
        #     self.ffn_types = []
        #     for i in range(self.super_layer_num):
        #         self.ffn_types.append('mlp_ffn')

        if args.shift_ffn:
            self.ffn_types = args.decoder_ffn_types
        else:
            self.ffn_types = []
            for i in range(self.super_layer_num):
                self.ffn_types.append('mlp_ffn')

        print('FFN types (Decoder): ', self.ffn_types)

        # for shiftadd
        # self.block_types = [args.block_types] * args.decoder_layers
        self.block_types = [args.decoder_block_types] * args.decoder_layers

        self.super_dropout = args.dropout
        self.super_activation_dropout = getattr(args, 'activation_dropout', 0)

        self.super_embed_scale = math.sqrt(self.super_embed_dim)

        self.decoder_kernel_size_list = args.decoder_kernel_size_list

        # the configs of current sampled arch
        self.sample_embed_dim = None
        self.sample_ffn_embed_dim = None
        self.sample_layer_num = None
        self.sample_self_attention_heads = None
        self.sample_ende_attention_heads = None
        self.sample_arbitrary_ende_attn = None

        self.sample_dropout = None
        self.sample_activation_dropout = None

        self.sample_embed_scale = None


        # the configs of current sampled arch
        self.register_buffer('version', torch.Tensor([3]))

        self.share_input_output_embed = args.share_decoder_input_output_embed

        self.output_embed_dim = args.decoder_output_dim

        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens


        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, self.super_embed_dim, padding_idx,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer_shiftadd_v2(args, layer_idx=i, no_encoder_attn=no_encoder_attn, kernel_size=self.decoder_kernel_size_list[i], ffn_type=self.ffn_types[i])
            for i in range(self.super_layer_num)
        ])

        self.adaptive_softmax = None

        self.project_out_dim = Linear(self.super_embed_dim, self.output_embed_dim, bias=False) \
            if self.super_embed_dim != self.output_embed_dim and not args.tie_adaptive_weights else None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), self.output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)

        if args.decoder_normalize_before and not getattr(args, 'no_decoder_final_norm', False):
            self.layer_norm = LayerNormSuper(self.super_embed_dim)
        else:
            self.layer_norm = None
        self.get_attn = args.get_attn

        self.vocab_original_scaling = args.vocab_original_scaling

    def set_sample_config(self, config:dict):

        self.sample_embed_dim = config['decoder']['decoder_embed_dim']
        self.sample_encoder_embed_dim = config['encoder']['encoder_embed_dim']

        # Caution: this is a list for all layers
        self.sample_ffn_embed_dim = config['decoder']['decoder_ffn_embed_dim']

        # Caution: this is a list for all layers
        self.sample_self_attention_heads = config['decoder']['decoder_self_attention_heads']

        # Caution: this is a list for all layers
        self.sample_ende_attention_heads = config['decoder']['decoder_ende_attention_heads']

        self.sample_arbitrary_ende_attn = config['decoder']['decoder_arbitrary_ende_attn']

        self.sample_layer_num = config['decoder']['decoder_layer_num']

        # shiftadd
        self.block_types = config['decoder']['decoder_block_types']

        self.sample_dropout = calc_dropout(self.super_dropout, self.sample_embed_dim, self.super_embed_dim)
        self.sample_activation_dropout = calc_dropout(self.super_activation_dropout, self.sample_embed_dim, self.super_embed_dim)

        self.sample_embed_scale = math.sqrt(self.sample_embed_dim) if not self.vocab_original_scaling else self.super_embed_scale

        self.embed_tokens.set_sample_config(sample_embed_dim=self.sample_embed_dim, part='decoder')

        if self.layer_norm is not None:
            self.layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)

        for i, layer in enumerate(self.layers):
            # not exceed sample layer number
            if i < self.sample_layer_num:
                layer.set_sample_config(is_identity_layer=False,
                                        sample_embed_dim=self.sample_embed_dim,
                                        sample_encoder_embed_dim=self.sample_encoder_embed_dim,
                                        sample_ffn_embed_dim_this_layer=self.sample_ffn_embed_dim[i],
                                        sample_self_attention_heads_this_layer=self.sample_self_attention_heads[i],
                                        sample_ende_attention_heads_this_layer=self.sample_ende_attention_heads[i],
                                        sample_dropout=self.sample_dropout,
                                        sample_activation_dropout=self.sample_activation_dropout,
                                        block_types=self.block_types[i] if not isinstance(self.block_types[i], list) else self.block_types[i][0])
            # exceeds sample layer number
            else:
                layer.set_sample_config(is_identity_layer=True)



    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, num_bits=-1, num_bits_grad=-1, **unused):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(prev_output_tokens, encoder_out, incremental_state, num_bits=num_bits, num_bits_grad=num_bits_grad)
        x = self.output_layer(x)
        return x, extra

    def extract_features(self, prev_output_tokens, encoder_out=None, incremental_state=None, num_bits=-1, num_bits_grad=-1, **unused):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if positions is not None:
            positions = positions[..., :self.sample_embed_dim]

        if incremental_state is not None:
            # only take the last token in to the decoder
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.sample_embed_scale * self.embed_tokens(prev_output_tokens, part='decoder')

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.sample_dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None
        attns = []
        inner_states = [x]

        # decoder layers
        for i, layer in enumerate(self.layers):
            encoder_out_feed = None
            encoder_padding_mask_feed = None

            if encoder_out is not None:
                # only use the last layer
                if i >= self.sample_layer_num or self.sample_arbitrary_ende_attn[i] == -1:
                    encoder_out_feed = encoder_out['encoder_out']
                # concat one second last output layer
                elif self.sample_arbitrary_ende_attn[i] == 1:
                    encoder_out_feed = torch.cat([encoder_out['encoder_out'], encoder_out['encoder_out_all'][-2]], dim=0)
                elif self.sample_arbitrary_ende_attn[i] == 2:
                    encoder_out_feed = torch.cat([encoder_out['encoder_out'], encoder_out['encoder_out_all'][-2], encoder_out['encoder_out_all'][-3]], dim=0)
                else:
                    raise NotImplementedError("arbitrary_ende_attn should in [-1, 1, 2]")

            if encoder_out['encoder_padding_mask'] is not None:
                if i >= self.sample_layer_num or self.sample_arbitrary_ende_attn[i] == -1:
                    encoder_padding_mask_feed = encoder_out['encoder_padding_mask']
                # concat one more
                elif self.sample_arbitrary_ende_attn[i] == 1:
                    encoder_padding_mask_feed = torch.cat([encoder_out['encoder_padding_mask'], encoder_out['encoder_padding_mask']], dim=1)
                # concat two more
                elif self.sample_arbitrary_ende_attn[i] == 2:
                    encoder_padding_mask_feed = torch.cat([encoder_out['encoder_padding_mask'], encoder_out['encoder_padding_mask'], encoder_out['encoder_padding_mask']], dim=1)
                else:
                    raise NotImplementedError("arbitrary_ende_attn should in [-1, 1, 2]")

            # print('\n   input: ', x.shape)
            x, attn = layer(
                x,
                encoder_out_feed,
                encoder_padding_mask_feed,
                incremental_state,
                self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
                num_bits=num_bits, num_bits_grad=num_bits_grad
            )
            inner_states.append(x)
            attns.append(attn)


        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)
        if not self.get_attn:
            attns = attns[-1]
        return x, {'attn': attns, 'inner_states': inner_states}

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                return F.linear(features, self.embed_tokens.sampled_weight('decoder'))
            else:
                return F.linear(features, self.embed_out[:, :self.sample_embed_dim])
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device or self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)

        for i in range(len(self.layers)):
            # update layer norms
            layer_norm_map = {
                '0': 'self_attn_layer_norm',
                '1': 'encoder_attn_layer_norm',
                '2': 'final_layer_norm'
            }
            for old, new in layer_norm_map.items():
                for m in ('weight', 'bias'):
                    k = '{}.layers.{}.layer_norms.{}.{}'.format(name, i, old, m)
                    if k in state_dict:
                        state_dict['{}.layers.{}.{}.{}'.format(name, i, new, m)] = state_dict[k]
                        del state_dict[k]

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict


class TransformerEncoderLayer_shiftadd_v2(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args, layer_idx, kernel_size=0, ffn_type='mlp_ffn'):
        super().__init__()

        self.my_mode = args.my_mode

        self.ffn_type = ffn_type

        # the configs of super arch
        self.super_embed_dim = args.encoder_embed_dim
        self.super_ffn_embed_dim_this_layer = args.encoder_ffn_embed_dim
        self.super_self_attention_heads_this_layer = args.encoder_attention_heads

        # for shiftadd
        # self.block_types = args.block_types
        self.block_types = args.encoder_block_types

        self.super_dropout = args.dropout
        self.super_activation_dropout = getattr(args, 'activation_dropout', 0)

        # the configs of current sampled arch
        self.sample_embed_dim = None
        self.sample_ffn_embed_dim_this_layer = None
        self.sample_self_attention_heads_this_layer = None

        self.sample_dropout = None
        self.sample_activation_dropout = None

        self.is_identity_layer = None

        self.qkv_dim=args.qkv_dim

        # lightweight_conv
        assert self.block_types[0] == 'lightweight_conv'
        # self.conv_dim = self.qkv_dim
        self.conv_dim = self.super_embed_dim
        self.conv_out_dim = self.super_embed_dim
        padding_l = (
            kernel_size // 2
            if kernel_size % 2 == 1
            else ((kernel_size - 1) // 2, kernel_size // 2)
        )
        self.lightweight_conv = LightweightConv(
            self.conv_dim,
            kernel_size,
            padding_l=padding_l,
            weight_softmax=args.weight_softmax,
            num_heads=self.super_self_attention_heads_this_layer,
            weight_dropout=args.weight_dropout,
            with_linear=args.conv_linear,
            out_dim=self.conv_out_dim
        )
        # lightweight_add
        assert self.block_types[1] == 'lightweight_add'
        self.lightweight_add = LightweightAdd(
            self.conv_dim,
            kernel_size,
            padding_l=padding_l,
            weight_softmax=args.weight_softmax,
            num_heads=args.encoder_attention_heads,
            weight_dropout=args.weight_dropout,
            with_linear=args.conv_linear,
            out_dim=self.conv_out_dim
        )
        # lightweight_shiftadd
        assert self.block_types[2] == 'lightweight_shiftadd'
        self.lightweight_shiftadd = LightweightShiftadd(
            self.conv_dim,
            kernel_size,
            padding_l=padding_l,
            weight_softmax=args.weight_softmax,
            num_heads=args.encoder_attention_heads,
            weight_dropout=args.weight_dropout,
            with_linear=args.conv_linear,
            out_dim=self.conv_out_dim
        )
        # self_attention
        assert self.block_types[3] == 'self_attention'
        self.self_attention = MultiheadAttentionSuper(
            super_embed_dim=self.super_embed_dim, num_heads=self.super_self_attention_heads_this_layer, is_encoder=True,
            dropout=args.attention_dropout, self_attention=True, qkv_dim=self.qkv_dim
        )
        # self_attention+lightweight_conv
        assert self.block_types[4] == 'self_attention+lightweight_conv'
        self._self_attn_lightweight_conv = []
        embed_dims = []
        heads = []
        num_types = len(args.encoder_branch_type)
        for layer_type in args.encoder_branch_type:
            embed_dims.append(int(layer_type.split(':')[2]))
            heads.append(int(layer_type.split(':')[3]))
            self._self_attn_lightweight_conv.append(self.get_layer(args, kernel_size, int(embed_dims[-1] / 2), heads[-1], layer_type, _type='conv'))
        self.self_attn_lightweight_conv = MultiBranch(self._self_attn_lightweight_conv, embed_dims)
        # self_attention+lightweight_add
        assert self.block_types[5] == 'self_attention+lightweight_add'
        self._self_attn_lightweight_add = []
        embed_dims = []
        heads = []
        num_types = len(args.encoder_branch_type)
        for layer_type in args.encoder_branch_type:
            embed_dims.append(int(layer_type.split(':')[2]))
            heads.append(int(layer_type.split(':')[3]))
            self._self_attn_lightweight_add.append(self.get_layer(args, kernel_size, int(embed_dims[-1] / 2), heads[-1], layer_type, _type='add'))
        self.self_attn_lightweight_add = MultiBranch(self._self_attn_lightweight_add, embed_dims)
        # self_attention+lightweight_shiftadd
        assert self.block_types[6] == 'self_attention+lightweight_shiftadd'
        self._self_attn_lightweight_shiftadd = []
        embed_dims = []
        heads = []
        num_types = len(args.encoder_branch_type)
        for layer_type in args.encoder_branch_type:
            embed_dims.append(int(layer_type.split(':')[2]))
            heads.append(int(layer_type.split(':')[3]))
            self._self_attn_lightweight_shiftadd.append(self.get_layer(args, kernel_size, int(embed_dims[-1] / 2), heads[-1], layer_type, _type='shiftadd'))
        self.self_attn_lightweight_shiftadd = MultiBranch(self._self_attn_lightweight_shiftadd, embed_dims)

        self.multi_branch_heads = heads

        # super block
        self.super_block = [
            self.lightweight_conv,
            self.lightweight_add,
            self.lightweight_shiftadd,
            self.self_attention,
            self.self_attn_lightweight_conv,
            self.self_attn_lightweight_add,
            self.self_attn_lightweight_shiftadd
        ]

        self.idx = [i for i in range(7)]
        self.act_path = args.act_path

        self.self_attn_layer_norm = LayerNormSuper(self.super_embed_dim)
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu')
        )
        self.normalize_before = args.encoder_normalize_before

        if self.ffn_type == 'mlp_ffn':
            print('use normal ffn!')
            if args.encoder_glu:
                self.fc1 = LinearSuper(super_in_dim=self.super_embed_dim, super_out_dim=self.super_ffn_embed_dim_this_layer, uniform_=init.uniform_, non_linear='glu')
            else:
                self.fc1 = LinearSuper(super_in_dim=self.super_embed_dim, super_out_dim=self.super_ffn_embed_dim_this_layer, uniform_=init.uniform_, non_linear='relu')
            self.fc2 = LinearSuper(super_in_dim=self.super_ffn_embed_dim_this_layer, super_out_dim=self.super_embed_dim, uniform_=init.uniform_, non_linear='linear')
        elif self.ffn_type == 'shift_ffn':
            print('use shift ffn!')
            self.fc1 = shift_kernel_super(self.super_embed_dim, self.super_ffn_embed_dim_this_layer)
            self.fc2 = shift_kernel_super(self.super_ffn_embed_dim_this_layer, self.super_embed_dim)
        else:
            print('wrong ffn type!')
            exit()

        self.final_layer_norm = LayerNormSuper(self.super_embed_dim)

    def get_layer(self, args, conv_kernel_size, out_dim, num_heads, layer_type, _type='conv'):
        kernel_size = layer_type.split(':')[1]
        if kernel_size == 'default':
            kernel_size = conv_kernel_size
        else:
            kernel_size = int(kernel_size)

        padding_l = kernel_size // 2 if kernel_size % 2 == 1 else ((kernel_size - 1) // 2, kernel_size // 2)

        conv_dim = int(self.super_embed_dim / 2)

        if 'lightweight' in layer_type:
            if _type == 'conv':
                layer = LightweightConv(
                    conv_dim, kernel_size, padding_l=padding_l, weight_softmax=args.weight_softmax,
                    num_heads=num_heads,  weight_dropout=args.weight_dropout,
                    with_linear=args.conv_linear, out_dim=out_dim,
                )
            elif _type == 'add':
                layer = LightweightAdd(
                    conv_dim, kernel_size, padding_l=padding_l, weight_softmax=args.weight_softmax,
                    num_heads=num_heads,  weight_dropout=args.weight_dropout,
                    with_linear=args.conv_linear, out_dim=out_dim,
                )
            elif _type == 'shiftadd':
                layer = LightweightShiftadd(
                    conv_dim, kernel_size, padding_l=padding_l, weight_softmax=args.weight_softmax,
                    num_heads=num_heads,  weight_dropout=args.weight_dropout,
                    with_linear=args.conv_linear, out_dim=out_dim,
                )
            else:
                print('wrong type!!!!')
                exit()
        elif 'attn' in layer_type:
            layer = MultiheadAttentionSuper(
                super_embed_dim=out_dim, num_heads=int(self.super_self_attention_heads_this_layer/2), is_encoder=True,
                dropout=args.attention_dropout, self_attention=True, qkv_dim=conv_dim
            )
        else:
            print('wrong type!!!!')
            exit()
        return layer


    def set_sample_config(self, is_identity_layer, sample_embed_dim=None, sample_ffn_embed_dim_this_layer=None, sample_self_attention_heads_this_layer=None,
                    sample_dropout=None, sample_activation_dropout=None, block_types=None):

        if is_identity_layer:
            self.is_identity_layer = True
            return

        self.is_identity_layer = False

        self.sample_embed_dim = sample_embed_dim
        self.sample_ffn_embed_dim_this_layer = sample_ffn_embed_dim_this_layer
        self.sample_self_attention_heads_this_layer = sample_self_attention_heads_this_layer

        self.block_types = block_types

        self.sample_dropout = sample_dropout
        self.sample_activation_dropout = sample_activation_dropout

        self.self_attn_layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)

        self.self_attention.set_sample_config(sample_q_embed_dim=self.sample_embed_dim, sample_attention_heads=self.sample_self_attention_heads_this_layer)

        self.fc1.set_sample_config(sample_in_dim=self.sample_embed_dim, sample_out_dim=self.sample_ffn_embed_dim_this_layer)
        self.fc2.set_sample_config(sample_in_dim=self.sample_ffn_embed_dim_this_layer, sample_out_dim=self.sample_embed_dim)

        self.final_layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)

        self.multi_branch_heads[0] = self.sample_self_attention_heads_this_layer

        # self.super_block[0].set_sample_config(sample_in_dim=self.qkv_dim, sample_out_dim=sample_embed_dim)
        # self.super_block[1].set_sample_config(sample_in_dim=self.qkv_dim, sample_out_dim=sample_embed_dim)
        # self.super_block[2].set_sample_config(sample_in_dim=self.qkv_dim, sample_out_dim=sample_embed_dim)
        self.super_block[0].set_sample_config(sample_in_dim=self.sample_embed_dim, sample_out_dim=sample_embed_dim)
        self.super_block[1].set_sample_config(sample_in_dim=self.sample_embed_dim, sample_out_dim=sample_embed_dim)
        self.super_block[2].set_sample_config(sample_in_dim=self.sample_embed_dim, sample_out_dim=sample_embed_dim)
        self.super_block[3].set_sample_config(sample_q_embed_dim=self.sample_embed_dim, sample_attention_heads=self.sample_self_attention_heads_this_layer)
        self.super_block[4].set_sample_config(qkv_dim=self.qkv_dim, sample_embed_dim=sample_embed_dim, sample_attention_heads_list=self.multi_branch_heads)
        self.super_block[5].set_sample_config(qkv_dim=self.qkv_dim, sample_embed_dim=sample_embed_dim, sample_attention_heads_list=self.multi_branch_heads)
        self.super_block[6].set_sample_config(qkv_dim=self.qkv_dim, sample_embed_dim=sample_embed_dim, sample_attention_heads_list=self.multi_branch_heads)


        codebook = {
            'lightweight_conv'    : 0,
            'lightweight_add'     : 1,
            'lightweight_shiftadd': 2,
            'self_attention'      : 3,
            'self_attention+lightweight_conv'     : 4,
            'self_attention+lightweight_add'      : 5,
            'self_attention+lightweight_shiftadd' : 6,
        }
        # print(self.block_types)
        self.sample_idx = codebook[self.block_types]
        # self.super_block = self.super_block[idx]

        # print(self.sample_idx)

        # print(self.my_mode)


    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {
            '0': 'self_attn_layer_norm',
            '1': 'final_layer_norm'
        }
        for old, new in layer_norm_map.items():
            for m in ('weight', 'bias'):
                k = '{}.layer_norms.{}.{}'.format(name, old, m)
                if k in state_dict:
                    state_dict[
                        '{}.{}.{}'.format(name, new, m)
                    ] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask, num_bits=-1, num_bits_grad=-1, attn_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape (T_tgt, T_src), where
            T_tgt is the length of query, while T_src is the length of key,
            though here both query and key is x here,
            attn_mask[t_tgt, t_src] = 1 means when calculating embedding
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if self.is_identity_layer:
            return x
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.byte(), -1e8)
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        # TODO: to formally solve this problem, we need to change fairseq's
        # MultiheadAttention. We will do this later on.

        # x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)

        # super block
        # print(x.shape, self.sample_embed_dim)
        # print(self.super_block)

        # print(self.my_mode)

        if self.my_mode == 'train':
            random.shuffle(self.idx)
            res = []
            for i in range(self.act_path):
                activate = self.idx[i]
                # print('activate {}-th path'.format(activate))
                # print(self.super_block[activate])
                if activate < 3:
                    # lightweight conv/add/shiftadd
                    res.append(self.super_block[activate](x, num_bits=num_bits, num_bits_grad=num_bits_grad))
                else:
                    # self_attn or multi-branch
                    res.append(self.super_block[activate](query=x, key=x, value=x, key_padding_mask=encoder_padding_mask, num_bits=num_bits, num_bits_grad=num_bits_grad)[0])
            # print(res[0].shape)
            # print(res[1].shape)
            x = res[0]
            for i in range(1, self.act_path):
                x += res[i]

            x /= self.act_path
        elif self.my_mode == 'test':
            activate = self.sample_idx
            # print('encoder | activate {}-th path'.format(activate))
            if activate < 3:
                x = self.super_block[activate](x, num_bits=num_bits, num_bits_grad=num_bits_grad)
            else:
                x = self.super_block[activate](query=x, key=x, value=x, key_padding_mask=encoder_padding_mask, num_bits=num_bits, num_bits_grad=num_bits_grad)[0]
        else:
            print('wrong mode, choose from [train, test]')
            exit()

        x = F.dropout(x, p=self.dropout, training=self.training)
        x[:residual.size(0),:,:] = residual + x[:residual.size(0),:,:]
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x, num_bits=num_bits, num_bits_grad=num_bits_grad))
        # x = self.activation_fn(self.fc1(x), inplace=False)
        x = F.dropout(x, p=self.sample_activation_dropout, training=self.training)
        x = self.fc2(x, num_bits=num_bits, num_bits_grad=num_bits_grad)
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        return x

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x


class TransformerDecoderLayer_shiftadd_v2(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, layer_idx, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False, kernel_size=0, ffn_type='mlp_ffn'):
        super().__init__()

        self.my_mode = args.my_mode

        self.ffn_type = ffn_type

        self.add_bias_kv = add_bias_kv
        self.add_zero_attn = add_zero_attn

        # the configs of super arch
        self.super_embed_dim = args.decoder_embed_dim
        self.super_encoder_embed_dim = args.encoder_embed_dim
        self.super_ffn_embed_dim_this_layer = args.decoder_ffn_embed_dim
        self.super_self_attention_heads_this_layer = args.decoder_attention_heads
        self.super_ende_attention_heads_this_layer = args.decoder_attention_heads

        # for shiftadd
        # self.block_types = args.block_types
        self.block_types = args.decoder_block_types

        self.super_dropout = args.dropout
        self.super_activation_dropout = getattr(args, 'activation_dropout', 0)

        # the configs of current sampled arch
        self.sample_embed_dim = None
        self.sample_encoder_embed_dim = None
        self.sample_ffn_embed_dim_this_layer = None
        self.sample_self_attention_heads_this_layer = None
        self.sample_ende_attention_heads_this_layer = None

        self.sample_dropout = None
        self.sample_activation_dropout = None

        self.is_identity_layer = None


        self.qkv_dim = args.qkv_dim


        self.layer_idx = layer_idx

        # # lightweight_conv
        # assert args.block_types[0] == 'lightweight_conv'
        # # self.conv_dim = self.qkv_dim
        # self.conv_dim = self.super_embed_dim
        # self.conv_out_dim = self.super_embed_dim
        # padding_l = (
        #     kernel_size // 2
        #     if kernel_size % 2 == 1
        #     else ((kernel_size - 1) // 2, kernel_size // 2)
        # )
        # self.lightweight_conv = LightweightConv(
        #     self.conv_dim,
        #     kernel_size,
        #     padding_l=padding_l,
        #     weight_softmax=args.weight_softmax,
        #     num_heads=self.super_self_attention_heads_this_layer,
        #     weight_dropout=args.weight_dropout,
        #     with_linear=args.conv_linear,
        #     out_dim=self.conv_out_dim
        # )
        # # lightweight_add
        # assert args.block_types[1] == 'lightweight_add'
        # self.lightweight_add = LightweightAdd(
        #     self.conv_dim,
        #     kernel_size,
        #     padding_l=padding_l,
        #     weight_softmax=args.weight_softmax,
        #     num_heads=args.encoder_attention_heads,
        #     weight_dropout=args.weight_dropout,
        #     with_linear=args.conv_linear,
        #     out_dim=self.conv_out_dim
        # )
        # # lightweight_shiftadd
        # assert args.block_types[2] == 'lightweight_shiftadd'
        # self.lightweight_shiftadd = LightweightShiftadd(
        #     self.conv_dim,
        #     kernel_size,
        #     padding_l=padding_l,
        #     weight_softmax=args.weight_softmax,
        #     num_heads=args.encoder_attention_heads,
        #     weight_dropout=args.weight_dropout,
        #     with_linear=args.conv_linear,
        #     out_dim=self.conv_out_dim
        # )


        # self_attention
        assert self.block_types[0] == 'self_attention'
        self.self_attention = MultiheadAttentionSuper(
            super_embed_dim=self.super_embed_dim,
            num_heads=self.super_self_attention_heads_this_layer,
            is_encoder=False,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=True,
            qkv_dim=self.qkv_dim
        )
        # self_attention+lightweight_conv
        assert self.block_types[1] == 'self_attention+lightweight_conv'
        self._self_attn_lightweight_conv = []
        embed_dims = []
        heads = []
        num_types = len(args.decoder_branch_type)
        for layer_type in args.decoder_branch_type:
            embed_dims.append(int(layer_type.split(':')[2]))
            heads.append(int(layer_type.split(':')[3]))
            self._self_attn_lightweight_conv.append(self.get_layer(args, kernel_size, int(embed_dims[-1] / 2), heads[-1], layer_type, _type='conv'))
        self.self_attn_lightweight_conv = MultiBranch(self._self_attn_lightweight_conv, embed_dims)
        # self_attention+lightweight_add
        assert self.block_types[2] == 'self_attention+lightweight_add'
        self._self_attn_lightweight_add = []
        embed_dims = []
        heads = []
        num_types = len(args.decoder_branch_type)
        for layer_type in args.decoder_branch_type:
            embed_dims.append(int(layer_type.split(':')[2]))
            heads.append(int(layer_type.split(':')[3]))
            self._self_attn_lightweight_add.append(self.get_layer(args, kernel_size, int(embed_dims[-1] / 2), heads[-1], layer_type, _type='add'))
        self.self_attn_lightweight_add = MultiBranch(self._self_attn_lightweight_add, embed_dims)
        # self_attention+lightweight_shiftadd
        assert self.block_types[3] == 'self_attention+lightweight_shiftadd'
        self._self_attn_lightweight_shiftadd = []
        embed_dims = []
        heads = []
        num_types = len(args.decoder_branch_type)
        for layer_type in args.decoder_branch_type:
            embed_dims.append(int(layer_type.split(':')[2]))
            heads.append(int(layer_type.split(':')[3]))
            self._self_attn_lightweight_shiftadd.append(self.get_layer(args, kernel_size, int(embed_dims[-1] / 2), heads[-1], layer_type, _type='shiftadd'))
        self.self_attn_lightweight_shiftadd = MultiBranch(self._self_attn_lightweight_shiftadd, embed_dims)

        self.multi_branch_heads = heads

        # super block
        self.super_block = [
            # self.lightweight_conv,
            # self.lightweight_add,
            # self.lightweight_shiftadd,
            self.self_attention,
            self.self_attn_lightweight_conv,
            self.self_attn_lightweight_add,
            self.self_attn_lightweight_shiftadd
        ]

        # self.idx = [i for i in range(7)]
        self.idx = [i for i in range(4)]
        self.act_path = args.act_path

        # self.self_attn = MultiheadAttentionSuper(
        #     is_encoder=False,
        #     super_embed_dim=self.super_embed_dim,
        #     num_heads=self.super_self_attention_heads_this_layer,
        #     dropout=args.attention_dropout,
        #     add_bias_kv=add_bias_kv,
        #     add_zero_attn=add_zero_attn,
        #     self_attention=True,
        #     qkv_dim=self.qkv_dim
        # )
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu')
        )
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, 'char_inputs', False)
        self.self_attn_layer_norm = LayerNormSuper(self.super_embed_dim)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = MultiheadAttentionSuper(
                super_embed_dim=self.super_embed_dim,
                num_heads=self.super_ende_attention_heads_this_layer,
                is_encoder=False,
                super_kdim=self.super_encoder_embed_dim,
                super_vdim=self.super_encoder_embed_dim,
                dropout=args.attention_dropout,
                encoder_decoder_attention=True,
                qkv_dim=self.qkv_dim
            )
            self.encoder_attn_layer_norm = LayerNormSuper(self.super_embed_dim)

        if self.ffn_type == 'mlp_ffn':
            print('use normal ffn!')
            if args.decoder_glu:
                self.fc1 = LinearSuper(super_in_dim=self.super_embed_dim, super_out_dim=self.super_ffn_embed_dim_this_layer,
                                   uniform_=init.uniform_, non_linear='glu')
            else:
                self.fc1 = LinearSuper(super_in_dim=self.super_embed_dim, super_out_dim=self.super_ffn_embed_dim_this_layer,
                                   uniform_=init.uniform_, non_linear='relu')
            self.fc2 = LinearSuper(super_in_dim=self.super_ffn_embed_dim_this_layer, super_out_dim=self.super_embed_dim,
                                   uniform_=init.uniform_, non_linear='linear')
        elif self.ffn_type == 'shift_ffn':
            print('use shift ffn!')
            self.fc1 = shift_kernel_super(self.super_embed_dim, self.super_ffn_embed_dim_this_layer)
            self.fc2 = shift_kernel_super(self.super_ffn_embed_dim_this_layer, self.super_embed_dim)
        else:
            print('wrong ffn type!')
            exit()

        self.final_layer_norm = LayerNormSuper(self.super_embed_dim)
        self.need_attn = True

        self.onnx_trace = False

    def get_layer(self, args, conv_kernel_size, out_dim, num_heads, layer_type, _type='conv'):
        kernel_size = layer_type.split(':')[1]
        if kernel_size == 'default':
            kernel_size = conv_kernel_size
        else:
            kernel_size = int(kernel_size)

        # padding_l = kernel_size // 2 if kernel_size % 2 == 1 else ((kernel_size - 1) // 2, kernel_size // 2)
        padding_l = kernel_size - 1

        conv_dim = int(self.super_embed_dim / 2)

        if 'lightweight' in layer_type:
            if _type == 'conv':
                layer = LightweightConv(
                    conv_dim, kernel_size, padding_l=padding_l, weight_softmax=args.weight_softmax,
                    num_heads=num_heads,  weight_dropout=args.weight_dropout,
                    with_linear=args.conv_linear, out_dim=out_dim,
                )
            elif _type == 'add':
                layer = LightweightAdd(
                    conv_dim, kernel_size, padding_l=padding_l, weight_softmax=args.weight_softmax,
                    num_heads=num_heads,  weight_dropout=args.weight_dropout,
                    with_linear=args.conv_linear, out_dim=out_dim,
                )
            elif _type == 'shiftadd':
                layer = LightweightShiftadd(
                    conv_dim, kernel_size, padding_l=padding_l, weight_softmax=args.weight_softmax,
                    num_heads=num_heads,  weight_dropout=args.weight_dropout,
                    with_linear=args.conv_linear, out_dim=out_dim,
                )
            else:
                print('wrong type!!!!')
                exit()
        elif 'attn' in layer_type:
            layer = MultiheadAttentionSuper(
                super_embed_dim=out_dim, num_heads=int(self.super_self_attention_heads_this_layer/2), is_encoder=False,
                dropout=args.attention_dropout, add_bias_kv=self.add_bias_kv,
                add_zero_attn=self.add_zero_attn, self_attention=True, qkv_dim=conv_dim
            )
        else:
            print('wrong type!!!!')
            exit()
        return layer


    def set_sample_config(self, is_identity_layer, sample_embed_dim=None, sample_encoder_embed_dim=None, sample_ffn_embed_dim_this_layer=None,
        sample_self_attention_heads_this_layer=None, sample_ende_attention_heads_this_layer=None, sample_dropout=None, sample_activation_dropout=None, block_types=None):

        if is_identity_layer:
            self.is_identity_layer = True
            return

        self.is_identity_layer = False

        self.sample_embed_dim = sample_embed_dim
        self.sample_encoder_embed_dim = sample_encoder_embed_dim
        self.sample_ffn_embed_dim_this_layer = sample_ffn_embed_dim_this_layer
        self.sample_self_attention_heads_this_layer = sample_self_attention_heads_this_layer
        self.sample_ende_attention_heads_this_layer = sample_ende_attention_heads_this_layer

        self.sample_dropout = sample_dropout
        self.sample_activation_dropout = sample_activation_dropout


        self.self_attn_layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)
        self.encoder_attn_layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)

        self.self_attention.set_sample_config(sample_q_embed_dim=self.sample_embed_dim, sample_attention_heads=self.sample_self_attention_heads_this_layer)
        self.encoder_attn.set_sample_config(sample_q_embed_dim=self.sample_embed_dim, sample_kv_embed_dim=self.sample_encoder_embed_dim, sample_attention_heads=self.sample_ende_attention_heads_this_layer)

        self.fc1.set_sample_config(sample_in_dim=self.sample_embed_dim, sample_out_dim=self.sample_ffn_embed_dim_this_layer)
        self.fc2.set_sample_config(sample_in_dim=self.sample_ffn_embed_dim_this_layer, sample_out_dim=self.sample_embed_dim)

        self.final_layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)

        self.multi_branch_heads[0] = self.sample_self_attention_heads_this_layer

        # self.super_block[0].set_sample_config(sample_in_dim=self.sample_embed_dim, sample_out_dim=sample_embed_dim)
        # self.super_block[1].set_sample_config(sample_in_dim=self.sample_embed_dim, sample_out_dim=sample_embed_dim)
        # self.super_block[2].set_sample_config(sample_in_dim=self.sample_embed_dim, sample_out_dim=sample_embed_dim)
        self.super_block[0].set_sample_config(sample_q_embed_dim=self.sample_embed_dim, sample_attention_heads=self.sample_self_attention_heads_this_layer)
        self.super_block[1].set_sample_config(qkv_dim=self.qkv_dim, sample_embed_dim=sample_embed_dim, sample_attention_heads_list=self.multi_branch_heads)
        self.super_block[2].set_sample_config(qkv_dim=self.qkv_dim, sample_embed_dim=sample_embed_dim, sample_attention_heads_list=self.multi_branch_heads)
        self.super_block[3].set_sample_config(qkv_dim=self.qkv_dim, sample_embed_dim=sample_embed_dim, sample_attention_heads_list=self.multi_branch_heads)

        self.block_types = block_types

        codebook = {
            # 'lightweight_conv'    : 0,
            # 'lightweight_add'     : 1,
            # 'lightweight_shiftadd': 2,
            'self_attention'      : 0,
            'self_attention+lightweight_conv'     : 1,
            'self_attention+lightweight_add'      : 2,
            'self_attention+lightweight_shiftadd' : 3,
        }
        # print(self.block_types)
        self.sample_idx = codebook[self.block_types]


    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(
        self,
        x,
        encoder_out=None,
        encoder_padding_mask=None,
        incremental_state=None,
        prev_self_attn_state=None,
        prev_attn_state=None,
        self_attn_mask=None,
        self_attn_padding_mask=None,
        num_bits=-1,
        num_bits_grad=-1
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if self.is_identity_layer:
            return x, None

        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            self.self_attn._set_input_buffer(incremental_state, saved_state)



        # x, attn = self.self_attn(
        #     query=x,
        #     key=x,
        #     value=x,
        #     key_padding_mask=self_attn_padding_mask,
        #     incremental_state=incremental_state,
        #     need_weights=False,
        #     attn_mask=self_attn_mask,
        # )


        if self.my_mode == 'train':
            random.shuffle(self.idx)
            res = []
            for i in range(self.act_path):
                activate = self.idx[i]
                # print('activate {}-th path'.format(activate))
                # print(self.super_block[activate])

                # self_attn or multi-branch
                res.append(self.super_block[activate](query=x, key=x, value=x, key_padding_mask=self_attn_padding_mask,
                                                      incremental_state=incremental_state,
                                                      need_weights=False,
                                                      attn_mask=self_attn_mask,
                                                      num_bits=num_bits, num_bits_grad=num_bits_grad)[0])
            # print(res[0].shape)
            # print(res[1].shape)
            x = res[0]
            for i in range(1, self.act_path):
                x += res[i]

            x /= self.act_path
        elif self.my_mode == 'test':
            activate = self.sample_idx
            # print('decoder | activate {}-th path'.format(activate))

            x = self.super_block[activate](query=x, key=x, value=x, key_padding_mask=self_attn_padding_mask,
                                                      incremental_state=incremental_state,
                                                      need_weights=False,
                                                      attn_mask=self_attn_mask,
                                                      num_bits=num_bits, num_bits_grad=num_bits_grad)[0]
        else:
            print('wrong mode, choose from [train, test]')
            exit()


        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        if self.encoder_attn is not None:
            residual = x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
            if prev_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_attn_state
                saved_state = {"prev_key": prev_key, "prev_value": prev_value}
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=(not self.training and self.need_attn),
            )
            x = F.dropout(x, p=self.sample_dropout, training=self.training)
            x = residual + x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x, num_bits=num_bits, num_bits_grad=num_bits_grad))
        # x = self.activation_fn(self.fc1(x), inplace=False)
        x = F.dropout(x, p=self.sample_activation_dropout, training=self.training)
        x = self.fc2(x, num_bits=num_bits, num_bits_grad=num_bits_grad)
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            self_attn_state = saved_state["prev_key"], saved_state["prev_value"]
            return x, attn, self_attn_state
        return x, attn

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

def calc_dropout(dropout, sample_embed_dim, super_embed_dim):
    return dropout * 1.0 * sample_embed_dim / super_embed_dim

def Embedding(num_embeddings, embedding_dim, padding_idx):
    return EmbeddingSuper(num_embeddings, embedding_dim, padding_idx=padding_idx)

def Linear(in_features, out_features, bias=True, uniform_=None, non_linear='linear'):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight) if uniform_ is None else uniform_(m.weight, non_linear=non_linear)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m

@register_model_architecture('transformersuper_shiftadd_v2', 'transformersuper_shiftadd_v2')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.adaptive_input = getattr(args, 'adaptive_input', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)


@register_model_architecture('transformersuper_shiftadd_v2', 'transformersuper_shiftadd_v2_iwslt_de_en')
def transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    base_architecture(args)


@register_model_architecture('transformersuper_shiftadd_v2', 'transformersuper_shiftadd_v2_wmt_en_de')
def transformer_wmt_en_de(args):
    base_architecture(args)

@register_model_architecture('transformersuper_shiftadd_v2', 'transformersuper_shiftadd_v2_wmt_en_fr')
def transformer_wmt_en_de(args):
    base_architecture(args)

# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture('transformersuper_shiftadd_v2', 'transformersuper_shiftadd_v2_vaswani_wmt_en_de_big')
def transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.dropout = getattr(args, 'dropout', 0.3)
    base_architecture(args)


@register_model_architecture('transformersuper_shiftadd_v2', 'transformersuper_shiftadd_v2_vaswani_wmt_en_fr_big')
def transformer_vaswani_wmt_en_fr_big(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    transformer_vaswani_wmt_en_de_big(args)


@register_model_architecture('transformersuper_shiftadd_v2', 'transformersuper_shiftadd_v2_wmt_en_de_big')
def transformer_wmt_en_de_big(args):
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    transformer_vaswani_wmt_en_de_big(args)


# default parameters used in tensor2tensor implementation
@register_model_architecture('transformersuper_shiftadd_v2', 'transformersuper_shiftadd_v2_wmt_en_de_big_t2t')
def transformer_wmt_en_de_big_t2t(args):
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', True)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.1)
    transformer_vaswani_wmt_en_de_big(args)
