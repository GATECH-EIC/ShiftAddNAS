import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from . import MultiheadAttention
from . import MultiheadAttentionSuper

class MultiBranch(nn.Module):
    def __init__(self, branches, embed_dim_list):
        super().__init__()
        self.branches = nn.ModuleList(branches)
        self.embed_dim_list = embed_dim_list

    def set_sample_config(self, qkv_dim, sample_embed_dim, sample_attention_heads_list):

        sample_embed_dim = int(sample_embed_dim / 2)
        qkv_dim = int(qkv_dim / 2)

        for idx, embed_dim in enumerate(self.embed_dim_list):
            branch = self.branches[idx]
            branch_type = type(branch)

            sample_attention_heads = sample_attention_heads_list[idx]

            # print(qkv_dim, sample_embed_dim, sample_attention_heads)

            if branch_type == MultiheadAttention or branch_type == MultiheadAttentionSuper:
                branch.set_sample_config(sample_q_embed_dim=sample_embed_dim, sample_attention_heads=sample_attention_heads)
            else:
                # branch.set_sample_config(sample_in_dim=qkv_dim, sample_out_dim=sample_embed_dim)
                branch.set_sample_config(sample_in_dim=sample_embed_dim, sample_out_dim=sample_embed_dim)

    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None, need_weights=True, static_kv=False, attn_mask=None, num_bits=-1, num_bits_grad=-1):
        tgt_len, bsz, embed_size = query.size()

        # print(query.size())
        # assert sum(self.embed_dim_list) == embed_size
        out = []
        attn = None
        start = 0
        for idx, embed_dim in enumerate(self.embed_dim_list):
            branch = self.branches[idx]
            branch_type = type(branch)

            # new added
            # print('ori embed_dim: ', embed_dim)
            embed_dim = int(embed_size / 2)
            # embed_dim = 256
            # print('new embed_dim: ', embed_dim)

            q = query[...,start:start+embed_dim]
            if key is not None:
                assert value is not None
                k, v = key[..., start:start+embed_dim], value[..., start:start+embed_dim]
            start += embed_dim

            # print(branch_type)

            if branch_type == MultiheadAttention or branch_type == MultiheadAttentionSuper:
                # print('--- start MultiBranch')
                # print('q, k, v: ', q.shape, k.shape, v.shape)
                x, attn = branch(q, k, v, key_padding_mask, incremental_state, need_weights, static_kv, attn_mask, num_bits=num_bits, num_bits_grad=num_bits_grad)
                # print('attn out: ', x.shape)
                # print('attn out: ', attn.shape)
            else:
                mask = key_padding_mask
                if mask is not None:
                    q = q.masked_fill(mask.transpose(0, 1).unsqueeze(2), 0)
                x = branch(q.contiguous(), incremental_state=incremental_state, num_bits=num_bits, num_bits_grad=num_bits_grad)
                # print('Lighweight out: ', x.shape)
            out.append(x)

        out = torch.cat(out, dim=-1)
        # print('MultiBranch out: ', out.size())
        return out, attn