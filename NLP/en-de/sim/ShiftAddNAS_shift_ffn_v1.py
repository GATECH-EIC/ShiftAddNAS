import os
import math
import copy
import csv
import numpy as np

def self_attention(embedding_dim, num_head, qkv_dim=512, idx=0, num_token=30):

    q_op = {
        "idx": idx,
        "type": "FC",
        "batch": num_token,
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "input_H": 1,
        "input_W": 1,
        "input_C": embedding_dim,
        "output_E": 1,
        "output_F": 1,
        "output_M": qkv_dim
    }

    k_op = {
        "idx": idx+1,
        "type": "FC",
        "batch": num_token,
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "input_H": 1,
        "input_W": 1,
        "input_C": embedding_dim,
        "output_E": 1,
        "output_F": 1,
        "output_M": qkv_dim
    }

    v_op = {
        "idx": idx+2,
        "type": "FC",
        "batch": num_token,
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "input_H": 1,
        "input_W": 1,
        "input_C": embedding_dim,
        "output_E": 1,
        "output_F": 1,
        "output_M": qkv_dim
    }

    attn_op = {
        "idx": idx+3,
        "type": "FC",
        "batch": num_token * num_head,
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "input_H": 1,
        "input_W": 1,
        "input_C": qkv_dim // num_head,
        "output_E": 1,
        "output_F": 1,
        "output_M": num_token
    }

    attn_v_op = {
        "idx": idx+4,
        "type": "FC",
        "batch": num_token * num_head,
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "input_H": 1,
        "input_W": 1,
        "input_C": num_token,
        "output_E": 1,
        "output_F": 1,
        "output_M": qkv_dim // num_head
    }

    out_op = {
        "idx": idx+5,
        "type": "FC",
        "batch": num_token,
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "input_H": 1,
        "input_W": 1,
        "input_C": qkv_dim,
        "output_E": 1,
        "output_F": 1,
        "output_M": embedding_dim
    }

    all_op = [q_op, k_op, v_op, attn_op, attn_v_op, out_op]

    return all_op

def mlp(embedding_dim, ffn_dim, idx=0, num_token=30):

    fc1_op = {
        "idx": idx,
        "type": "FC",
        "batch": num_token,
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "input_H": 1,
        "input_W": 1,
        "input_C": embedding_dim,
        "output_E": 1,
        "output_F": 1,
        "output_M": ffn_dim
    }

    fc2_op = {
        "idx": idx+1,
        "type": "FC",
        "batch": num_token,
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "input_H": 1,
        "input_W": 1,
        "input_C": ffn_dim,
        "output_E": 1,
        "output_F": 1,
        "output_M": embedding_dim
    }

    all_op = [fc1_op, fc2_op]

    return all_op

def shift_mlp(embedding_dim, ffn_dim, idx=0, num_token=30):

    fc1_op = {
        "idx": idx,
        "type": "Shift",
        "batch": num_token,
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "input_H": 1,
        "input_W": 1,
        "input_C": embedding_dim,
        "output_E": 1,
        "output_F": 1,
        "output_M": ffn_dim
    }

    fc2_op = {
        "idx": idx+1,
        "type": "Shift",
        "batch": num_token,
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "input_H": 1,
        "input_W": 1,
        "input_C": ffn_dim,
        "output_E": 1,
        "output_F": 1,
        "output_M": embedding_dim
    }

    all_op = [fc1_op, fc2_op]

    return all_op

def lightweight(embedding_dim, num_head, idx=0, num_token=30, kernel_size=3, padding=1):

    fc1_op = {
        "idx": idx,
        "type": "Shift",
        "batch": num_token,
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "input_H": 1,
        "input_W": 1,
        "input_C": embedding_dim,
        "output_E": 1,
        "output_F": 1,
        "output_M": embedding_dim
    }

    conv_op = {
        "idx": idx+1,
        "type": "Conv",
        "batch": embedding_dim // num_head,
        "kernel_size": int(np.sqrt(kernel_size)),
        "stride": 1,
        "padding": padding,
        "input_H": num_token,
        "input_W": 1,
        "input_C": num_head,
        "output_E": num_token,
        "output_F": 1,
        "output_M": num_head
    }

    fc2_op = {
        "idx": idx+2,
        "type": "Shift",
        "batch": num_token,
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "input_H": 1,
        "input_W": 1,
        "input_C": embedding_dim,
        "output_E": 1,
        "output_F": 1,
        "output_M": embedding_dim
    }

    all_op = [fc1_op, conv_op, fc2_op]

    return all_op



def self_attention_ende(embedding_dim, embedding_dim_encoder, num_head_1, num_head_2, qkv_dim=512, idx=0, num_token=30, ende=-1):

    if ende == -1:
        ende_num_token = num_token
    elif ende == 1:
        ende_num_token = num_token * 2
    elif ende == 2:
        ende_num_token = num_token * 3
    else:
        print('wrong arbitrary_ende choice!')
        exit()

    ende_q_op = {
        "idx": idx,
        "type": "FC",
        "batch": num_token,
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "input_H": 1,
        "input_W": 1,
        "input_C": embedding_dim,
        "output_E": 1,
        "output_F": 1,
        "output_M": qkv_dim
    }

    ende_k_op = {
        "idx": idx+1,
        "type": "FC",
        "batch": ende_num_token,
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "input_H": 1,
        "input_W": 1,
        "input_C": embedding_dim_encoder,
        "output_E": 1,
        "output_F": 1,
        "output_M": qkv_dim
    }

    ende_v_op = {
        "idx": idx+2,
        "type": "FC",
        "batch": ende_num_token,
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "input_H": 1,
        "input_W": 1,
        "input_C": embedding_dim_encoder,
        "output_E": 1,
        "output_F": 1,
        "output_M": qkv_dim
    }

    ende_attn_op = {
        "idx": idx+3,
        "type": "FC",
        "batch": num_token * num_head_2,
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "input_H": 1,
        "input_W": 1,
        "input_C": qkv_dim // num_head_2,
        "output_E": 1,
        "output_F": 1,
        "output_M": ende_num_token
    }

    ende_attn_v_op = {
        "idx": idx+4,
        "type": "FC",
        "batch": num_token * num_head_2,
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "input_H": 1,
        "input_W": 1,
        "input_C": ende_num_token,
        "output_E": 1,
        "output_F": 1,
        "output_M": qkv_dim // num_head_2
    }

    ende_out_op = {
        "idx": idx+5,
        "type": "FC",
        "batch": num_token,
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "input_H": 1,
        "input_W": 1,
        "input_C": qkv_dim,
        "output_E": 1,
        "output_F": 1,
        "output_M": embedding_dim
    }

    all_op = [ende_q_op, ende_k_op, ende_v_op, ende_attn_op, ende_attn_v_op, ende_out_op]

    return all_op



def valid_tiling(tiling_factor, FC_flag, num_bits=32):

    if num_bits == 32:
        mem_factor = 1
        cmp_factor = 1
    elif num_bits == 8:
        mem_factor = 2
        cmp_factor = 4

    valid = True
    N3 = tiling_factor["N3"]
    N0 = tiling_factor["N0"]
    N = tiling_factor["N"]
    if not (N == N3*N0):
        valid = False
    valid = True
    M3 = tiling_factor["M3"]
    M2 = tiling_factor["M2"]
    M1 = tiling_factor["M1"]
    M0 = tiling_factor["M0"]
    M = tiling_factor["M"]
    if not (M == M3*M2*M1*M0):
        valid = False
    if not (M0 <= 16 * mem_factor):          # output RF
        valid = False
    C3 = tiling_factor["C3"]
    C2 = tiling_factor["C2"]
    C1 = tiling_factor["C1"]
    C0 = tiling_factor["C0"]
    C = tiling_factor["C"]
    if not (C == C3*C2*C1*C0):
        valid = False
    if not (C3 == 1):
        valid = False
    E1 = tiling_factor["E1"]
    E3 = tiling_factor["E3"]
    E = tiling_factor["E"]
    if not (E == E1*E3):
        valid = False
    if not FC_flag:
        if not (E1 != 1):
            valid = False
    R = tiling_factor["R"]
    S = tiling_factor["S"]
    if not (M1*C1*E1*R < 14*12 * cmp_factor): # number of PEs
        valid = False
    if not (C0*S < 12*mem_factor):          # input RF
        valid = False
    if not (M0*C0*S < 192*5*mem_factor):      # weight RF
        valid = False
    stride = tiling_factor["stride"]
    F = tiling_factor["F"]
    if not (N0*C1*C0*((E1-1)*stride+R)*((F-1)*stride+S) + N0*M2*M1*M0*E1*F < 65536*mem_factor): # SRAM
        valid = False
    return valid

def get_latency(tiling_factor, unit_latency):
    N3 = tiling_factor["N3"]
    N0 = tiling_factor["N0"]
    N = tiling_factor["N"]
    M3 = tiling_factor["M3"]
    M2 = tiling_factor["M2"]
    M1 = tiling_factor["M1"]
    M0 = tiling_factor["M0"]
    M = tiling_factor["M"]
    C3 = tiling_factor["C3"]
    C2 = tiling_factor["C2"]
    C1 = tiling_factor["C1"]
    C0 = tiling_factor["C0"]
    C = tiling_factor["C"]
    E1 = tiling_factor["E1"]
    E3 = tiling_factor["E3"]
    E = tiling_factor["E"]
    R = tiling_factor["R"]
    S = tiling_factor["S"]
    stride = tiling_factor["stride"]
    F = tiling_factor["F"]

    latency = N3*N0*M3*M2*M0*C3*C2*C0*E3*F*S*unit_latency
    return latency

def get_energy(tiling_factor, unit_energy):
    N3 = tiling_factor["N3"]
    N0 = tiling_factor["N0"]
    N = tiling_factor["N"]
    M3 = tiling_factor["M3"]
    M2 = tiling_factor["M2"]
    M1 = tiling_factor["M1"]
    M0 = tiling_factor["M0"]
    M = tiling_factor["M"]
    C3 = tiling_factor["C3"]
    C2 = tiling_factor["C2"]
    C1 = tiling_factor["C1"]
    C0 = tiling_factor["C0"]
    C = tiling_factor["C"]
    E1 = tiling_factor["E1"]
    E3 = tiling_factor["E3"]
    E = tiling_factor["E"]
    R = tiling_factor["R"]
    S = tiling_factor["S"]
    stride = tiling_factor["stride"]
    F = tiling_factor["F"]

    H = tiling_factor["H"]
    W = tiling_factor["W"]
    num_ifmap = N*H*W*C # input feature map size
    num_weight = R*S*C*M # weight size
    num_ofmap = N*E*F*M # output size

    computation = N*E*F*M*R*S*C
    DRAM_ifmap = M3 * num_ifmap
    DRAM_weight = N3 * E3 * num_weight
    DRAM_ofmap = ( max((2*C3-1), 1) ) * num_ofmap
    GB_ifmap = M3 * M2 * num_ifmap
    GB_ofmap = ( max(2 * C3 * (C2-1), 1) ) * num_ofmap
    NoC_ifmap = M3 * M2 * M1 * R * E / H * num_ifmap
    NoC_weight = N3 * E3 * E1 * num_weight
    NoC_ofmap = ( max(C3 * C2 * (C1 * R - 1), 1) ) * num_ofmap
    RF_ifmap = M3 * M2 * M1 * R * E / H * M0 * S * F / W * num_ifmap
    RF_weight = N3 * E3 * E1 * F * N0 * num_weight
    RF_ofmap = ( max(C3 * C2 * C1 * R * (C0 * S - 1 ) * 2, 1) ) * num_ofmap
    energy = computation * unit_energy["unit_comp"] \
             + (DRAM_ifmap + DRAM_weight + DRAM_ofmap) * unit_energy["unit_DRAM"] \
             + (DRAM_ifmap + DRAM_ofmap) * unit_energy["unit_DRAM_GB"] \
             + (GB_ifmap + GB_ofmap) * unit_energy["unit_GB"] \
             + (NoC_ifmap + NoC_weight) * unit_energy["unit_NoC"] \
             + (NoC_ofmap) * unit_energy["unit_NoC_psum"] \
             + (RF_ifmap + RF_ofmap) * unit_energy["unit_RF"] \
             + (RF_weight) * unit_energy["unit_RF_weight"]
    return energy, [computation * unit_energy["unit_comp"], \
        (DRAM_ifmap + DRAM_weight + DRAM_ofmap) * unit_energy["unit_DRAM"], \
        (DRAM_ifmap + DRAM_ofmap) * unit_energy["unit_DRAM_GB"], \
        (GB_ifmap + GB_ofmap) * unit_energy["unit_GB"], \
        (NoC_ifmap + NoC_weight) * unit_energy["unit_NoC"] + (NoC_ofmap) * unit_energy["unit_NoC_psum"], \
        (RF_ifmap + RF_ofmap) * unit_energy["unit_RF"] + (RF_weight) * unit_energy["unit_RF_weight"], \
        DRAM_ifmap * (unit_energy["unit_DRAM"] + unit_energy["unit_DRAM_GB"]) + GB_ifmap * unit_energy["unit_GB"] + NoC_ifmap * unit_energy["unit_NoC"] + RF_ifmap * unit_energy["unit_RF"], \
        DRAM_weight * unit_energy["unit_DRAM"] + NoC_weight * unit_energy["unit_NoC"] + RF_weight * unit_energy["unit_RF_weight"], \
        DRAM_ofmap * (unit_energy["unit_DRAM"] + unit_energy["unit_DRAM_GB"]) + GB_ofmap * unit_energy["unit_GB"] + NoC_ofmap * unit_energy["unit_NoC_psum"] + RF_ofmap * unit_energy["unit_RF"]        ]


# Refer: https://www.geeksforgeeks.org/print-all-prime-factors-of-a-given-number/
def primeFactors(n):
    prime_list = []
    # Print the number of two's that divide n
    while n % 2 == 0:
        prime_list.append(2)
        n = n / 2

    # n must be odd at this point
    # so a skip of 2 ( i = i + 2) can be used
    for i in range(3,int(math.sqrt(n))+1,2):

        # while i divides n , print i ad divide n
        while n % i== 0:
            prime_list.append(int(i))
            n = n / i

    # Condition if n is a prime
    # number greater than 2
    if n > 2:
        prime_list.append(int(n))
    return prime_list

def possible_mul(x,l):
    if len(l) == 1:
        raw_list = [x*l[0], x*1]
        clean_list = list(dict.fromkeys(raw_list))
        return clean_list
    else:
        raw_list = possible_mul(x*l[0], l[1:]) + possible_mul(x*1, l[1:])
        clean_list = list(dict.fromkeys(raw_list))
        return clean_list

def tile(num, tile_size):
    if tile_size == 1:
        return [[num]]
    else:
        if num == 1:
            prime_list = [1]
        else:
            prime_list = primeFactors(num)

        tile_list = []
        selected_list = possible_mul(1, prime_list)
        for selected in selected_list:
            # select 1 for current the first position
            for options in tile(int(num/selected), tile_size-1):
                to_append = [selected,] + options
                if to_append not in tile_list:
                    tile_list.append(to_append)
        return tile_list

def get_unit_energy(layer_dict, block_type, num_bits=32):

    assert num_bits in [32, 8]

    if num_bits == 32:
        unit_mult = 3.7
        unit_add = 0.9
        unit_shift = 0.13
    elif num_bits == 8:
        unit_mult = 0.2
        unit_add = 0.03
        unit_shift = 0.024

    unit_energy = {}
    if (layer_dict["type"] == "AvgP") or ((layer_dict["type"] == "MaxP")):
        unit_energy["unit_comp"] = 0.0/(1e9) # mJ/MAC
    elif (layer_dict["type"] == "Conv") and (block_type == "self_attention+lightweight_conv"):
        unit_energy["unit_comp"] = unit_mult/(1e9) # mJ/MAC
    elif (layer_dict["type"] == "Conv") and (block_type == "self_attention+lightweight_add"):
        unit_energy["unit_comp"] = unit_add/(1e9) # mJ/MAC
    elif (layer_dict["type"] == "Conv") and (block_type == "self_attention+lightweight_shiftadd"):
        unit_energy["unit_comp"] = unit_add/(1e9) # mJ/MAC
    elif (layer_dict["type"] == "Shift") and (block_type == "self_attention+lightweight_shiftadd"):
        unit_energy["unit_comp"] = unit_shift/(1e9) # mJ/MAC
    else:
        unit_energy["unit_comp"] = unit_mult/(1e9) # mJ/MAC

    if layer_dict["type"] == "Shift":
        num_bits = (16 + 5) // 2
        unit_energy["unit_DRAM"] = 200/(1e9) * (num_bits / 16) # mJ/16 bits
        unit_energy["unit_DRAM_GB"] = 0.0/(1e9) * (num_bits / 16) # mJ/16 bits
        unit_energy["unit_GB"] = 6.0/(1e9) * 3.7 * (num_bits / 16) # mJ/16 bits
        unit_energy["unit_NoC"] = 2.0/(1e9) * 3.7 * (num_bits / 16) # mJ/16 bits
        unit_energy["unit_NoC_psum"] = 1.0/(1e9) * 3.7 * (num_bits / 16) # mJ/16 bits
        unit_energy["unit_RF"] = 1.0/(1e9) * 3.7 * (num_bits / 16) # mJ/16 bits
        unit_energy["unit_RF_weight"] = 2.0/(1e9) * 3.7 * (num_bits / 16) # mJ/16 bits

    elif layer_dict["type"] == "Conv":
        # on chip
        unit_energy["unit_DRAM"] = 0 # mJ/16 bits
        unit_energy["unit_DRAM_GB"] = 0 # mJ/16 bits
        unit_energy["unit_GB"] = 0 # mJ/16 bits
        unit_energy["unit_NoC"] = 0 # mJ/16 bits
        unit_energy["unit_NoC_psum"] = 1.0/(1e9) * 3.7 * (num_bits / 16) # mJ/16 bits
        unit_energy["unit_RF"] = 1.0/(1e9) * 3.7 * (num_bits / 16) # mJ/16 bits
        unit_energy["unit_RF_weight"] = 2.0/(1e9) * 3.7 * (num_bits / 16) # mJ/16 bits

    else:
        unit_energy["unit_DRAM"] = 200/(1e9) * (num_bits / 16) # mJ/16 bits
        unit_energy["unit_DRAM_GB"] = 0.0/(1e9) * (num_bits / 16) # mJ/16 bits
        unit_energy["unit_GB"] = 6.0/(1e9) * 3.7 * (num_bits / 16) # mJ/16 bits
        unit_energy["unit_NoC"] = 2.0/(1e9) * 3.7 * (num_bits / 16) # mJ/16 bits
        unit_energy["unit_NoC_psum"] = 1.0/(1e9) * 3.7 * (num_bits / 16) # mJ/16 bits
        unit_energy["unit_RF"] = 1.0/(1e9) * 3.7 * (num_bits / 16) # mJ/16 bits
        unit_energy["unit_RF_weight"] = 2.0/(1e9) * 3.7 * (num_bits / 16) # mJ/16 bits

    # need to confirm
    # unit_latency = 1.0 / 1e9 # 1GHz
    if layer_dict["type"] == "Shift":
        # do not need to store back to off-chip memory (plus 16 bits benefits)
        unit_latency = 1.0 / (250e6) / 2 # 250MHz
    else:
        unit_latency = 1.0 / (250e6) # 250MHz

    print(unit_energy)

    return unit_energy, unit_latency

# gives the energy (mJ), latency (ms)
def get_OPs_HW_metric(layer_dict, v_stats=False,v_show_optimal=False,v_align=False, block_type='self_attention', num_bits=32):
    # constant defination
    if layer_dict["type"] == "FC":
        FC_flag = True
    elif layer_dict["type"] == "Shift":
        FC_flag = True
    else:
        FC_flag = False
    # unit_energy = {}
    # if (layer_dict["type"] == "AvgP") or ((layer_dict["type"] == "MaxP")):
    #     unit_energy["unit_comp"] = 0.0/(1e9) # mJ/MAC
    # else:
    #     unit_energy["unit_comp"] = 1.0/(1e9) # mJ/MAC
    # unit_energy["unit_DRAM"] = 200/(1e9) # mJ/16 bits
    # unit_energy["unit_DRAM_GB"] = 0.0/(1e9) # mJ/16 bits
    # unit_energy["unit_GB"] = 6.0/(1e9) # mJ/16 bits
    # unit_energy["unit_NoC"] = 2.0/(1e9) # mJ/16 bits
    # unit_energy["unit_NoC_psum"] = 1.0/(1e9) # mJ/16 bits
    # unit_energy["unit_RF"] = 1.0/(1e9) # mJ/16 bits
    # unit_energy["unit_RF_weight"] = 2.0/(1e9) # mJ/16 bits
    # unit_latency = 1.0/(250e6)*(1e3) # ms

    unit_energy, unit_latency = get_unit_energy(layer_dict, block_type, num_bits)

    # Add basic information to tiling_factor
    base_tiling_factor = {}
    base_tiling_factor["N"] = layer_dict["batch"]
    base_tiling_factor["H"] = layer_dict["input_H"]
    base_tiling_factor["W"] = layer_dict["input_W"]
    base_tiling_factor["C"] = layer_dict["input_C"]
    base_tiling_factor["R"] = layer_dict["kernel_size"]
    base_tiling_factor["S"] = layer_dict["kernel_size"]
    base_tiling_factor["M"] = layer_dict["output_M"]
    base_tiling_factor["E"] = layer_dict["output_E"]
    base_tiling_factor["F"] = layer_dict["output_F"]
    base_tiling_factor["stride"] = layer_dict["stride"]
    # tile N to N0 * N3
    N_tile_list = tile(base_tiling_factor["N"], 2)
    # tile M to M0 * M1 * M2 * M3
    M_tile_list = tile(base_tiling_factor["M"], 4)
    # filter out M0 > 16 options
    for tile_option in M_tile_list:
        if tile_option[0] > 16:
            M_tile_list.remove(tile_option)
    # tile C to C0 * C1 * C2 * C3
    C_tile_list = tile(base_tiling_factor["C"], 4)
    # filter out C3 != 1 options
    for tile_option in C_tile_list:
        if tile_option[3] != 1:
            C_tile_list.remove(tile_option)
    # tile E to E1 * E3
    E_tile_list = tile(base_tiling_factor["E"], 2)
    # filter out E1 == 1 options
    if not FC_flag:
        for tile_option in E_tile_list:
            if tile_option[0] == 1:
                E_tile_list.remove(tile_option)

    energy_list = []
    breakdown_list = []
    latency_list = []
    tiling_factor_list = []

    for N_tile in N_tile_list:
        for M_tile in M_tile_list:
            for C_tile in C_tile_list:
                for E_tile in E_tile_list:
                    tiling_factor = copy.deepcopy(base_tiling_factor)
                    tiling_factor["N0"] = N_tile[0]
                    tiling_factor["N3"] = N_tile[1]
                    tiling_factor["M0"] = M_tile[0]
                    tiling_factor["M1"] = M_tile[1]
                    tiling_factor["M2"] = M_tile[2]
                    tiling_factor["M3"] = M_tile[3]
                    tiling_factor["C0"] = C_tile[0]
                    tiling_factor["C1"] = C_tile[1]
                    tiling_factor["C2"] = C_tile[2]
                    tiling_factor["C3"] = C_tile[3]
                    tiling_factor["E1"] = E_tile[0]
                    tiling_factor["E3"] = E_tile[1]
                    if valid_tiling(tiling_factor, FC_flag, num_bits=num_bits):
                        energy, breakdown  =  get_energy(tiling_factor, unit_energy)
                        latency = get_latency(tiling_factor, unit_latency)

                        energy_list.append(energy)
                        breakdown_list.append(breakdown)
                        latency_list.append(latency)
                        tiling_factor_list.append(tiling_factor)

    # tiling factor search M, C, E
    max_energy = max(energy_list)
    min_energy = min(energy_list)

    max_latency = max(latency_list)
    min_latency = min(latency_list)

    min_normal_metric = (energy_list[0]-min_energy)/max_energy + (latency_list[0]-min_latency)/max_latency

    total_optimal_tiling_factor_idx = [0]
    latency_optimal_tiling_factor_idx = []
    energy_optimal_tiling_factor_idx = []

    for i in range(len(tiling_factor_list)):
        tiling_factor = tiling_factor_list[i]
        energy = energy_list[i]
        latency = latency_list[i]

        normal_metric = (energy_list[i]-min_energy)/max_energy + (latency_list[i]-min_latency)/max_latency

        # update total optimal
        if normal_metric < min_normal_metric:
            min_normal_metric = normal_metric
            total_optimal_tiling_factor_idx = [i]
        elif normal_metric == min_normal_metric:
            total_optimal_tiling_factor_idx.append(i)
        if latency == min_latency:
            latency_optimal_tiling_factor_idx.append(i)
        if energy == min_energy:
            energy_optimal_tiling_factor_idx.append(i)

    if v_stats:
        print("max_energy: {} mJ, min_energy: {} mJ".format(max_energy, min_energy))
        print("max_latency: {} ms, min_latency: {} ms".format(max_latency, min_latency))

    if len(latency_optimal_tiling_factor_idx) > 1:
        if v_show_optimal:
            print("Notice!!!!!, There are multiple latency optimal tiling factor")
    for optimal_idx in latency_optimal_tiling_factor_idx:
        if v_show_optimal:
            print("==Latency OPTIMAL==")
            print("Latency optimal tiling factor: {}, with Energy: {} mJ, Latency: {} ms".format(tiling_factor_list[optimal_idx], energy_list[optimal_idx], latency_list[optimal_idx]))
            print("==Latency OPTIMAL==")

    if len(energy_optimal_tiling_factor_idx) > 1:
        if v_show_optimal:
            print("Notice!!!!!, There are multiple energy optimal tiling factor")
    for optimal_idx in energy_optimal_tiling_factor_idx:
        if v_show_optimal:
            print("==Energy OPTIMAL==")
            print("Energy optimal tiling factor: {}, with Energy: {} mJ, Latency: {} ms".format(tiling_factor_list[optimal_idx], energy_list[optimal_idx], latency_list[optimal_idx]))
            print("==Energy OPTIMAL==")

    if len(total_optimal_tiling_factor_idx) > 1:
        if v_align:
            print("Notice!!!!!, There are multiple total optimal tiling factor")
    for optimal_idx in total_optimal_tiling_factor_idx:
        if ((optimal_idx in latency_optimal_tiling_factor_idx) and (optimal_idx in energy_optimal_tiling_factor_idx)):
            if v_align:
                print("==TOTAL OPTIMAL==")
                print("Good News, this tiling factor is latency optimal + energy optimal:")
                print("Total optimal tiling factor: {}, with Energy: {} (min: {}) mJ, Latency: {} (min: {}) ms".format(tiling_factor_list[optimal_idx], energy_list[optimal_idx], min_energy, latency_list[optimal_idx], min_latency))
                print("==TOTAL OPTIMAL==")
        else:
            if v_align:
                print("==TOTAL OPTIMAL==")
                print("Bad News, this tiling factor is NOT latency optimal + energy optimal:")
                print("Total optimal tiling factor: {}, with Energy: {} (min: {}) mJ, Latency: {} (min: {}) ms".format(tiling_factor_list[optimal_idx], energy_list[optimal_idx], min_energy, latency_list[optimal_idx], min_latency))
                print("==TOTAL OPTIMAL==")
    return energy_list[total_optimal_tiling_factor_idx[0]], latency_list[total_optimal_tiling_factor_idx[0]], breakdown_list[total_optimal_tiling_factor_idx[0]], min_energy, min_latency


def prediction(OPs_list, save_name, block_type="self_attention", num_bits=32):
    for item in OPs_list:
        idx = item["idx"]
        energy, latency, breakdown, min_energy, min_latency = get_OPs_HW_metric(OPs_list[idx],v_stats=False,v_show_optimal=False,v_align=True, block_type=block_type, num_bits=num_bits)
        OPs_list[idx]["energy"] = energy
        OPs_list[idx]["latency"]= latency
        print("============================>{}st OPs, energy: {} (min: {}) mJ, latency: {} (min: {}) ms".format(idx, energy, min_energy, latency, min_latency))
        print("                               >energy breakdown: computation: {} mJ; DRAM: {} mJ; DRAM-GB: {} mJ; GB: {} mJ; NoC: {} mJ; RF: {} mJ".format(breakdown[0], breakdown[1], breakdown[2], breakdown[3], breakdown[4], breakdown[5]))
        print("                               >energy breakdown: input: {} mJ; weight: {} mJ; output: {} mJ".format(breakdown[6], breakdown[7], breakdown[8]))
        data = [idx, breakdown[0]*1e9, energy, latency, breakdown[6], breakdown[7], breakdown[8], breakdown[0], breakdown[1], breakdown[2], breakdown[3], breakdown[4], breakdown[5] ]
        with open(save_name, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(data)

def main_encoder(config, save_name='example.csv', num_bits=32):

    row_title = ['idx', 'FLOPs', 'Energy (mJ)', 'Latency (ms)', 'input', 'weight', 'output', 'computation', 'DRAM', 'DRAM-GB', 'GB', 'NoC', 'RF']
    with open(save_name, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(row_title)

    kernel_size_list = [3, 7, 15, 31, 31, 31]

    OPs_list = []

    # encoder
    idx = 0
    for i in range(config["encoder_layers"]):
        embedding_dim = config["encoder_embedding_dim"]
        ffn_dim = config["encoder_ffn_dim"][i]
        num_head = config["encoder_num_head"][i]
        block_type = config["encoder_block_types"][i]
        qkv_dim = 512
        num_token = 30

        if block_type == "self_attention":
            OPs_list = []
            idx = 0
            ops = self_attention(embedding_dim=embedding_dim, num_head=num_head, qkv_dim=qkv_dim, idx=idx, num_token=num_token)
            idx += len(ops)
            OPs_list.extend(ops)

            ops = mlp(embedding_dim=embedding_dim, ffn_dim=ffn_dim, idx=idx, num_token=num_token)
            idx += len(ops)
            OPs_list.extend(ops)

        elif block_type == "self_attention+lightweight_conv":
            OPs_list = []
            idx = 0
            ops = self_attention(embedding_dim=embedding_dim//2, num_head=num_head, qkv_dim=qkv_dim, idx=idx, num_token=num_token)
            idx += len(ops)
            OPs_list.extend(ops)

            kernel_size = kernel_size_list[i]
            padding = (kernel_size - 1) // 2
            ops = lightweight(embedding_dim=embedding_dim//2, num_head=num_head, idx=idx, num_token=num_token, kernel_size=kernel_size, padding=padding)
            idx += len(ops)
            OPs_list.extend(ops)

            ops = mlp(embedding_dim=embedding_dim, ffn_dim=ffn_dim, idx=idx, num_token=num_token)
            idx += len(ops)
            OPs_list.extend(ops)

        elif block_type == "self_attention+lightweight_add":
            OPs_list = []
            idx = 0
            ops = self_attention(embedding_dim=embedding_dim//2, num_head=num_head, qkv_dim=qkv_dim, idx=idx, num_token=num_token)
            idx += len(ops)
            OPs_list.extend(ops)

            kernel_size = kernel_size_list[i]
            padding = (kernel_size - 1) // 2
            ops = lightweight(embedding_dim=embedding_dim//2, num_head=num_head, idx=idx, num_token=num_token, kernel_size=kernel_size, padding=padding)
            idx += len(ops)
            OPs_list.extend(ops)

            ops = mlp(embedding_dim=embedding_dim, ffn_dim=ffn_dim, idx=idx, num_token=num_token)
            idx += len(ops)
            OPs_list.extend(ops)

        elif block_type == "self_attention+lightweight_shiftadd":
            OPs_list = []
            idx = 0
            ops = self_attention(embedding_dim=embedding_dim//2, num_head=num_head, qkv_dim=qkv_dim, idx=idx, num_token=num_token)
            idx += len(ops)
            OPs_list.extend(ops)

            kernel_size = kernel_size_list[i]
            padding = (kernel_size - 1) // 2
            ops = lightweight(embedding_dim=embedding_dim//2, num_head=num_head, idx=idx, num_token=num_token, kernel_size=kernel_size, padding=padding)
            idx += len(ops)
            OPs_list.extend(ops)

            ops = mlp(embedding_dim=embedding_dim, ffn_dim=ffn_dim, idx=idx, num_token=num_token)
            idx += len(ops)
            OPs_list.extend(ops)

        else:
            print('unsupported block type: {}'.format(block_type))
            exit()

        prediction(OPs_list, save_name=save_name, block_type=block_type, num_bits=num_bits)


# def self_attention(embedding_dim, num_head, qkv_dim=512, idx=0, num_token=30):
# def mlp(embedding_dim, ffn_dim, idx=0, num_token=30):
# def lightweight(embedding_dim, num_head, idx=0, num_token=30, kernel_size=3, padding=1):
# def self_attention_ende(embedding_dim, embedding_dim_encoder, num_head_1, num_head_2, qkv_dim=512, idx=0, num_token=30, ende=-1):

def main_decoder(config, save_name='example.csv', num_bits=32):

    row_title = ['idx', 'FLOPs', 'Energy (mJ)', 'Latency (ms)', 'input', 'weight', 'output', 'computation', 'DRAM', 'DRAM-GB', 'GB', 'NoC', 'RF']
    with open(save_name, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(row_title)

    OPs_list = []
    kernel_size_list = [3, 7, 15, 31, 31, 31]

    # decoder
    idx = 0
    for i in range(config["decoder_layers"]):
        embedding_dim = config["decoder_embedding_dim"]
        embedding_dim_encoder = config["encoder_embedding_dim"]
        ffn_dim = config["decoder_ffn_dim"][i]
        num_head_1 = config["decoder_num_head_1"][i]
        num_head_2 = config["decoder_num_head_2"][i]
        block_type = config["decoder_block_types"][i]
        ende = config["arbitrary_ende"][i]
        qkv_dim = 512
        num_token = 30

        if block_type == "self_attention":
            OPs_list = []
            idx = 0
            ops = self_attention(embedding_dim=embedding_dim, num_head=num_head_1, qkv_dim=qkv_dim, idx=idx, num_token=num_token)
            idx += len(ops)
            OPs_list.extend(ops)

            ops = self_attention_ende(embedding_dim=embedding_dim, embedding_dim_encoder=embedding_dim_encoder, num_head_1=num_head_1,
                                num_head_2=num_head_2, qkv_dim=qkv_dim, idx=idx, num_token=num_token, ende=ende)
            idx += len(ops)
            OPs_list.extend(ops)

            if i == 0 or i == (config["decoder_layers"]-1):
                ops = mlp(embedding_dim=embedding_dim, ffn_dim=ffn_dim, idx=idx, num_token=num_token)
            else:
                print('use shift_mlp')
                ops = shift_mlp(embedding_dim=embedding_dim, ffn_dim=ffn_dim, idx=idx, num_token=num_token)
            idx += len(ops)
            OPs_list.extend(ops)

        elif block_type == "self_attention+lightweight_conv":
            OPs_list = []
            idx = 0
            ops = self_attention(embedding_dim=embedding_dim//2, num_head=num_head_1, qkv_dim=qkv_dim, idx=idx, num_token=num_token)
            idx += len(ops)
            OPs_list.extend(ops)

            kernel_size = kernel_size_list[i]
            padding = (kernel_size - 1) // 2
            ops = lightweight(embedding_dim=embedding_dim//2, num_head=num_head_1, idx=idx, num_token=num_token, kernel_size=kernel_size, padding=padding)
            idx += len(ops)
            OPs_list.extend(ops)

            ops = self_attention_ende(embedding_dim=embedding_dim, embedding_dim_encoder=embedding_dim_encoder, num_head_1=num_head_1,
                                num_head_2=num_head_2, qkv_dim=qkv_dim, idx=idx, num_token=num_token, ende=ende)
            idx += len(ops)
            OPs_list.extend(ops)

            if i == 0 or i == (config["decoder_layers"]-1):
                ops = mlp(embedding_dim=embedding_dim, ffn_dim=ffn_dim, idx=idx, num_token=num_token)
            else:
                print('use shift_mlp')
                ops = shift_mlp(embedding_dim=embedding_dim, ffn_dim=ffn_dim, idx=idx, num_token=num_token)
            idx += len(ops)
            OPs_list.extend(ops)

        elif block_type == "self_attention+lightweight_add":
            OPs_list = []
            idx = 0
            ops = self_attention(embedding_dim=embedding_dim//2, num_head=num_head_1, qkv_dim=qkv_dim, idx=idx, num_token=num_token)
            idx += len(ops)
            OPs_list.extend(ops)

            kernel_size = kernel_size_list[i]
            padding = (kernel_size - 1) // 2
            ops = lightweight(embedding_dim=embedding_dim//2, num_head=num_head_1, idx=idx, num_token=num_token, kernel_size=kernel_size, padding=padding)
            idx += len(ops)
            OPs_list.extend(ops)

            ops = self_attention_ende(embedding_dim=embedding_dim, embedding_dim_encoder=embedding_dim_encoder, num_head_1=num_head_1,
                                num_head_2=num_head_2, qkv_dim=qkv_dim, idx=idx, num_token=num_token, ende=ende)
            idx += len(ops)
            OPs_list.extend(ops)

            if i == 0 or i == (config["decoder_layers"]-1):
                ops = mlp(embedding_dim=embedding_dim, ffn_dim=ffn_dim, idx=idx, num_token=num_token)
            else:
                print('use shift_mlp')
                ops = shift_mlp(embedding_dim=embedding_dim, ffn_dim=ffn_dim, idx=idx, num_token=num_token)
            idx += len(ops)
            OPs_list.extend(ops)

        elif block_type == "self_attention+lightweight_shiftadd":
            OPs_list = []
            idx = 0
            ops = self_attention(embedding_dim=embedding_dim//2, num_head=num_head_1, qkv_dim=qkv_dim, idx=idx, num_token=num_token)
            idx += len(ops)
            OPs_list.extend(ops)

            kernel_size = kernel_size_list[i]
            padding = (kernel_size - 1) // 2
            ops = lightweight(embedding_dim=embedding_dim//2, num_head=num_head_1, idx=idx, num_token=num_token, kernel_size=kernel_size, padding=padding)
            idx += len(ops)
            OPs_list.extend(ops)

            ops = self_attention_ende(embedding_dim=embedding_dim, embedding_dim_encoder=embedding_dim_encoder, num_head_1=num_head_1,
                                num_head_2=num_head_2, qkv_dim=qkv_dim, idx=idx, num_token=num_token, ende=ende)
            idx += len(ops)
            OPs_list.extend(ops)

            if i == 0 or i == (config["decoder_layers"]-1):
                ops = mlp(embedding_dim=embedding_dim, ffn_dim=ffn_dim, idx=idx, num_token=num_token)
            else:
                print('use shift_mlp')
                ops = shift_mlp(embedding_dim=embedding_dim, ffn_dim=ffn_dim, idx=idx, num_token=num_token)
            idx += len(ops)
            OPs_list.extend(ops)

        else:
            print('unsupported block type: {}'.format(block_type))
            exit()

        prediction(OPs_list, save_name=save_name, block_type=block_type, num_bits=num_bits)



if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='ShiftAdd Simulator')
    parser.add_argument('--num_bits', default=32, help='bits')

    args = parser.parse_args()

    # 300ms
    # config = {
    #     # encoder
    #     "encoder_embedding_dim": 512,
    #     "encoder_ffn_dim": [3072, 4096, 4096, 4096, 4096, 4096],
    #     # "encoder_ffn_dim": [3072, 3072, 3072, 3072, 3072, 3072],
    #     "encoder_layers": 6,
    #     "encoder_num_head": [4, 4, 4, 4, 4, 16],
    #     "encoder_block_types": ["self_attention+lightweight_conv", "self_attention+lightweight_conv", "self_attention+lightweight_conv",
    #                             "self_attention+lightweight_add", "self_attention+lightweight_add", "self_attention"],
    #     # decoder
    #     "decoder_embedding_dim": 768,
    #     "decoder_ffn_dim": [4096, 3072, 4096, 4096],
    #     # "decoder_ffn_dim": [2048, 2048, 3072, 3072],
    #     "decoder_layers": 4,
    #     "decoder_num_head_1": [16, 4, 4, 4],
    #     "decoder_num_head_2": [8, 4, 4, 4],
    #     "decoder_block_types": ["self_attention", "self_attention+lightweight_shiftadd", "self_attention+lightweight_add", "self_attention+lightweight_conv"],
    #     # ende
    #     "arbitrary_ende": [2, 1, 1, 1]
    # }

    # num_bits = args.num_bits

    # if os.path.exists('300ms_encoder_32bits.csv'):
    #     os.remove('300ms_encoder_32bits.csv')

    # if os.path.exists('300ms_decoder_32bits.csv'):
    #     os.remove('300ms_decoder_32bits.csv')

    # main_encoder(config, save_name='300ms_encoder_32bits.csv')
    # main_decoder(config, save_name='300ms_decoder_32bits.csv')

    # num_bits = 8

    # if os.path.exists('300ms_encoder_8bits.csv'):
    #     os.remove('300ms_encoder_8bits.csv')

    # if os.path.exists('300ms_decoder_8bits.csv'):
    #     os.remove('300ms_decoder_8bits.csv')

    # main_encoder(config, save_name='300ms_encoder_8bits.csv', num_bits=8)
    # main_decoder(config, save_name='300ms_decoder_8bits.csv', num_bits=8)


    # 350ms
    config = {
        # encoder
        "encoder_embedding_dim": 512,
        "encoder_ffn_dim": [3072, 3072, 2048, 3072, 4096, 4096],
        # "encoder_ffn_dim": [3072, 3072, 3072, 3072, 3072, 3072],
        "encoder_layers": 6,
        "encoder_num_head": [4, 4, 4, 4, 4, 16],
        "encoder_block_types": ["self_attention+lightweight_add", "self_attention+lightweight_shiftadd",
                "self_attention+lightweight_add", "self_attention+lightweight_add", "self_attention", "self_attention"],
        # decoder
        "decoder_embedding_dim": 768,
        "decoder_ffn_dim": [4096, 4096, 4096, 4096, 4096],
        # "decoder_ffn_dim": [2048, 2048, 3072, 3072],
        "decoder_layers": 5,
        "decoder_num_head_1": [16, 4, 4, 4, 4],
        "decoder_num_head_2": [8, 4, 4, 4, 4],
        "decoder_block_types": ["self_attention+lightweight_conv", "self_attention+lightweight_shiftadd",
                "self_attention+lightweight_add", "self_attention", "self_attention+lightweight_conv"],
        # ende
        "arbitrary_ende": [2, -1, 2, 2, 2]
    }

    num_bits = args.num_bits

    if os.path.exists('350ms_encoder_32bits_shift_ffn_v1.csv'):
        os.remove('350ms_encoder_32bits_shift_ffn_v1.csv')

    if os.path.exists('350ms_decoder_32bits_shift_ffn_v1.csv'):
        os.remove('350ms_decoder_32bits_shift_ffn_v1.csv')

    main_encoder(config, save_name='350ms_encoder_32bits_shift_ffn_v1.csv')
    main_decoder(config, save_name='350ms_decoder_32bits_shift_ffn_v1.csv')


    # num_bits = 8

    # if os.path.exists('350ms_encoder_8bits.csv'):
    #     os.remove('350ms_encoder_8bits.csv')

    # if os.path.exists('350ms_decoder_8bits.csv'):
    #     os.remove('350ms_decoder_8bits.csv')

    # main_encoder(config, save_name='350ms_encoder_8bits.csv', num_bits=8)
    # main_decoder(config, save_name='350ms_decoder_8bits.csv', num_bits=8)


    # 300ms (Jul. 23)
    # config = {
    #     # encoder
    #     "encoder_embedding_dim": 512,
    #     "encoder_ffn_dim": [3072, 3072, 3072, 3072, 3072, 3072],
    #     # "encoder_ffn_dim": [3072, 3072, 3072, 3072, 3072, 3072],
    #     "encoder_layers": 6,
    #     "encoder_num_head": [4, 4, 4, 4, 4, 16],
    #     "encoder_block_types": ["self_attention+lightweight_conv", "self_attention+lightweight_conv", "self_attention+lightweight_conv",
    #                             "self_attention+lightweight_add", "self_attention+lightweight_add", "self_attention"],
    #     # decoder
    #     "decoder_embedding_dim": 768,
    #     "decoder_ffn_dim": [2048, 2048, 3072, 3072],
    #     # "decoder_ffn_dim": [2048, 2048, 3072, 3072],
    #     "decoder_layers": 4,
    #     "decoder_num_head_1": [16, 4, 4, 4],
    #     "decoder_num_head_2": [8, 4, 4, 4],
    #     "decoder_block_types": ["self_attention", "self_attention+lightweight_shiftadd", "self_attention+lightweight_add", "self_attention+lightweight_conv"],
    #     # ende
    #     "arbitrary_ende": [2, 1, 2, 1]
    # }

    # num_bits = args.num_bits

    # if os.path.exists('300ms_Jul_23_encoder_32bits.csv'):
    #     os.remove('300ms_Jul_23_encoder_32bits.csv')

    # if os.path.exists('300ms_Jul_23_decoder_32bits.csv'):
    #     os.remove('300ms_Jul_23_decoder_32bits.csv')

    # main_encoder(config, save_name='300ms_Jul_23_encoder_32bits.csv')
    # main_decoder(config, save_name='300ms_Jul_23_decoder_32bits.csv')


    # 200ms (update Aug. 4)
    # config = {
    #     # encoder
    #     "encoder_embedding_dim": 512,
    #     "encoder_ffn_dim": [2048, 4096, 2048, 1024, 1024, 3072],
    #     "encoder_layers": 6,
    #     "encoder_num_head": [8, 4, 4, 4, 4, 8],
    #     "encoder_block_types": ["self_attention", "self_attention+lightweight_shiftadd",
    #         "self_attention+lightweight_conv", "self_attention+lightweight_conv",
    #         "self_attention+lightweight_conv", "self_attention"],
    #     # decoder
    #     "decoder_embedding_dim": 512,
    #     "decoder_ffn_dim": [4096, 4096, 4096],
    #     # "decoder_ffn_dim": [2048, 2048, 3072, 3072],
    #     "decoder_layers": 3,
    #     "decoder_num_head_1": [4, 8, 16],
    #     "decoder_num_head_2": [4, 4, 4],
    #     "decoder_block_types": ["self_attention+lightweight_conv", "self_attention",
    #         "self_attention"],
    #     # ende
    #     "arbitrary_ende": [2, 2, 2]
    # }

    # num_bits = args.num_bits

    # if os.path.exists('200ms_Aug_4_encoder_32bits.csv'):
    #     os.remove('200ms_Aug_4_encoder_32bits.csv')

    # if os.path.exists('200ms_Aug_4_decoder_32bits.csv'):
    #     os.remove('200ms_Aug_4_decoder_32bits.csv')

    # main_encoder(config, save_name='200ms_Aug_4_encoder_32bits.csv')
    # main_decoder(config, save_name='200ms_Aug_4_decoder_32bits.csv')


    # num_bits = 8

    # if os.path.exists('200ms_Aug_4_encoder_8bits.csv'):
    #     os.remove('200ms_Aug_4_encoder_8bits.csv')

    # if os.path.exists('200ms_Aug_4_decoder_8bits.csv'):
    #     os.remove('200ms_Aug_4_decoder_8bits.csv')

    # main_encoder(config, save_name='200ms_Aug_4_encoder_8bits.csv', num_bits=8)
    # main_decoder(config, save_name='200ms_Aug_4_decoder_8bits.csv', num_bits=8)


    # 250ms (update Aug. 4)
    # config = {
    #     # encoder
    #     "encoder_embedding_dim": 512,
    #     "encoder_ffn_dim": [2048, 2048, 4096, 4096, 1024, 4096],
    #     "encoder_layers": 6,
    #     "encoder_num_head": [8, 4, 16, 16, 8, 4],
    #     "encoder_block_types": ["self_attention+lightweight_conv", "self_attention+lightweight_conv",
    #         "self_attention+lightweight_add", "self_attention+lightweight_add", "self_attention", "self_attention"],
    #     # decoder
    #     "decoder_embedding_dim": 512,
    #     "decoder_ffn_dim": [4096, 3072, 3072],
    #     # "decoder_ffn_dim": [2048, 2048, 3072, 3072],
    #     "decoder_layers": 3,
    #     "decoder_num_head_1": [4, 16, 8],
    #     "decoder_num_head_2": [4, 16, 4],
    #     "decoder_block_types": ["self_attention+lightweight_conv", "self_attention",
    #         "self_attention"],
    #     # ende
    #     "arbitrary_ende": [2, 2, 2]
    # }

    # num_bits = args.num_bits

    # if os.path.exists('250ms_Aug_4_encoder_32bits.csv'):
    #     os.remove('250ms_Aug_4_encoder_32bits.csv')

    # if os.path.exists('250ms_Aug_4_decoder_32bits.csv'):
    #     os.remove('250ms_Aug_4_decoder_32bits.csv')

    # main_encoder(config, save_name='250ms_Aug_4_encoder_32bits.csv')
    # main_decoder(config, save_name='250ms_Aug_4_decoder_32bits.csv')


    # num_bits = 8

    # if os.path.exists('250ms_Aug_4_encoder_8bits.csv'):
    #     os.remove('250ms_Aug_4_encoder_8bits.csv')

    # if os.path.exists('250ms_Aug_4_decoder_8bits.csv'):
    #     os.remove('250ms_Aug_4_decoder_8bits.csv')

    # main_encoder(config, save_name='250ms_Aug_4_encoder_8bits.csv', num_bits=8)
    # main_decoder(config, save_name='250ms_Aug_4_decoder_8bits.csv', num_bits=8)

    # 300ms (update Aug. 4)
    # config = {
    #     # encoder
    #     "encoder_embedding_dim": 512,
    #     "encoder_ffn_dim": [2048, 1024, 1024, 1024, 2048, 4096],
    #     "encoder_layers": 6,
    #     "encoder_num_head": [4, 8, 8, 4, 4, 8],
    #     "encoder_block_types": ["self_attention+lightweight_conv", "self_attention+lightweight_conv",
    #         "self_attention", "self_attention+lightweight_add", "self_attention", "self_attention"],
    #     # decoder
    #     "decoder_embedding_dim": 512,
    #     "decoder_ffn_dim": [4096, 3072, 3072],
    #     # "decoder_ffn_dim": [2048, 2048, 3072, 3072],
    #     "decoder_layers": 3,
    #     "decoder_num_head_1": [4, 16, 8],
    #     "decoder_num_head_2": [4, 16, 16],
    #     "decoder_block_types": ["self_attention+lightweight_conv", "self_attention",
    #         "self_attention"],
    #     # ende
    #     "arbitrary_ende": [2, 2, -1]
    # }

    # num_bits = args.num_bits

    # if os.path.exists('300ms_Aug_4_encoder_32bits.csv'):
    #     os.remove('300ms_Aug_4_encoder_32bits.csv')

    # if os.path.exists('300ms_Aug_4_decoder_32bits.csv'):
    #     os.remove('300ms_Aug_4_decoder_32bits.csv')

    # main_encoder(config, save_name='300ms_Aug_4_encoder_32bits.csv')
    # main_decoder(config, save_name='300ms_Aug_4_decoder_32bits.csv')


    # num_bits = 8

    # if os.path.exists('300ms_Aug_4_encoder_8bits.csv'):
    #     os.remove('300ms_Aug_4_encoder_8bits.csv')

    # if os.path.exists('300ms_Aug_4_decoder_8bits.csv'):
    #     os.remove('300ms_Aug_4_decoder_8bits.csv')

    # main_encoder(config, save_name='300ms_Aug_4_encoder_8bits.csv', num_bits=8)
    # main_decoder(config, save_name='300ms_Aug_4_decoder_8bits.csv', num_bits=8)