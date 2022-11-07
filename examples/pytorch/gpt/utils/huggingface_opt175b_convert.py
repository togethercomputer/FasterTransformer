# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Convert huggingface Meta OPT model. Use https://huggingface.co/facebook/opt-125m as demo.
'''

import argparse
import configparser
import multiprocessing
import numpy as np
from pathlib import Path
import torch

import os
import sys
from datetime import datetime
from transformers import OPTForCausalLM, AutoModelForCausalLM # transformers-4.20.0.dev0
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../../..")
sys.path.append(dir_path)


def get_weight_data_type(data_type):
    if data_type == "fp16":
        return np.float16
    else:
        assert False, f"Invalid weight data type {data_type}"


def fuse_qkv_weight(q, k, v):
    qkv = torch.cat([q, k, v], dim=-1)
    return qkv


def split_and_convert_layer(layer_index, saved_dir, partition_num, model_dict):
    print(f"<split_and_convert_layer>: handle layer: {layer_index}")
    save_path_prefix = saved_dir + "/model.layers." + str(layer_index) + "."

    # fuse qkv model and bias
    q_weight = model_dict[f'self_attn.q_proj.weight']
    k_weight = model_dict[f'self_attn.k_proj.weight']
    v_weight = model_dict[f'self_attn.v_proj.weight']
    q_bias = model_dict[f'self_attn.q_proj.bias']
    k_bias = model_dict[f'self_attn.k_proj.bias']
    v_bias = model_dict[f'self_attn.v_proj.bias']
    qkv_weight = fuse_qkv_weight(q_weight, k_weight, v_weight)
    qkv_bias = fuse_qkv_weight(q_bias, k_bias, v_bias)

    # the parameter that does not need to be partitioned:
    model_dict['self_attn_layer_norm.bias'].detach().cpu().numpy().astype(torch.float16).tofile(
        save_path_prefix+'input_layernorm.bias')
    model_dict['self_attn_layer_norm.weight'].detach().cpu().numpy().astype(torch.float16).tofile(
        save_path_prefix + 'input_layernorm.weight')
    model_dict['final_layer_norm.bias'].detach().cpu().numpy().astype(torch.float16).tofile(
        save_path_prefix + 'post_attention_layernorm.bias')
    model_dict['final_layer_norm.weight'].detach().cpu().numpy().astype(torch.float16).tofile(
        save_path_prefix + 'post_attention_layernorm.weight')
    qkv_bias.detach().cpu().numpy().astype(torch.float16).tofile(
        save_path_prefix + 'attention.query_key_value.bias')
    model_dict['self_attn.out_proj.bias'].detach().cpu().numpy().astype(torch.float16).tofile(
        save_path_prefix + 'attention.dense.bias')
    model_dict['fc1.bias'].detach().cpu().numpy().astype(torch.float16).tofile(
        save_path_prefix + 'mlp.dense_h_to_4h.bias')
    model_dict['fc2.bias'].detach().cpu().numpy().astype(torch.float16).tofile(
        save_path_prefix + 'mlp.dense_4h_to_h.bias')

    # the parameter that need to be partitioned:
    split_qkv_weights = np.split(qkv_weight.detach().cpu().numpy().astype(torch.float16), partition_num)
    for i in range(partition_num):
        split_qkv_weights[i].tofile(save_path_prefix+'attention.query_key_value.weight' + f".{i}.bin")
    split_out_weights = np.split(model_dict['self_attn.out_proj.weight'].detach().cpu().numpy().astype(torch.float16),
                                 partition_num)
    for i in range(partition_num):
        split_out_weights[i].tofile(save_path_prefix+'attention.dense.weight' + f".{i}.bin")
    split_fc1_weights = np.split(model_dict['fc1.weight'].detach().cpu().numpy().astype(torch.float16),
                                 partition_num)
    for i in range(partition_num):
        split_fc1_weights[i].tofile(save_path_prefix+'mlp.dense_h_to_4h.weight' + f".{i}.bin")
    split_fc2_weights = np.split(model_dict['fc2.weight'].detach().cpu().numpy().astype(torch.float16),
                                 partition_num)
    for i in range(partition_num):
        split_fc2_weights[i].tofile(save_path_prefix + 'mlp.dense_4h_to_h.weight' + f".{i}.bin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--saved_dir', '-o', type=str, help='file name of output file', required=True)
    parser.add_argument('--in_file', '-i', type=str, help='file name of input checkpoint file', required=True)
    parser.add_argument('--layer_index', '-l_i', type=int, help='file name of output file', default=1)
    parser.add_argument('--partition_num', '-t_g', type=int, help='How many gpus for inference', default=8)
    args = parser.parse_args()
    print("\n=============== Argument ===============")
    for key in vars(args):
        print(f"{key}: {vars(args)[key]}")
    print("========================================")

    start_time = datetime.now()
    model_dict = torch.load(args.in_file)
    split_and_convert_layer(args.layer_index, args.saved_dir, args.partition_num, model_dict)
    stop_time = datetime.now()
    run_time = (stop_time - start_time)
    print(f"[INFO] Spend {run_time} (h:m:s) to convert the model")
