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


def convert_lm_head(saved_dir, model_dict):
    save_path_prefix = saved_dir + "/model."
    model_dict['final_layer_norm.bias'].detach().cpu().numpy().astype(np.float16).tofile(
        save_path_prefix+'final_layernorm.bias.bin')
    model_dict['final_layer_norm.weight'].detach().cpu().numpy().astype(np.float16).tofile(
        save_path_prefix+'final_layernorm.weight.bin')
    model_dict['lm_head.weight'].detach().cpu().numpy().astype(np.float16).tofile(
        save_path_prefix+'lm_head.weight.bin')
    
def convert_embs(saved_dir, model_dict):
    save_path_prefix = saved_dir + "/model."
    model_dict['embed_tokens.weight'].detach().cpu().numpy().astype(np.float16).tofile(
        save_path_prefix+'wte.bin')
    model_dict['embed_positions.weight'].detach().cpu().numpy().astype(np.float16).tofile(
        save_path_prefix+'wpe.bin')

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
    model_dict['self_attn_layer_norm.bias'].detach().cpu().numpy().astype(np.float16).tofile(
        save_path_prefix+'input_layernorm.bias.bin')
    model_dict['self_attn_layer_norm.weight'].detach().cpu().numpy().astype(np.float16).tofile(
        save_path_prefix + 'input_layernorm.weight.bin')
    model_dict['final_layer_norm.bias'].detach().cpu().numpy().astype(np.float16).tofile(
        save_path_prefix + 'post_attention_layernorm.bias.bin')
    model_dict['final_layer_norm.weight'].detach().cpu().numpy().astype(np.float16).tofile(
        save_path_prefix + 'post_attention_layernorm.weight.bin')
    model_dict['self_attn.out_proj.bias'].detach().cpu().numpy().astype(np.float16).tofile(
        save_path_prefix + 'attention.dense.bias.bin')
    model_dict['fc2.bias'].detach().cpu().numpy().astype(np.float16).tofile(
        save_path_prefix + 'mlp.dense_4h_to_h.bias.bin')

    # the parameter that need to be partitioned:
    split_qkv_weights = np.split(qkv_weight.detach().cpu().numpy().astype(np.float16), partition_num, axis=-1)
    for i in range(partition_num):
        split_qkv_weights[i].tofile(save_path_prefix+'attention.query_key_value.weight' + f".{i}.bin")
    split_qkv_bias = np.split(qkv_bias.detach().cpu().numpy().astype(np.float16), partition_num, axis=-1)
    for i in range(partition_num):
        split_qkv_bias[i].tofile(save_path_prefix+'attention.query_key_value.bias' + f".{i}.bin")
        
    split_out_weights = np.split(model_dict['self_attn.out_proj.weight'].detach().cpu().numpy().astype(np.float16), 
                                 partition_num, axis=0)
    for i in range(partition_num):
        split_out_weights[i].tofile(save_path_prefix+'attention.dense.weight' + f".{i}.bin")
        
    split_fc1_weights = np.split(model_dict['fc1.weight'].detach().cpu().numpy().astype(np.float16), partition_num, axis=-1)
    for i in range(partition_num):
        split_fc1_weights[i].tofile(save_path_prefix+'mlp.dense_h_to_4h.weight' + f".{i}.bin")
    split_fc1_bias = np.split(model_dict['fc1.bias'].detach().cpu().numpy().astype(np.float16), partition_num, axis=-1)
    for i in range(partition_num):
        split_fc1_bias[i].tofile(save_path_prefix+'mlp.dense_h_to_4h.bias' + f".{i}.bin")
    
    split_fc2_weights = np.split(model_dict['fc2.weight'].detach().cpu().numpy().astype(np.float16),
                                 partition_num, axis=0)
    for i in range(partition_num):
        split_fc2_weights[i].tofile(save_path_prefix + 'mlp.dense_4h_to_h.weight' + f".{i}.bin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--saved_dir', '-o', type=str, help='file name of output file', 
                        default="/workspace/Port_FasterTransformer/build/model/opt-175b-tp8/8-gpu")
    parser.add_argument('--in_dir', '-i', type=str, help='file name of input checkpoint file', 
                        default="/workspace/Port_FasterTransformer/build/model/opt-175b-new/")
    parser.add_argument('--layer_index', '-l_i', type=int, help='file name of output file', default=0)
    parser.add_argument('--partition_num', '-t_g', type=int, help='How many gpus for inference', default=8)
    args = parser.parse_args()
    print("\n=============== Argument ===============")
    for key in vars(args):
        print(f"{key}: {vars(args)[key]}")
    print("========================================")


    start_time = datetime.now()
    
    
    print("---------------- lm_head modules ------------------")
    lm_dict = torch.load(args.in_dir+'pytorch_lm_head.pt')
    for layer_name in lm_dict:
        print(f"{layer_name}: {lm_dict[layer_name].shape}")
    convert_lm_head(args.saved_dir, lm_dict)
    
    print("---------------- embs modules ------------------")
    embs_dict = torch.load(args.in_dir+'pytorch_embs.pt')
    for layer_name in embs_dict:
        print(f"{layer_name}: {embs_dict[layer_name].shape}")
    convert_embs(args.saved_dir, embs_dict)

    for i in range(96):
        print(f"---------------- layer modules <{i}> ------------------")
        layer_dict = torch.load(args.in_dir+'pytorch_'+ str(i) + '.pt')
        for layer_name in layer_dict:
            print(f"{layer_name}: {layer_dict[layer_name].shape}")
        split_and_convert_layer(i, args.saved_dir, args.partition_num, layer_dict)
    
    stop_time = datetime.now()
    run_time = (stop_time - start_time)
    print(f"[INFO] Spend {run_time} (h:m:s) to convert the model")
