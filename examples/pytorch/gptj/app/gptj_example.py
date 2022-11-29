from __future__ import print_function

from torch.nn.utils.rnn import pad_sequence
import random
import os
import sys
import argparse
import timeit
import torch
import numpy as np
import time
dir_path = os.path.dirname(os.path.realpath(__file__))
from utils.gptj import GPTJ
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_model_name', type=str, default='EleutherAI/gpt-j-6B',
                        help='hugging face model name (used to load config).')
    parser.add_argument('--output_len', type=int, default=32,
                        help='output sequence length to generate.')
    parser.add_argument('--beam_width', type=int, default=1,
                        help='beam width for beam search. Using sampling when beam width is 1.')
    parser.add_argument('--top_k', type=int, default=1,
                        help='top k candidate num')
    parser.add_argument('--top_p', type=float, default=0.,
                        help='top p probability threshold')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='temperature')
    parser.add_argument('--len_penalty', type=float, default=0.,
                        help='len_penalty')
    parser.add_argument('--beam_search_diversity_rate', type=float, default=0.,
                        help='beam_search_diversity_rate')
    parser.add_argument('--tensor_para_size', type=int, default=1,
                        help='tensor parallel size')
    parser.add_argument('--pipeline_para_size', type=int, default=1,
                        help='pipeline parallel size')
    parser.add_argument('--ckpt_path', type=str, default='../models/megatron-models/c-model/345m/1-gpu',
                        help='path to the checkpoint file.')
    parser.add_argument('--lib_path', type=str, default='./lib/libth_gpt.so',
                        help='path to the pyt_fastertransformer dynamic lib file.')
    parser.add_argument('--max_batch_size', type=int, default=8,
                        help='max batch size.')
    parser.add_argument('--repetition_penalty', type=float, default=1.,
                        help='repetition penalty')
    parser.add_argument('--infer_data_type', type=str, choices=['fp32', 'fp16', 'bf16'], default='fp16',
                        help='Data type for inference computation!')
    parser.add_argument('--sample_input_file', type=str, default=None,
                        help='path to sample input file. If not set, it runs with no context inputs.')
    parser.add_argument('--sample_output_file', type=str, default=None,
                        help='path to sample output file.')
    parser.add_argument('--enable_random_seed', action='store_true',
                        help='is enable the random seed.')
    parser.add_argument('--weights_data_type', type=str, default="fp32", choices=["fp32", "fp16"],
                        help='Data type of FT checkpoint weights. Note this is not the dtype for inference computation!')
    args = parser.parse_args()
    
    print("\n=============== Arguments ===============")
    for arg in vars(args):
        print("{}: {}".format(arg, getattr(args, arg)))
    print("=========================================\n")
    
    hf_config = vars(AutoConfig.from_pretrained(args.hf_model_name))
    head_num = hf_config['n_head']   
    size_per_head = hf_config['n_embd'] // head_num
    layer_num = hf_config['n_layer']
    vocab_size = hf_config['vocab_size']
    rotary_embedding_dim = hf_config['rotary_dim']
    start_id = hf_config['bos_token_id']
    end_id = hf_config['eos_token_id']    
    max_seq_len = hf_config['n_positions']
    
    tensor_para_size = args.tensor_para_size
    pipeline_para_size = args.pipeline_para_size
    max_batch_size = args.max_batch_size

    output_len = args.output_len
    beam_width = args.beam_width
    top_k = args.top_k
    top_p = args.top_p
    temperature = args.temperature
    len_penalty = args.len_penalty
    beam_search_diversity_rate = args.beam_search_diversity_rate
    repetition_penalty = args.repetition_penalty
    
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Inputs
    contexts = []
  
    with open(args.sample_input_file, "r") as f:
        contexts = f.read().splitlines()
        batch_size = min(len(contexts), max_batch_size)
    contexts = contexts[:batch_size]    
    start_ids = [torch.IntTensor(tokenizer.encode(c)) for c in contexts]
    start_lengths = [len(ids) for ids in start_ids]
            
    start_ids = pad_sequence(start_ids, batch_first=True, padding_value=end_id)
    start_lengths = torch.IntTensor(start_lengths)
    print(f"start_ids: shape ({start_ids.shape}) ids: {start_ids}")
    print("[INFO] batch size: {}".format(batch_size))

    if args.enable_random_seed == True:
        random_seed_tensor = torch.randint(0, 10000, size=[batch_size], dtype=torch.int64)
    else:
        random_seed_tensor = torch.zeros(size=[batch_size], dtype=torch.int64)

    # Prepare model.
    print("<gptj_example:main> started to declare model")
    gptj = GPTJ(head_num, size_per_head, layer_num, vocab_size, rotary_embedding_dim, 
                start_id, end_id, max_seq_len, tensor_para_size, pipeline_para_size,
                lib_path=args.lib_path, weights_data_type=args.weights_data_type)
    if not gptj.load(ckpt_path=args.ckpt_path, infer_data_type=args.infer_data_type):
        print("[WARNING] Checkpoint file not found. Model loading is skipped.")
        
    time.sleep(20)
    
    for i in range(10):
        with torch.no_grad():
            # Generate tokens.
            tokens_batch = gptj(start_ids,
                                start_lengths,
                                output_len,
                                beam_width,
                                top_k * torch.ones(size=[batch_size], dtype=torch.int32),
                                top_p * torch.ones(size=[batch_size], dtype=torch.float32),
                                beam_search_diversity_rate * torch.ones(size=[batch_size], dtype=torch.float32),
                                temperature * torch.ones(size=[batch_size], dtype=torch.float32),
                                len_penalty * torch.ones(size=[batch_size], dtype=torch.float32),
                                repetition_penalty * torch.ones(size=[batch_size], dtype=torch.float32),
                                random_seed_tensor)
            # only a thread (rank 0) gets the output, while the others are supposed to return None.
            if tokens_batch is not None:
                outputs = []
                tokens_batch = tokens_batch.cpu().numpy()
                for i, (context, tokens) in enumerate(zip(contexts, tokens_batch)):
                    for beam_id in range(beam_width):
                        token = tokens[beam_id][start_lengths[i]:]  # exclude context input from the output
                        output = tokenizer.decode(token)
                        outputs.append(output)
                        print(f"[INFO] batch {i}, beam {beam_id}: \n[Context]\n{context}\n\n[Output]\n{output}\n")

                if args.sample_output_file:
                    with open(args.sample_output_file, "w+") as f:
                        outputs = [o.replace("\n", "\\n") for o in outputs]
                        f.writelines("\n".join(outputs))
        

if __name__ == '__main__':
    main()
