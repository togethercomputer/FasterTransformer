# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

from __future__ import print_function

from torch.nn.utils.rnn import pad_sequence
import random
import os
import sys
import argparse
import timeit
import torch
import gc
import torch.distributed as dist
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../..")
from examples.pytorch.gpt.utils.parallel_gpt import ParallelGPT
import examples.pytorch.gpt.utils.gpt_token_encoder as encoder
from transformers import AutoTokenizer, AutoConfig


def profiling_torch_tensor_memory():
    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if obj.is_cuda:
                    # print(type(obj), obj.size(), obj.element_size())
                    total_size += obj.nelement() * obj.element_size()
        except: 
            pass
    print(f"<profiling_torch_tensor_memory>: total HBM of torch tensors: {total_size/1073741824} GB.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_model_name', type=str, default="facebook/opt-66b",
                        help='hf model name to load HF config file.')
    parser.add_argument('--input_len', type=int, default=128,
                        help='input sequence length to generate.')
    parser.add_argument('--output_len', type=int, default=32,
                        help='output sequence length to generate.')
    parser.add_argument('--beam_width', type=int, default=1,
                        help='beam width for beam search. Using sampling when beam width is 1.')
    parser.add_argument('--top_k', type=int, default=50,
                        help='top k candidate num')
    parser.add_argument('--top_p', type=float, default=0.,
                        help='top p probability threshold')
    parser.add_argument('--temperature', type=float, default=0.1,
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
    parser.add_argument('--lib_path', type=str, default='./lib/libth_parallel_gpt.so',
                        help='path to the pyt_fastertransformer dynamic lib file.')
    parser.add_argument('--vocab_file', type=str, default="../models/gpt2-vocab.json",
                        help='vocabulary file.')
    parser.add_argument('--max_batch_size', type=int, default=1,
                        help='max batch size.')
    parser.add_argument('--repetition_penalty', type=float, default=1.,
                        help='repetition penalty')
    parser.add_argument('--max_seq_len', type=int, default=2048,
                        help='max sequence length for position embedding table.')
    parser.add_argument('--data_type', type=str, choices=['fp32', 'fp16', 'bf16'], default='fp16',
                        help='Data type of FT Inference computation')
    parser.add_argument('--time', action='store_true',
                        help='whether or not to measure time elapsed.')
    parser.add_argument('--sample_input_file', type=str, default=None,
                        help='path to sample input file. If not set, it runs with no context inputs.')
    parser.add_argument('--sample_output_file', type=str, default=None,
                        help='path to sample output file.')
    parser.add_argument('--enable_random_seed', action='store_true',
                        help='is use the random seed for sentences in a batch.')
    parser.add_argument('--int8_mode', type=int, default=0,
                        help='int8 mode.')
    parser.add_argument('--weights_data_type', type=str, default="fp32", choices=["fp32", "fp16"],
                        help='Data type of FT checkpoint weights')
    parser.add_argument('--return_cum_log_probs', type=int, default=0, choices=[0, 1, 2],
                        help='Whether to compute the cumulative log probsbility of sentences.'
                             ' 0: do not return the cumulative log probs '
                             ' 1: return the cumulative log probs of generated sequences'
                             ' 2: return the cumulative log probs of sequences')
    parser.add_argument('--shared_contexts_ratio', type=float, default=1.0,
                        help='Triggers the shared context optimization when'
                             'compact_size <= shared_contexts_ratio * batch_size'
                             'A value of 0.0 deactivate the optimization')

    args = parser.parse_args()

    if args.hf_model_name == 'facebook/opt-175b':
        hf_config = vars(AutoConfig.from_pretrained('facebook/opt-66b'))
        head_num = 96
        layer_num = 96
        max_seq_len = 2050
        size_per_head = 128
        tokenizer = AutoTokenizer.from_pretrained('facebook/opt-66b')
    else:
        hf_config = vars(AutoConfig.from_pretrained(args.hf_model_name))
        head_num = hf_config['num_attention_heads']
        layer_num = hf_config['num_hidden_layers']
        max_seq_len = hf_config['max_position_embeddings']
        size_per_head = hf_config['hidden_size'] // head_num
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model_name)
    
    start_id = hf_config['bos_token_id']
    end_id = hf_config['eos_token_id']  
    vocab_size = 50272
    layernorm_eps = 1e-5
    layernorm_type = 'pre_layernorm' if hf_config['do_layer_norm_before'] else 'post_layernorm'
    activation_type = 'Relu' if hf_config['activation_function'] == 'relu' else 'Gelu'
    has_post_decoder_layernorm = layernorm_type == 'pre_layernorm'
    lib_path = '/workspace/Port_FasterTransformer/build/lib/libth_parallel_gpt.so'
    
    tokenizer.pad_token = tokenizer.eos_token

    output_len = args.output_len
    beam_width = args.beam_width
    top_k = args.top_k
    top_p = args.top_p
    temperature = args.temperature
    len_penalty = args.len_penalty
    beam_search_diversity_rate = args.beam_search_diversity_rate
    tensor_para_size = args.tensor_para_size
    pipeline_para_size = args.pipeline_para_size
    max_batch_size = args.max_batch_size
    repetition_penalty = args.repetition_penalty
    weights_data_type = args.weights_data_type
    return_cum_log_probs = args.return_cum_log_probs
    return_output_length = return_cum_log_probs > 0
    shared_contexts_ratio = args.shared_contexts_ratio

    print("\n=============== Arguments ===============")
    for arg in vars(args):
        print("{}: {}".format(arg, getattr(args, arg)))
    print("=========================================\n")

    enc = tokenizer
    torch.manual_seed(0)

    
    with torch.no_grad():
        # Prepare model.
        gpt = ParallelGPT(head_num, size_per_head, vocab_size, start_id, end_id,
                          layer_num, max_seq_len, tensor_para_size, pipeline_para_size,
                          lib_path=lib_path, int8_mode=args.int8_mode, weights_data_type=weights_data_type,
                          layernorm_type = layernorm_type, has_post_decoder_layernorm = has_post_decoder_layernorm,  
                          activation_type = activation_type, layernorm_eps=layernorm_eps,
                          shared_contexts_ratio=shared_contexts_ratio)
        
        if dist.get_rank() == 0:
            print("=========Profiling before opt.load=========")
            profiling_torch_tensor_memory()
        
        torch.cuda.empty_cache()
        
        # if not gpt.load(ckpt_path=args.ckpt_path):
        if not gpt.load_w_type(ckpt_path=args.ckpt_path, infer_data_type='fp16'):
            print("[WARNING] Checkpoint file not found. Model loading is skipped.")
        
        if dist.get_rank() == 0:
            print("=========Profiling after opt.load=========")
            profiling_torch_tensor_memory()
        
        # if args.data_type == 'fp16':
        #    gpt.half()
        # elif args.data_type == 'bf16':
        #    gpt.bfloat16()

    with torch.no_grad():
        # Generate tokens.
        # Inputs
        for batch_size in [1, 16]:
            contexts = []
            if args.sample_input_file:  # conditional case
                with open(args.sample_input_file, "r") as f:
                    context = f.read().splitlines()
                    assert len(context) == 1 
                contexts = [ context[0] for _ in range(batch_size)]
                start_ids = [torch.IntTensor(enc.encode(c)) for c in contexts]
            
            start_lengths = [len(ids) for ids in start_ids]
            start_ids = pad_sequence(start_ids, batch_first=True, padding_value=end_id)
            start_lengths = torch.IntTensor(start_lengths)
            if args.enable_random_seed == True:
                random_seed_tensor = torch.randint(0, 10000, size=[batch_size], dtype=torch.int64)
            else:
                random_seed_tensor = torch.zeros([batch_size], dtype=torch.int64)
            
            for output_len in [32, 64, 128]:
                print("========================= opt-input-data: =======================")
                # print(f"start_ids: {start_ids.shape}")
                # print(f"start_lengths: {start_lengths.shape}")
                # print(f"output_len: {output_len}")
                
                tokens_batch = gpt(start_ids,
                                start_lengths,
                                output_len,
                                beam_width,
                                top_k * torch.ones(size=[max_batch_size], dtype=torch.int32),
                                top_p * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                beam_search_diversity_rate * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                temperature * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                len_penalty * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                repetition_penalty * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                random_seed_tensor,
                                return_output_length,
                                return_cum_log_probs)
                # only a thread (rank 0) gets the output, while the others are supposed to return None.
                if tokens_batch is not None:
                    if return_cum_log_probs > 0:
                        tokens_batch, _, cum_log_probs = tokens_batch
                        print('[INFO] Log probs of sentences:', cum_log_probs)
                    outputs = []
                    tokens_batch = tokens_batch.cpu().numpy()
                    for i, (context, tokens) in enumerate(zip(contexts, tokens_batch)):
                        for beam_id in range(beam_width):
                            token = tokens[beam_id][start_lengths[i]:]  # exclude context input from the output
                            output = enc.decode(token)
                            outputs.append(output)
                            if i == 0 and dist.get_rank() == 0:
                                print(f"[INFO] batch {i}, beam {beam_id}: \n[Context]\n{context}\n\n[Output]\n{output}\n")

                    if args.sample_output_file:
                        with open(args.sample_output_file, "w+") as f:
                            outputs = [o.replace("\n", "\\n") for o in outputs]
                            f.writelines("\n".join(outputs))

                # Measure inference time.
                if args.time:
                    iterations = 5
                    for i in range(iterations):
                        tokens_batch = gpt(start_ids,
                                        start_lengths,
                                        output_len,
                                        beam_width,
                                        top_k * torch.ones(size=[max_batch_size], dtype=torch.int32),
                                        top_p * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                        beam_search_diversity_rate * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                        temperature * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                        len_penalty * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                        repetition_penalty * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                        random_seed_tensor,
                                        return_output_length,
                                        return_cum_log_probs)

                    time = timeit.default_timer()
                    for i in range(iterations):
                        tokens_batch = gpt(start_ids,
                                        start_lengths,
                                        output_len,
                                        beam_width,
                                        top_k * torch.ones(size=[max_batch_size], dtype=torch.int32),
                                        top_p * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                        beam_search_diversity_rate * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                        temperature * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                        len_penalty * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                        repetition_penalty * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                        random_seed_tensor,
                                        return_output_length,
                                        return_cum_log_probs)
                    time_elapsed = timeit.default_timer() - time
                    
                    if dist.get_rank() == 0:
                        print(f"start_ids: {start_ids.shape}")
                        print(f"start_lengths: {start_lengths.shape}")
                        print(f"output_len: {output_len}")
                        print("[INFO] OPT time costs: {:.2f} ms".format(time_elapsed * 1000 / iterations))


if __name__ == '__main__':
    main()
