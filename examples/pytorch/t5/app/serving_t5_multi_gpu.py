import os
from typing import Dict
import argparse
import timeit
import datetime
import logging
import configparser
import math
from utils.ft_decoding import FTT5DecodingWeight, FTT5Decoding, FTT5
from utils.ft_encoder import FTT5EncoderWeight, FTT5Encoder
from together_worker.fast_inference import FastInferenceInterface
from together_web3.computer import RequestTypeLanguageModelInference
from together_web3.together import TogetherWeb3, TogetherClientOptions
import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from utils.para_utils import *
from transformers import AutoTokenizer, AutoConfig, T5Config
import numpy as np



class FastT5Inference(FastInferenceInterface):
    def __init__(self, model_name: str, args=None) -> None:    
        try:
            if not dist.is_initialized():
                dist.init_process_group(backend='mpi')
        except:
            print("[INFO] WARNING: Have initialized the process group")
        
        args['worker_name'] = 'worker'+str(dist.get_rank())
        args['workers'] = dist.get_world_size()
        args['rank'] = dist.get_rank()
        
        super().__init__(model_name, args if args is not None else {})
        print("\n=============== Arguments ===============")
        print(args.keys())
        print(args)
        #for key in args.keys():
        #    print("{}: {}".format(arg, getattr(args, arg)))
        print("=========================================\n")
        self.tensor_para_size = args['tensor_para_size']
        self.pipeline_para_size = 1
        self.max_batch_size = args['max_batch_size']
        self.random_seed_tensor = torch.zeros([self.max_batch_size], dtype=torch.int64)
        self.task_info={
            "prompt_seqs": None,
            "output_len":16,
            "beam_width": 1,
            "top_k": 50,
            "top_p": 0,
            "beam_search_diversity_rate": 0,
            "temperature": 0.1,
            "len_penalty": 0,
            "repetition_penalty": 1.0,
            "return_output_log_probs": 0,
            "return_cum_log_probs":0,
            "return_cross_attentions":0
        }
        self.tokenizer = AutoTokenizer.from_pretrained(args['hf_model_name'], use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        ckpt_path = args['ckpt_path']
        ckpt_config = configparser.ConfigParser()
        ckpt_config_path = os.path.join(ckpt_path, 'config.ini')
        if os.path.isfile(ckpt_config_path):
            ckpt_config.read(ckpt_config_path)
        else:
            assert False, "[ERROR] This example only support loading model with FT format directly."
        weight_data_type = np.float16
        relative_attention_max_distance = 128
        self.end_id = ckpt_config.getint("decoder", "eos_token_id")
        encoder_config = T5Config(vocab_size=ckpt_config.getint("encoder", "vocab_size"),
                                  d_model=ckpt_config.getint("encoder", "d_model"),
                                  d_kv=ckpt_config.getint("encoder", "d_kv"),
                                  d_ff=ckpt_config.getint("encoder", "d_ff"),
                                  num_layers=ckpt_config.getint("encoder", "num_layers"),
                                  num_decoder_layers=ckpt_config.getint("encoder", "num_decoder_layers"),
                                  num_heads=ckpt_config.getint("encoder", "num_heads"),
                                  relative_attention_num_buckets=ckpt_config.getint(
                                      "encoder", "relative_attention_num_buckets_or_max_pos_seq_len"),
                                  feed_forward_proj=ckpt_config.get("encoder", "feed_forward_proj"),
                                  pad_token_id=ckpt_config.getint("encoder", "pad_token_id"),
                                  eos_token_id=ckpt_config.getint("encoder", "eos_token_id"),
                                  is_gated_act=ckpt_config.getboolean("encoder", "is_gated_act", fallback=0),
                                  )
        decoder_config = T5Config(vocab_size=ckpt_config.getint("decoder", "vocab_size"),
                                  d_model=ckpt_config.getint("decoder", "d_model"),
                                  d_kv=ckpt_config.getint("decoder", "d_kv"),
                                  d_ff=ckpt_config.getint("decoder", "d_ff"),
                                  num_layers=ckpt_config.getint("decoder", "num_layers"),
                                  num_decoder_layers=ckpt_config.getint("decoder", "num_decoder_layers"),
                                  num_heads=ckpt_config.getint("decoder", "num_heads"),
                                  relative_attention_num_buckets=ckpt_config.getint(
                                      "decoder", "relative_attention_num_buckets_or_max_pos_seq_len"),
                                  feed_forward_proj=ckpt_config.get("decoder", "feed_forward_proj"),
                                  pad_token_id=ckpt_config.getint("decoder", "pad_token_id"),
                                  eos_token_id=ckpt_config.getint("decoder", "eos_token_id"),
                                  decoder_start_token_id=ckpt_config.getint("decoder", "decoder_start_token_id"),
                                  is_gated_act=ckpt_config.getboolean("decoder", "is_gated_act", fallback=0),
                                  )
        
        assert decoder_config.feed_forward_proj == encoder_config.feed_forward_proj
        assert decoder_config.feed_forward_proj == encoder_config.feed_forward_proj
        
        t5_with_bias = ckpt_config.getboolean("structure", "t5_with_bias")
        use_gated_activation = encoder_config.is_gated_act
        position_embedding_type = 0 if ckpt_config.get('structure', 'position_embedding_type') == 'relative' else 1
        activation_type = encoder_config.feed_forward_proj
        lib_path = '/workspace/Port_FasterTransformer/build/lib/libth_t5.so'

        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L1660
        # if tie_word_embeddings == True, scale the decoder output by sequence_output = sequence_output * (self.model_dim**-0.5)
        tie_word_embeddings = ckpt_config.getboolean("decoder", "tie_word_embeddings")
        ft_encoder_weight = FTT5EncoderWeight(
            encoder_config,
            self.tensor_para_size,
            self.pipeline_para_size,
            t5_with_bias=t5_with_bias,
            use_gated_activation=use_gated_activation,
            position_embedding_type=position_embedding_type,
            weight_data_type=weight_data_type
        )
        ft_decoding_weight = FTT5DecodingWeight(
            decoder_config,
            self.tensor_para_size,
            self.pipeline_para_size,
            t5_with_bias=t5_with_bias,
            use_gated_activation=use_gated_activation,
            position_embedding_type=position_embedding_type,
            weight_data_type=weight_data_type,
        )
        
        start_time = datetime.datetime.now()
        ft_encoder_weight.load_from_bin(ckpt_path)
        stop_time = datetime.datetime.now()
        print(f"[INFO] load FT encoder model spend {(stop_time - start_time).total_seconds()} sec")
        start_time = datetime.datetime.now()
        ft_decoding_weight.load_from_bin(ckpt_path)
        stop_time = datetime.datetime.now()
        print(f"[INFO] load FT decoding model spend {(stop_time - start_time).total_seconds()} sec")
        ft_encoder_weight.to_half()
        ft_decoding_weight.to_half()
        
        ft_encoder_weight.to_cuda()
        ft_decoding_weight.to_cuda()

        q_scaling = 1.0 / (math.sqrt(encoder_config.d_kv))
        remove_padding = True
        torch.manual_seed(0)
        with torch.no_grad():
            ft_encoder = FTT5Encoder(ft_encoder_weight.w, lib_path, encoder_config.num_heads,
                                    encoder_config.d_kv, encoder_config.d_ff,
                                    encoder_config.d_model, remove_padding, encoder_config.num_layers,
                                    encoder_config.relative_attention_num_buckets,
                                    relative_attention_max_distance, False, q_scaling, self.tensor_para_size,
                                    self.pipeline_para_size, t5_with_bias,
                                    position_embedding_type, activation_type=activation_type)

            ft_decoding = FTT5Decoding(ft_decoding_weight.w, lib_path,
                                    decoder_config.num_heads, decoder_config.d_kv,
                                    decoder_config.d_ff, encoder_config.d_model,
                                    decoder_config.d_model, decoder_config.num_layers,
                                    decoder_config.decoder_start_token_id, decoder_config.eos_token_id,
                                    decoder_config.vocab_size, q_scaling,
                                    decoder_config.relative_attention_num_buckets, max_distance=relative_attention_max_distance,
                                    tensor_para_size=self.tensor_para_size, pipeline_para_size=self.pipeline_para_size,
                                    t5_with_bias=t5_with_bias, position_embedding_type=position_embedding_type,
                                    activation_type=activation_type, tie_word_embeddings=tie_word_embeddings)

            self.ft_t5 = FTT5(ft_encoder, ft_decoding)
        print(f"<FastT5Inference.__init__> initialization done <{args['hf_model_name']}>")
        
    def _sync_task_info(self):
        print(f"<FastT5Inference._sync_task_info> enter rank-<{dist.get_rank()}>")
        dist.barrier()
        if dist.get_rank() == 0:
            dist.broadcast_object_list([self.task_info], src=0)
        else:
            info = [None]
            torch.distributed.broadcast_object_list(info, src=0)
            self.task_info = info[0]
        dist.barrier()
        print(f"<FastT5Inference._sync_task_info> leave rank-<{dist.get_rank()}, task_info:{self.task_info}>")
    
    def dispatch_request(self, args, env) -> Dict:
        print(f"dispatch_request get {args}")
        args = args[0]
        args = {k: v for k, v in args.items() if v is not None}
        # Inputs
        self.task_info["prompt_seqs"] = [args['prompt']]
        self.task_info["output_len"] = get_int(args.get("max_tokens", 16), default=16)
        self.task_info["beam_width"] = get_int(args.get("beam_width", 1), default=1)
        self.task_info["top_k"] = get_int(args.get("top_k", 50), default=50)
        self.task_info["top_p"] = get_float(args.get("top_p", 0.0), default=0.0)
        self.task_info["beam_search_diversity_rate"] = get_float(args.get("beam_search_diversity_rate", 0.0), default=0.0)
        self.task_info["temperature"] = get_float(args.get("temperature", 0.8), default=0.1)
        self.task_info["len_penalty"] = get_float(args.get("len_penalty", 0.0), default=0.0)
        self.task_info["repetition_penalty"] = get_float(args.get("repetition_penalty", 1.0), default=1.0)
        self.task_info["stop"] = args.get("stop", [])
        self.task_info["stream_tokens"] = args.get("stream_tokens", False)
        self.task_info["return_output_log_probs"] = args.get("return_output_log_probs", 0)
        self.task_info["return_cum_log_probs"] = args.get("return_cum_log_probs", 0)
        self.task_info["return_cross_attentions"] = args.get("return_cross_attentions", 0)
        
        
        if len(self.task_info["prompt_seqs"][0]) == 0 or self.task_info["output_len"] == 0:
            inferenece_result = []
            item = {'choices': [], }
            for beam_id in range(self.task_info["beam_width"]):
                choice = {
                    "text": '',
                    "index": beam_id,
                    "finish_reason": "length"
                }
            item['choices'].append(choice)
            inferenece_result.append(item)
            #  So far coordinator does not support batch. 
            result = {
                "result_type": RequestTypeLanguageModelInference,
                "choices": inferenece_result[0]['choices'],
                "raw_compute_time": 0.0
            }
            print(f"<FastT5Inference.dispatch_request> (not FT runs, 0 input or output) return: {result}")
            return result
        else:
            self._sync_task_info()
            result = self._run_inference()
            print(f"<FastT5Inference.dispatch_request> return: {result}")
            return result
        
    def _run_inference(self):
        print(f"<FastOPTInference._run_inference> enter rank-<{dist.get_rank()}>")
        
        with torch.no_grad():
            contexts = self.task_info["prompt_seqs"]
            line_tokens = self.tokenizer(contexts, return_tensors='pt')
            
            time = timeit.default_timer()
            max_batch_size = self.max_batch_size
            tokens_batch, ft_output_len = self.ft_t5(
                                      line_tokens,
                                      None,
                                      self.task_info["beam_width"],
                                      self.task_info["output_len"],
                                      self.task_info["top_k"] * torch.ones(size=[max_batch_size], dtype=torch.int32),
                                      self.task_info["top_p"] * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                      self.task_info["beam_search_diversity_rate"] * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                      self.task_info["temperature"] * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                      self.task_info["len_penalty"] * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                      self.task_info["repetition_penalty"] * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                      self.random_seed_tensor,
                                      False, #self.task_info["return_output_log_probs"],
                                      False, #self.task_info["return_cum_log_probs"],
                                      False #self.task_info["return_cross_attentions"]
                                      )
            # only a thread (rank 0) gets the output, while the others are supposed to return None.
            time_elapsed = timeit.default_timer() - time
        
        if dist.get_rank() == 0:
            print("[INFO] T5 time costs: {:.2f} ms. ft_output_len: <{}>".format(time_elapsed * 1000, ft_output_len))
            
            assert tokens_batch is not None
        
            inferenece_result = []
            
            for i in range(len(contexts)):
                item = {'choices': [], }
                for beam_id in range(self.task_info["beam_width"]):
                    token = tokens_batch[i, beam_id][: ft_output_len[i, beam_id]]  # exclude context input from the output
                    output = self.tokenizer.decode(token)
                    print(f"[INFO] batch {i}, beam {beam_id}: \n[Context]\n{contexts[i]}\n\n[Output]\n{output}\n")
                    choice = {
                        "text": post_processing_text(output, self.task_info["stop"]),
                        "index": beam_id,
                        "finish_reason": "length"
                    }
                item['choices'].append(choice)
                inferenece_result.append(item)
            #  So far coordinator does not support batch. 
            return {
                "result_type": RequestTypeLanguageModelInference,
                "choices": inferenece_result[0]['choices'],
                "raw_compute_time": time_elapsed
            }
        else:
            return None
    
    def worker(self):
        while True:
            self._sync_task_info()
            self._run_inference()


if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--together_model_name', type=str, default='Together-t5-11b',
                        help='worker name for together coordinator.')
    parser.add_argument('--hf_model_name', type=str, default='t5-11b',
                        help='hugging face model name (used to load config).')
    parser.add_argument('--ckpt_path', type=str, default='/workspace/Port_FasterTransformer/build/model/t5-11b-tp2/2-gpu',
                        help='path to the checkpoint file.')
    # parser.add_argument('--worker_name', type=str, default='worker1',
    #                      help='worker name for together coordinator.')
    parser.add_argument('--tensor_para_size', type=int, default=2,
                        help='tensor parallel size')
    parser.add_argument('--group_name', type=str, default='group1',
                        help='group name for together coordinator.')
    
    args = parser.parse_args()
    
    coord_url = os.environ.get("COORD_URL", "localhost")
    coordinator = TogetherWeb3(
        TogetherClientOptions(reconnect=True),
        http_url=f"http://{coord_url}:8092",
        websocket_url=f"ws://{coord_url}:8093/websocket"
    )
    fip = FastT5Inference(model_name=args.together_model_name, args={
        "coordinator": coordinator,
        "hf_model_name": args.hf_model_name,
        "tensor_para_size":args.tensor_para_size,
        "group_name": args.group_name,
        "ckpt_path": args.ckpt_path,
        "stream_tokens_pipe": False,
        "max_batch_size":1
    })
    fip.start()
