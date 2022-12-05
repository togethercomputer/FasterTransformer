import logging
import os
from typing import Dict
import sys
import timeit
from common.fast_inference import FastInferenceInterface
from common.together_web3.computer import RequestTypeLanguageModelInference
from common.together_web3.together import TogetherWeb3, TogetherClientOptions
import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../..")
from examples.pytorch.gpt.utils.parallel_gpt import ParallelGPT
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


class FastOPTInference(FastInferenceInterface):
    def __init__(self, model_name: str, args=None) -> None:
        try:
            if not dist.is_initialized():
                dist.init_process_group(backend='mpi')
        except:
            print("[INFO] WARNING: Have initialized the process group")
            
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
            "return_cum_log_probs": 0,
            "return_output_length":0,
        }
        
        hf_config = vars(AutoConfig.from_pretrained(args['hf_model_name']))
        head_num = hf_config['num_attention_heads']
        layer_num = hf_config['num_hidden_layers']
        start_id = hf_config['bos_token_id']
        self.end_id = hf_config['eos_token_id']
        size_per_head = hf_config['hidden_size'] // head_num
        vocab_size = 50272
        max_seq_len = 2048
        layernorm_eps = 1e-5
        layernorm_type = 'pre_layernorm' if hf_config['do_layer_norm_before'] else 'post_layernorm'
        activation_type = 'Relu' if hf_config['activation_function'] == 'relu' else 'Gelu'
        has_post_decoder_layernorm = layernorm_type == 'pre_layernorm'
        lib_path = '/workspace/Port_FasterTransformer/build/lib/libth_parallel_gpt.so'
        ckpt_path = '/workspace/Port_FasterTransformer/build/model/opt-66b-fp16-tp8/8-gpu'
        self.tokenizer = AutoTokenizer.from_pretrained(args['hf_model_name'])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        torch.manual_seed(0)
        with torch.no_grad():
            # Prepare model.
            self.opt_model = ParallelGPT(head_num, size_per_head, vocab_size, start_id, self.end_id, layer_num,
                                         max_seq_len, self.tensor_para_size, self.pipeline_para_size, lib_path,
                                         layernorm_eps, layernorm_type, activation_type, has_post_decoder_layernorm,
                                         int8_mode=0, weights_data_type='fp16')
            if not self.opt_model.load(ckpt_path=ckpt_path):
                print("[WARNING] Checkpoint file not found. Model loading is skipped.")
                
        print(f"<FastOPTInference.__init__> rank {dist.get_rank()} initialization done")

    def _sync_task_info(self):
        print(f"<FastOPTInference._sync_task_info> enter rank-<{dist.get_rank()}>")
        dist.barrier()
        if dist.get_rank() == 0:
            dist.broadcast_object_list([self.task_info], src=0)
        else:
            info = [None]
            torch.distributed.broadcast_object_list(info, src=0)
            self.task_info = info[0]
        dist.barrier()
        print(f"<FastOPTInference._sync_task_info> leave rank-<{dist.get_rank()}, task_info:{self.task_info}>")
        
    def dispatch_request(self, args, env) -> Dict:
        print(f"Rank {dist.get_rank()} get {args}")
        args = args[0]
        args = {k: v for k, v in args.items() if v is not None}
        # Inputs
        self.task_info["prompt_seqs"] = [args['prompt']]
        self.task_info["output_len"] = args.get("max_tokens", 16)
        self.task_info["beam_width"] = args.get("beam_width", 1)
        self.task_info["top_k"] = args.get("top_k", 50)
        self.task_info["top_p"] = args.get("top_p", 0)
        self.task_info["beam_search_diversity_rate"] = args.get("beam_search_diversity_rate", 0)
        self.task_info["temperature"] = args.get("temperature", 0.1)
        self.task_info["len_penalty"] = args.get("len_penalty", 0)
        self.task_info["repetition_penalty"] = args.get("repetition_penalty", 1.0)
        self.task_info["return_cum_log_probs"] = args.get("return_cum_log_probs", 0)
        self.task_info["return_output_length"] = args.get("return_output_length", 0)
        self.task_info["stream_tokens"] = args.get("stream_tokens", False)
        
        self._sync_task_info()
        result = self._run_inference()
        print(f"<FastOPTInference.dispatch_request> return: {result}")
        return result

    def _run_inference(self):
        print(f"<FastOPTInference._run_inference> enter rank-<{dist.get_rank()}>")
        
        with torch.no_grad():
            contexts = self.task_info["prompt_seqs"]
            start_ids = [torch.IntTensor(self.tokenizer.encode(c)) for c in contexts]
            start_lengths = [len(ids) for ids in start_ids]
            
            start_ids = pad_sequence(start_ids, batch_first=True, padding_value=self.end_id)
            start_lengths = torch.IntTensor(start_lengths)
            print(f"start_ids: length ({start_ids.shape[0]}) ids: {start_ids}")
            
            time = timeit.default_timer()
            max_batch_size = self.max_batch_size
            tokens_batch = self.opt_model(start_ids,
                                    start_lengths,
                                    self.task_info["output_len"],
                                    self.task_info["beam_width"],
                                    self.task_info["top_k"] * torch.ones(size=[max_batch_size], dtype=torch.int32),
                                    self.task_info["top_p"] * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                    self.task_info["beam_search_diversity_rate"] * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                    self.task_info["temperature"] * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                    self.task_info["len_penalty"] * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                    self.task_info["repetition_penalty"] * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                    self.random_seed_tensor,
                                    self.task_info["return_output_length"],
                                    self.task_info["return_cum_log_probs"],
                                    self.served,
                                    self.stream_tokens_pipe_w if self.task_info["stream_tokens"] else -1)
            # only a thread (rank 0) gets the output, while the others are supposed to return None.
            time_elapsed = timeit.default_timer() - time
        print("[INFO] OPT time costs: {:.2f} ms. <rank-{}>".format(time_elapsed * 1000, dist.get_rank()))
        
        if dist.get_rank() == 0:
            assert tokens_batch is not None
        
            if self.task_info["return_cum_log_probs"] > 0:
                tokens_batch, _, cum_log_probs = tokens_batch
                print('[INFO] Log probs of sentences:', cum_log_probs)

            inferenece_result = []
            tokens_batch = tokens_batch.cpu().numpy()
            
            for i, (context, tokens) in enumerate(zip(self.task_info["prompt_seqs"], tokens_batch)):
                item = {'choices': [], }
                for beam_id in range(self.task_info["beam_width"]):
                    token = tokens[beam_id][start_lengths[i]:]  # exclude context input from the output
                    output = self.tokenizer.decode(token)
                    print(f"[INFO] batch {i}, beam {beam_id}: \n[Context]\n{context}\n\n[Output]\n{output}\n")
                    choice = {
                        "text": output,
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
    coord_url = os.environ.get("COORD_URL", "127.0.0.1")
    coordinator = TogetherWeb3(
        TogetherClientOptions(reconnect=True),
        http_url=f"http://{coord_url}:8092",
        websocket_url=f"ws://{coord_url}:8093/websocket"
    )
    fip = FastOPTInference(model_name="opt66b", args={
        "coordinator": coordinator,
        "hf_model_name": "facebook/opt-66b",
        "tensor_para_size":8,
        "max_batch_size":1,
        "stream_tokens_pipe": True,
    })
    fip.start()
