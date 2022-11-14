import os
from typing import Dict
import sys
from common.fast_inference import FastInferenceInterface
from common.together_web3.computer import ImageModelInferenceChoice, RequestTypeImageModelInference
from common.together_web3.together import TogetherWeb3, TogetherClientOptions
import torch
import torch.distributed as dist

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
        
        hf_config = vars(AutoConfig.from_pretrained(args['hf_model_name']))
        head_num = hf_config['num_attention_heads']
        layer_num = hf_config['num_hidden_layers']
        start_id = hf_config['bos_token_id']
        end_id = hf_config['eos_token_id']
        size_per_head = hf_config['hidden_size'] // head_num
        
        output_len = 1
        vocab_size = 50272
        beam_width = 1
        top_k = 50
        top_p = 0
        temperature = 0.9
        len_penalty = 0
        beam_search_diversity_rate = 0
        tensor_para_size = args['tensor_para_size']
        pipeline_para_size = 1
        max_batch_size = args['max_batch_size']
        max_seq_len = 2048
        repetition_penalty = 1
        int8_mode = 0
        weights_data_type = 'fp16'
        return_cum_log_probs = 0
        return_output_length = return_cum_log_probs > 0
        shared_contexts_ratio = 1.0
        layernorm_eps = 1e-5
        layernorm_type = 'pre_layernorm' if hf_config['do_layer_norm_before'] else 'post_layernorm'
        activation_type = 'Relu' if hf_config['activation_function'] == 'relu' else 'Gelu'
        has_post_decoder_layernorm = layernorm_type == 'pre_layernorm'
        tokenizer = AutoTokenizer.from_pretrained(args['hf_model_name'])
        tokenizer.pad_token = tokenizer.eos_token
        lib_path = '/workspace/Port_FasterTransformer/build/lib/libth_parallel_gpt.so'
        torch.manual_seed(0)
        with torch.no_grad():
            # Prepare model.
            self.opt_model = ParallelGPT(head_num, size_per_head, vocab_size, start_id, end_id, layer_num,
                                         max_seq_len, tensor_para_size, pipeline_para_size, lib_path,
                                         layernorm_eps, layernorm_type, activation_type, has_post_decoder_layernorm,
                                         int8_mode=0, weights_data_type=weights_data_type)
        print(f"<FastOPTInference.__init__> rank {dist.get_rank()} initialization done")

    def dispatch_request(self, args, env) -> Dict:
        print(f"Rank {dist.get_rank()} get {args}")
        args = args[0]
        # Inputs
        contexts = [args['prompt']]
        batch_size = 1
        start_ids = [torch.IntTensor(tokenizer.encode(c)) for c in contexts]
    else:  # unconditional case
        batch_size = max_batch_size
        contexts = ['<|endoftext|>'] * batch_size
        start_ids = [torch.IntTensor([end_id for _ in range(args.input_len)])] * batch_size

    start_lengths = [len(ids) for ids in start_ids]
    input_len = max(start_lengths)

    start_ids = pad_sequence(start_ids, batch_first=True, padding_value=end_id)
    start_lengths = torch.IntTensor(start_lengths)

        
         
        

if __name__ == "__main__":
    coord_url = os.environ.get("COORD_URL", "127.0.0.1")
    coordinator = TogetherWeb3(
        TogetherClientOptions(),
        http_url=f"http://{coord_url}:8092",
        websocket_url=f"ws://{coord_url}:8093/websocket"
    )
    fip = FastOPTInference(model_name="opt66b", args={
        "coordinator": coordinator,
        "hf_model_name": "facebook/opt-66b",
        "tensor_para_size":8,
        "max_batch_size":1
    })
    fip.start()