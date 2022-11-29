import argparse
import time
from toma_client.coordinator_client import LocalCoordinatorClient
import traceback
from loguru import logger
from transformers import AutoTokenizer, AutoConfig
import math
import numpy as np
import random
import torch
import os
from utils.gptj import GPTJ
from torch.nn.utils.rnn import pad_sequence
import timeit


def post_processing_text(output_text, stop_tokens):
    print(f"<post_processing_text> output_text: {output_text}")

    filtered_stop_tokens = []
    for token in stop_tokens:
        if token != '':
            filtered_stop_tokens.append(token)
            
    print(f"<post_processing_text> stop_tokens: {filtered_stop_tokens}.")

    end_pos = len(output_text)
    print(f"<post_processing_text>1 end_pos: {end_pos}.")
    for stop_token in filtered_stop_tokens:
        if output_text.find(stop_token) != -1:
            end_pos = min(output_text.find(stop_token), end_pos)

    print(f"<post_processing_text>2 end_pos: {end_pos}.")
    print(f"<post_processing_text> text: {output_text}, end_pos: {end_pos}")
    post_processed_text = output_text[:end_pos]
    print(f"<post_processing_text> input: {output_text}")
    print(f"<post_processing_text> output: {post_processed_text}")
    return post_processed_text


def main():
    parser = argparse.ArgumentParser(description='Local Inference Runner with coordinator.')
    parser.add_argument('--job_id', type=str, default='-', metavar='S',
                        help='DB ID')
    parser.add_argument('--max-batch-size', type=int, default=1, metavar='S',
                        help='batch-size for inference (default:8)')
    parser.add_argument('--together_model_name', type=str, default='Together-gpt-JT-6B-v1',
                        help='worker name for together coordinator.')
    parser.add_argument('--hf_model_name', type=str, default='togethercomputer/GPT-JT-6B-v1',
                        help='hugging face model name (used to load config).')
    parser.add_argument('--ckpt_path', type=str, default='/workspace/Port_FasterTransformer/build/model/GPT-JT-6B-v1-tp1/1-gpu',
                        help='path to the checkpoint file.')
    args = parser.parse_args()
    print("\n=============== Arguments ===============")
    print(args)
    print("=========================================\n")
    local_cord_client = LocalCoordinatorClient(
        working_directory="/workspace/Port_FasterTransformer/build/model/workding_dir",
        coordinator_url="http://localhost:5000/eth",
    )
    assert (torch.cuda.is_available())
    try:
        max_batch_size = args.max_batch_size
        random_seed_tensor = torch.zeros([max_batch_size], dtype=torch.int64)
        task_info={
            "prompt_seqs": None,
            "output_len":16,
            "beam_width": 1,
            "top_k": 50,
            "top_p": 0,
            "beam_search_diversity_rate": 0,
            "temperature": 0.8,
            "len_penalty": 0,
            "repetition_penalty": 1.0,
            "return_cum_log_probs": 0,
            "return_output_length":0,
        }
        
        hf_config = vars(AutoConfig.from_pretrained(args.hf_model_name))
        head_num = hf_config['n_head']   
        layer_num = hf_config['n_layer']
        start_id = hf_config['bos_token_id']
        end_id = hf_config['eos_token_id']
        size_per_head = hf_config['n_embd'] // head_num
        vocab_size = hf_config['vocab_size']
        rotary_embedding_dim = hf_config['rotary_dim']
        max_seq_len = hf_config['n_positions']
        lib_path = '/workspace/Port_FasterTransformer/build/lib/libth_gptj.so'
        ckpt_path = args.ckpt_path
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        torch.manual_seed(0)
        
        # Prepare model.
        gptj_model = GPTJ(head_num, size_per_head, layer_num, vocab_size, rotary_embedding_dim, 
                          start_id, end_id, max_seq_len, 1, 1, lib_path=lib_path, weights_data_type='fp32')
   
        if not gptj_model.load(ckpt_path=ckpt_path, infer_data_type='fp16'):
            print("[WARNING] Checkpoint file not found. Model loading is skipped.")
        torch.cuda.empty_cache()
        print(f"<FastGPTJInference.__init__> initialization done")
        
    except Exception as e:
        print('Exception in model initialization inference:', e)
        error = traceback.format_exc()
        local_cord_client.update_status(args.job_id, "failed", returned_payload={"message": error})
        print(error)
        raise e

    try:
        while True:
            job_id = None
            raw_text = None
            try:
                instructions = local_cord_client.fetch_instructions(args.together_model_name, 0)
                last_instruction = instructions[-1]

                if last_instruction["message"] == "break":
                    logger.info("Received stop instruction.")
                    logger.info("# BREAK ")
                    break
                elif last_instruction["message"] == "continue":
                    #logger.info(f"Received keep instruction. <{args.together_model_name}>")
                    time.sleep(0.2)
                elif last_instruction["message"] == "run":
                    fetched_tasks = [x for x in instructions
                                     if x["message"] == "run" and x['payload']['status'] == 'submitted']

                    if len(fetched_tasks) > 0:
                        instruction = fetched_tasks[0]
                        logger.info("Instruction:")
                        logger.info(str(instruction))
                        # TODO: we assume len(payload) is 1, right?
                        query = instruction['payload']['payload'][0]
                        if isinstance(query['prompt'], list):
                            raw_text = query['prompt'][0]
                        elif isinstance(query['prompt'], str):
                            raw_text = query['prompt']
                        else:
                            print("wrong prompt format, it can only be str or list of str")
                            print(query['prompt'])

                        job_id = instruction['payload']['id']
                        print(f"Job <{job_id}> has been processed")

                        start_time = time.time()
                        task_info["prompt_seqs"] = [raw_text]
                        task_info["output_len"] = int(query.get("max_tokens", 16))
                        task_info["beam_width"] = int(query.get("beam_width", 1))
                        task_info["top_k"] = int(query.get("top_k", 50))
                        task_info["top_p"] = float(query.get("top_p", 0.0))
                        task_info["beam_search_diversity_rate"] = float(query.get("beam_search_diversity_rate", 0))
                        task_info["temperature"] = float(query.get("temperature", 0.1))
                        task_info["len_penalty"] = float(query.get("len_penalty", 0))
                        task_info["repetition_penalty"] = float(query.get("repetition_penalty", 1.0))
                        task_info["stop"] =  query.get("stop", [])
                        
                        with torch.no_grad():
                            contexts = task_info["prompt_seqs"]
                            start_ids = [torch.IntTensor(tokenizer.encode(c)) for c in contexts]
                            start_lengths = [len(ids) for ids in start_ids]
                            
                            start_ids = pad_sequence(start_ids, batch_first=True, padding_value=end_id)
                            start_lengths = torch.IntTensor(start_lengths)
                            print(f"start_ids: length ({start_ids.shape[0]}) ids: {start_ids}")
                            
                            max_batch_size = max_batch_size
                            print(task_info)
                            tokens_batch = gptj_model(start_ids,
                                                      start_lengths,
                                                      task_info["output_len"],
                                                      task_info["beam_width"],
                                                      task_info["top_k"] * torch.ones(size=[max_batch_size], dtype=torch.int32),
                                                      task_info["top_p"] * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                                      task_info["beam_search_diversity_rate"] * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                                      task_info["temperature"] * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                                      task_info["len_penalty"] * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                                      task_info["repetition_penalty"] * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                                      random_seed_tensor)
                            # only a thread (rank 0) gets the output, while the others are supposed to return None.                        
                        assert tokens_batch is not None
                        
                        if task_info["return_cum_log_probs"] > 0:
                            tokens_batch, _, cum_log_probs = tokens_batch
                            print('[INFO] Log probs of sentences:', cum_log_probs)

                        inferenece_result = []
                        tokens_batch = tokens_batch.cpu().numpy()
                        
                        for i, (context, tokens) in enumerate(zip(task_info["prompt_seqs"], tokens_batch)):
                            item = {'choices': [], }
                            for beam_id in range(task_info["beam_width"]):
                                token = tokens[beam_id][start_lengths[i]:]  # exclude context input from the output
                                print(f"[INFO] raw token: {token}")
                                output = tokenizer.decode(token)
                                print(f"[INFO] batch {i}, beam {beam_id}: \n[Context]\n{context}\n\n[Output]\n{output}\n")
                                choice = {
                                    "text": post_processing_text(output, task_info["stop"]),
                                    "index": beam_id,
                                    "finish_reason": "length"
                                }
                            item['choices'].append(choice)
                            inferenece_result.append(item)
                        #  So far coordinator does not support batch. 
                        returned_result = {"inference_result": inferenece_result}
                        
                        end_time = time.time()
                        
                        print(f"Job-{job_id} {args.hf_model_name} Inference takes {end_time - start_time}s")
                        print(f"returned result: {returned_result}")
                        return_payload = {
                            'request': query,
                            'result': returned_result,
                            'raw_compute_time': end_time - start_time
                        }
                        # local_cord_client.update_status(
                        local_cord_client.update_status_global_coordinator(
                            job_id,
                            "finished",
                            returned_payload=return_payload
                        )
                        local_cord_client.update_status(job_id, "finished", returned_payload={})

            except Exception as e:
                error = traceback.format_exc()
                local_cord_client.update_status(
                    job_id,
                    "failed",
                    returned_payload={"message": error}
                )
                print(error)

    except Exception as e:
        print('Exception in latency inference:', e)


if __name__ == "__main__":
    main()