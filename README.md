# Port_FasterTransformer 

To bring up a standalone node:

```console
mkdir .together
docker run --rm --gpus all \
  -e NUM_WORKERS=auto \
  -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
  -v $PWD/.together:/home/user/.together \
  -it togethercomputer/fastertransformer /usr/local/bin/together start \
    --color --config /home/user/cfg.yaml \
    --worker.model GPT-JT-6B-v1-tp1 --worker.model_type gptj
```

```console
docker run --rm --gpus '"device=3,4"' --ipc=host \
  -e NUM_WORKERS=auto \
  -v $PWD/.together:/home/user/.together \
  -it togethercomputer/fastertransformer /usr/local/bin/together start \
    --color --config /home/user/cfg.yaml \
    --worker.model opt-13b-tp2 --worker.model_type gpt
```

# Development commands

```console
docker build -t port_ft_gpt_jt -f GPT-JT-Dockerfile 

nvidia-docker run  --ipc=host --network=host --name port_ft -ti -v /root/fm/models/ft_model:/workspace/Port_FasterTransformer/build/model -v  /root/fm/dev/Port_FasterTransformer/examples/:/workspace/Port_FasterTransformer/examples -v  /root/fm/dev/Port_FasterTransformer/src/fastertransformer:/workspace/Port_FasterTransformer/src/fastertransformer  port_ft  bash

nvidia-docker run  --ipc=host --network=host --name port_ft -ti -v /home/binhang/active/ft_model:/workspace/Port_FasterTransformer/build/model -v  /home/binhang/active/Port_FasterTransformer/examples:/workspace/Port_FasterTransformer/examples -v  /home/binhang/active/Port_FasterTransformer/src/fastertransformer:/workspace/Port_FasterTransformer/src/fastertransformer  port_fasttransformer bash

mpirun -n 8 --allow-run-as-root python /workspace/Port_FasterTransformer/examples/pytorch/gpt/port_opt_inference.py --weights_data_type fp16 --data_type fp16 --vocab_size 50272 --max_batch_size 1 --max_seq_len 2048 --tensor_para_size 8 --ckpt_path /workspace/Port_FasterTransformer/build/model/opt-66b-fp16-tp8/8-gpu --lib_path /workspace/Port_FasterTransformer/build/lib/libth_parallel_gpt.so --vocab_file /workspace/Port_FasterTransformer/build/model/gpt2-vocab.json --merges_file /workspace/Port_FasterTransformer/build/model/gpt2-merges.txt --sample_input_file /workspace/Port_FasterTransformer/build/model/foo_txt_16.txt

mpirun -n 8 --allow-run-as-root python /workspace/Port_FasterTransformer/examples/pytorch/gpt/together_opt_inference.py

mpirun -n 1 --allow-run-as-root python /workspace/Port_FasterTransformer/examples/pytorch/gptj/app/gptj_example.py --weights_data_type fp32 --infer_data_type fp16 --tensor_para_size 1 --ckpt_path /workspace/Port_FasterTransformer/build/model/GPT-JT-6B-v1-tp1/1-gpu --lib_path /workspace/Port_FasterTransformer/build/lib/libth_gptj.so --sample_input_file /workspace/Port_FasterTransformer/build/model/foo_txt_16.txt

mpirun -n 2 --allow-run-as-root python /workspace/Port_FasterTransformer/examples/pytorch/gptneox/app/gptneox_example.py --use_gptj_residual --weights_data_type fp32 --infer_data_type fp16 --tensor_para_size 2 --ckpt_path /workspace/Port_FasterTransformer/build/model/gpt-neox-20b-tp2/2-gpu --lib_path /workspace/Port_FasterTransformer/build/lib/libth_gptneox.so --sample_input_file /workspace/Port_FasterTransformer/build/model/foo_txt_16.txt

mpirun -n 1 --allow-run-as-root python /workspace/Port_FasterTransformer/examples/pytorch/gptneox/app/gptneox_example.py --use_gptj_residual --weights_data_type fp32 --infer_data_type fp16 --tensor_para_size 1 --ckpt_path /workspace/Port_FasterTransformer/build/model/gpt-neox-20b-tp1/1-gpu --lib_path /workspace/Port_FasterTransformer/build/lib/libth_gptneox.so --sample_input_file /workspace/Port_FasterTransformer/build/model/foo_txt_16.txt
```
