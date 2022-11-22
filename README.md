docker build -t port_fasttransformer .

nvidia-docker run  --ipc=host --network=host --name port_ft -ti -v /root/fm/models/ft_model:/workspace/Port_FasterTransformer/build/model -v /root/fm/dev/Port_FasterTransformer/examples/pytorch/gpt:/workspace/Port_FasterTransformer/examples/pytorch/gpt port_fasttransformer  bash

nvidia-docker run  --ipc=host --network=host --name port_ft -ti -v /root/fm/models/ft_model:/workspace/Port_FasterTransformer/build/model -v /root/fm/dev/Port_FasterTransformer/examples/pytorch/gptj:/workspace/Port_FasterTransformer/examples/pytorch/gptj port_fasttransformer  bash


mpirun -n 8 --allow-run-as-root python /workspace/Port_FasterTransformer/examples/pytorch/gpt/port_opt_inference.py --weights_data_type fp16 --data_type fp16 --vocab_size 50272 --max_batch_size 1 --max_seq_len 2048 --tensor_para_size 8 --ckpt_path /workspace/Port_FasterTransformer/build/model/opt-66b-fp16-tp8/8-gpu --lib_path /workspace/Port_FasterTransformer/build/lib/libth_parallel_gpt.so --vocab_file /workspace/Port_FasterTransformer/build/model/gpt2-vocab.json --merges_file /workspace/Port_FasterTransformer/build/model/gpt2-merges.txt --sample_input_file /workspace/Port_FasterTransformer/build/model/foo_txt_16.txt


mpirun -n 8 --allow-run-as-root python /workspace/Port_FasterTransformer/examples/pytorch/gpt/together_opt_inference.py