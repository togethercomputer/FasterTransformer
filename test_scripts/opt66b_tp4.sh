export CUDA_VISIBLE_DEVICES=0,1,2,3

ARGS="--weights_data_type fp16 \
      --data_type fp16 \
      --max_seq_len 2048 \
      --layer_num 64 \
      --head_num 72 \
      --size_per_head 128 \
      --tensor_para_size 4 \
      --ckpt_path /workspace/Port_FasterTransformer/build/model/opt-66b-fp16-tp4/4-gpu \
      --lib_path /workspace/Port_FasterTransformer/build/lib/libth_gpt.so \
      --vocab_file /workspace/Port_FasterTransformer/build/model/gpt2-vocab.json \
      --merges_file /workspace/Port_FasterTransformer/build/model/gpt2-merges.txt \
      --sample_input_file /workspace/Port_FasterTransformer/build/model/foo_txt_128.txt \
      --time"

mpirun -n 4 --allow-run-as-root python examples/pytorch/gpt/gpt_example.py $ARGS