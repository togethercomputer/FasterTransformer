# If there is a new model, first convert HF format to FT format by running: (may take 20 minutes)
# python /workspace/Port_FasterTransformer/examples/pytorch/gptneox/app/utils/huggingface_gptneox_convert.py -i /workspace/Port_FasterTransformer/build/model/raw_models/GPT-NeoXT-20B-chat-v0.8.1 -o /workspace/Port_FasterTransformer/build/model/gpt-neoxT-latest-fp32-tp2 -i_g 2 -weight_data_type fp32  -m_n GPT-NeoXT-fp32-latest -p 16

/workspace/Port_FasterTransformer/build/model/together start &

mpirun -n 2 --allow-run-as-root python /workspace/Port_FasterTransformer/examples/pytorch/gptneox/app/serving_multi_gpu.py --ckpt_path /workspace/Port_FasterTransformer/build/model/gpt-neoxT-latest-fp32-tp2/2-gpu --weights_data_type fp32 --together_model_name together/gpt-neoxT-20B-chat-latest-fp32 --tensor_para_size 2
