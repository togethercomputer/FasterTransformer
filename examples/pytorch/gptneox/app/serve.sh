# If there is a new model, first convert HF format to FT format by running: (may take 20 minutes)
# python /workspace/Port_FasterTransformer/examples/pytorch/gptneox/app/utils/huggingface_gptneox_convert.py -i /workspace/Port_FasterTransformer/build/model/raw_models/GPT-NeoXT-20B-chat-v0.8.1 -o /workspace/Port_FasterTransformer/build/model/gpt-neoxT-latest-tp1 -i_g 1 -weight_data_type fp16  -m_n GPT-NeoXT-latest -p 16


/workspace/Port_FasterTransformer/build/model/together start &

python /workspace/Port_FasterTransformer/examples/pytorch/gptneox/app/serving_single_gpu.py --ckpt_path /workspace/Port_FasterTransformer/build/model/gpt-neoxT-latest-tp1/1-gpu --weights_data_type fp16 --together_model_name together/gpt-neoxT-20B-chat-latest
