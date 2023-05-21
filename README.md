# Deploy FT Inference of RedPajama Models Under TogetherCompute Infra

### Build the docker image:

    sudo docker build -t ft_redpajama --file Redpajama-Together-Dockerfile .

### Convert RedPajama model to FT format:

- Download the checkpoint of RedPajama model from Hugging Face (e.g., RedPajama-INCITE-Chat-7B-v0.1):


        git lfs clone https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-7B-v0.1


- Start the ft_redpajama container:


        sudo nvidia-docker run  --ipc=host --network=host --name test_ft_redpajama -ti -v /PATH_TO_PARENT_DIR_OF_DOWNLOADED_HF_WEIGHTS:/workspace/FasterTransformer/build/model ft_redpajama bash


- Run the converting script inside the container:


        python /workspace/FasterTransformer/examples/pytorch/gptneox/utils/huggingface_gptneox_convert.py -i /workspace/FasterTransformer/build/model/RedPajama-INCITE-Chat-7B-v0.1 -o /workspace/FasterTransformer/build/model/ft-RedPajama-INCITE-Chat-7B-v0.1 -i_g 1 -m_n RedPajama-INCITE-Chat-7B-v0.1 -weight_data_type fp16


### To deploy the model:


- Inside the container, start the together node:

        /usr/local/bin/together-node start


- Inside the container, start the worker process (probably need to change some args to support different models):


        python /workspace/FasterTransformer/examples/pytorch/gptneox/serving_redpajama_single_gpu.py