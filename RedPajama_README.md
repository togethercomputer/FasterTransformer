# Deploy FT Inference of RedPajama Models Under TogetherCompute Infra

## Build the docker image

```shell
$ sudo docker build -t ft_redpajama --file ./docker/Dockerfile.Together-Redpajama .
```

## Convert RedPajama model to FT format

- Download the checkpoint of RedPajama model from Hugging Face (e.g., RedPajama-INCITE-7B-Chat):

```shell
$ git lfs clone https://huggingface.co/togethercomputer/RedPajama-INCITE-7B-Chat
```

- Start the ft_redpajama container:

```shell
$ sudo nvidia-docker run  --ipc=host --network=host --name test_ft_redpajama -ti -v /PATH_TO_PARENT_DIR_OF_DOWNLOADED_HF_WEIGHTS:/workspace/FasterTransformer/build/model ft_redpajama bash
```

- Run the converting script inside the container:

```shell
$ python /workspace/FasterTransformer/examples/pytorch/gptneox/utils/huggingface_gptneox_convert.py -i /workspace/FasterTransformer/build/model/RedPajama-INCITE-7B-Chat -o /workspace/FasterTransformer/build/model/ft-RedPajama-INCITE-7B-Chat -i_g 1 -m_n RedPajama-INCITE-7B-Chat -weight_data_type fp16
```

## To deploy the model

- Inside the container, start the together node:

```shell
$ /usr/local/bin/together-node start
```

- Inside the container, start the worker process (probably need to change some args to support different models):

```shell
$ python /workspace/FasterTransformer/examples/pytorch/gptneox/serving_redpajama_single_gpu.py
```
