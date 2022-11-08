docker build -t port_fasttransformer .

nvidia-docker run --name port_ft -ti -v /root/fm/models/ft_model:/workspace/Port_FasterTransformer/build/model -v /root/fm/dev/Port_FasterTransformer/examples/pytorch/gpt:/workspace/Port_FasterTransformer/examples/pytorch/gpt port_fasttransformer  bash
