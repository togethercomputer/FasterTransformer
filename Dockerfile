FROM nvcr.io/nvidia/pytorch:22.04-py3

WORKDIR /workspace/Port_FasterTransformer
ADD . /workspace/Port_FasterTransformer

RUN mkdir -p build && \
    cd build && \
    git submodule update --init --recursive && \
    cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON .. &&\
    make && \
    pip install -r ../examples/pytorch/gpt/requirement.txt && \
    pip install -r ../examples/pytorch/gpt/common/requirement.txt && \
    rm -r ../src/fastertransformer/th_op/gptj/* && \ 
    rm -r ../examples/pytorch/gpt/*  && \
    rm -r ../examples/pytorch/gptj/*


