#pragma once

#include <cstdlib>
#include <map>
#include <string>

namespace fastertransformer {

class TokenPipe {
private:
    int pipe_fd_;
    int64_t request_id_;

public:
    TokenPipe(int pipe_fd, int64_t request_id) : pipe_fd_(pipe_fd), request_id_(request_id) {}

    void stream_tokens(std::unordered_map<std::string, Tensor>* tensors) {
        auto output_ids_iter = tensors->find("output_ids");
        auto sequence_length_iter = tensors->find("sequence_length");
        if (output_ids_iter == tensors->end()) return;
        if (sequence_length_iter == tensors->end()) return;
        auto &output_ids = output_ids_iter->second;
        auto &sequence_length = sequence_length_iter->second;
        // printf("output_ids size %ld, %ld\n", output_ids.size(), output_ids.sizeBytes());
        // printf("output_ids shape %ld, %ld, %ld\n", output_ids.shape[0], output_ids.shape[1], output_ids.shape[2]);
        int32_t *output_ids_cpu = (int32_t*)malloc(output_ids.sizeBytes());
        int32_t *sequence_length_cpu = (int32_t*)malloc(sequence_length.sizeBytes());
        cudaDeviceSynchronize();
        cudaMemcpy(output_ids_cpu, output_ids.getPtr<int32_t>(), output_ids.sizeBytes(), cudaMemcpyDeviceToHost);
        cudaMemcpy(sequence_length_cpu, sequence_length.getPtr<int32_t>(), sequence_length.sizeBytes(), cudaMemcpyDeviceToHost);
        int32_t token = output_ids_cpu[sequence_length_cpu[0]-1];
        free(output_ids_cpu);
        free(sequence_length_cpu);

	    char buf[4096];
        int len = snprintf(buf, sizeof(buf), "{ \"id\": %ld, \"token\": [ %d ] }\n", this->request_id_, token);
        printf("%d = write %d, \"%s\"\n", len, this->pipe_fd_, buf);
        write(this->pipe_fd_, buf, len);
    }

    static void stream_tokens_callback(std::unordered_map<std::string, Tensor>* tensors, void *opaque) {
        reinterpret_cast<TokenPipe*>(opaque)->stream_tokens(tensors);
    }
};

}  // namespace fastertransformer
