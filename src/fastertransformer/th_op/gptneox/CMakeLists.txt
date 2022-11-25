add_library(th_gptneox SHARED GptNeoXOp.cc)
target_link_libraries(th_gptneox PRIVATE "${TORCH_LIBRARIES}" GptNeoX th_utils nccl_utils)
