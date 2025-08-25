llama_add_compile_flags()

#
# libraries
#

# llama

add_library(llama
            ../include/llama.h
            llama.cpp
            llama-arch.cpp
            llama-batch.cpp
            llama-chat.cpp
            llama-context.cpp
            llama-context-load-utils.cpp
            llama-execute-decode.cpp
            llama-execute-kvupdate.cpp
            llama-graph-builder.cpp
            llama-graph-defrag.cpp
            llama-graph-deepseek2.cpp
            llama-graph-qwen2.cpp
            llama-graph-deepseek2ge.cpp
            llama-graph-utils.cpp
            llama-graph-ffn.cpp
            llama-graph-attn.cpp
            llama-hparams.cpp
            llama-impl.cpp
            llama-kv-cache.cpp
            llama-mmap.cpp
            llama-model-loader.cpp
            llama-model-load-utils.cpp
            llama-model.cpp
            llama-params.cpp
            llama-sampling.cpp
            llama-vocab.cpp
            unicode.h
            unicode.cpp
            unicode-data.cpp
            )

target_include_directories(llama PUBLIC . ../include ../common ../third_party)
target_compile_features   (llama PUBLIC cxx_std_17) # don't bump

# 添加MPI支持
if(LLAMA_MPI)
    target_include_directories(llama PUBLIC ${MPI_C_INCLUDE_PATH})
    target_link_libraries(llama PUBLIC ${MPI_C_LIBRARIES})
    target_compile_definitions(llama PUBLIC LLAMA_MPI_SUPPORT)
endif()

target_link_libraries(llama PUBLIC ggml)

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
