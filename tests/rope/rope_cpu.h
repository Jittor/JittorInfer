#include "ggml-cann.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "llama-impl.h"
#include <iostream>
#include <random>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <chrono>

void build_rope_graph_cpu(
    const std::vector<float> &input_host,
    const std::vector<int32_t> &positions_host,
    std::vector<float> &output_host,
    ggml_backend_t backend,
    int64_t batch_size,
    int64_t num_heads,
    int64_t head_dims,
    int64_t sequence_length,
    int n_dims,
    int mode
);