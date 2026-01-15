#include "llama-hparams.h"

#include "ggml.h"

uint32_t llama_hparams::n_head(uint32_t il) const {
    if (il < n_layer) {
        return n_head_arr[il];
    }

    GGML_ABORT("fatal error");
}

uint32_t llama_hparams::n_head_kv(uint32_t il) const {
    if (il < n_layer) {
        return n_head_kv_arr[il];
    }

    GGML_ABORT("fatal error");
}

uint32_t llama_hparams::n_ff(uint32_t il) const {
    if (il < n_layer) {
        return n_ff_arr[il];
    }

    GGML_ABORT("fatal error");
}

uint32_t llama_hparams::n_gqa(uint32_t il) const {
    const uint32_t n_head    = this->n_head(il);
    const uint32_t n_head_kv = this->n_head_kv(il);

    if (n_head_kv == 0) {
        return 0;
    }

    return n_head / n_head_kv;
}

uint32_t llama_hparams::n_embd_k_gqa(uint32_t il) const {
    if (enable_mla) {
        return n_lora_kv + n_rot;
    }
    const uint32_t n_head_kv  = this->n_head_kv(il);
    const uint32_t n_head_q   = this->n_head(il);
    const int      head_split = (enable_tensor_parallel && !enable_data_parallel) ? num_parallel : 1;
    
    // Calculate the actual number of KV heads this device needs
    uint32_t n_head_kv_local;
    if (n_head_kv >= (uint32_t)head_split) {
        // Normal case: KV heads can be evenly split
        n_head_kv_local = n_head_kv / head_split;
    } else if (n_head_kv > 0) {
        // GQA case: n_head_kv < head_split, each device needs a subset of KV heads
        // Calculate based on which Q heads this device handles
        const uint32_t n_rep = n_head_q / n_head_kv;  // GQA ratio
        const uint32_t n_q_per_device = n_head_q / head_split;
        const uint32_t kv_start = (tp_id * n_q_per_device) / n_rep;
        const uint32_t kv_end = ((tp_id + 1) * n_q_per_device + n_rep - 1) / n_rep;  // ceil
        n_head_kv_local = kv_end - kv_start;
    } else {
        n_head_kv_local = n_head_kv;
    }
    
    if (enable_cann_flash_attention) {
        const uint32_t max_embd = GGML_PAD(std::max(n_embd_head_k, n_embd_head_v), 64);
        return max_embd * n_head_kv_local;
    }

    return n_embd_head_k * n_head_kv_local;
}

uint32_t llama_hparams::n_embd_v_gqa(uint32_t il, bool enable_split) const {
    if (enable_mla) {
        return n_lora_kv;
    }
    const uint32_t n_head_kv  = this->n_head_kv(il);
    const uint32_t n_head_q   = this->n_head(il);
    const int      head_split = (enable_tensor_parallel && !enable_data_parallel && enable_split) ? num_parallel : 1;
    
    // Calculate the actual number of KV heads this device needs
    uint32_t n_head_kv_local;
    if (n_head_kv >= (uint32_t)head_split) {
        // Normal case: KV heads can be evenly split
        n_head_kv_local = n_head_kv / head_split;
    } else if (n_head_kv > 0) {
        // GQA case: n_head_kv < head_split, each device needs a subset of KV heads
        // Calculate based on which Q heads this device handles
        const uint32_t n_rep = n_head_q / n_head_kv;  // GQA ratio
        const uint32_t n_q_per_device = n_head_q / head_split;
        const uint32_t kv_start = (tp_id * n_q_per_device) / n_rep;
        const uint32_t kv_end = ((tp_id + 1) * n_q_per_device + n_rep - 1) / n_rep;  // ceil
        n_head_kv_local = kv_end - kv_start;
    } else {
        n_head_kv_local = n_head_kv;
    }
    
    if (enable_cann_flash_attention) {
        const uint32_t max_embd = GGML_PAD(std::max(n_embd_head_k, n_embd_head_v), 64);
        return max_embd * n_head_kv_local;
    }

    return n_embd_head_v * n_head_kv_local;
}

uint32_t llama_hparams::n_embd_k_s() const {
    if (wkv_head_size != 0) {
        // for RWKV models
        return token_shift_count * n_embd;
    }

    // TODO: maybe support other convolution strides than 1
    // NOTE: since the first column of the conv_state is shifted out each time, it's not actually needed
    return (ssm_d_conv > 0 ? ssm_d_conv - 1 : 0) * ssm_d_inner;
}

uint32_t llama_hparams::n_embd_v_s() const {
    if (wkv_head_size != 0) {
        // corresponds to RWKV's wkv_states size
        return n_embd * wkv_head_size;
    }

    // corresponds to Mamba's ssm_states size
    return ssm_d_state * ssm_d_inner;
}
