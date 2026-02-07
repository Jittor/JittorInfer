#include "llama-kv-cache.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>

#include "ggml.h"
#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-impl.h"
#include "llama-model.h"

static const llama_kv_cache_slot_info llama_kv_cache_slot_info_failed{ false };

uint32_t llama_kv_cache_get_padding(const struct llama_cparams & cparams) {
    // the FA kernels require padding to avoid extra runtime boundary checks
    return cparams.flash_attn ? 256u : 32u;
}

bool llama_kv_cache_can_shift(const struct llama_kv_cache & kv) {
    return kv.can_shift;
}

bool llama_kv_cache_init(struct llama_kv_cache & cache, const llama_model & model, const llama_cparams & cparams,
                         ggml_type type_k, ggml_type type_v, uint32_t kv_size, bool offload) {
    const struct llama_hparams & hparams = model.hparams;

    const int32_t n_layer = hparams.n_layer;

    cache.has_shift = false;

    cache.recurrent = llama_model_is_recurrent(&model);
    cache.v_trans   = !cache.recurrent && !cparams.flash_attn;
    cache.can_shift = !cache.recurrent && model.arch != LLM_ARCH_DEEPSEEK2;  // not supported due to MLA

    LLAMA_LOG_INFO("%s: kv_size = %d, offload = %d, type_k = '%s', type_v = '%s', n_layer = %d, can_shift = %d\n",
                   __func__, kv_size, offload, ggml_type_name(type_k), ggml_type_name(type_v), n_layer,
                   cache.can_shift);

    cache.head = 0;
    cache.size = kv_size;
    cache.used = 0;
    cache.page_attention = cparams.page_attention;
    if (cache.page_attention) {
        cache.page_size = 128;
        cache.page_num = kv_size / cache.page_size;
        cache.seq_num_max = cparams.n_seq_max;
        cache.page_num_per_seq = cache.page_num / cache.seq_num_max;
        if (kv_size % cache.page_size != 0) {
            LLAMA_LOG_ERROR("%s: kv_size must be a multiple of page_size\n", __func__);
            return false;
        }
        if (kv_size % (cache.seq_num_max * cache.page_size) != 0) {
            LLAMA_LOG_ERROR("%s: kv_size must be a multiple of n_seq_max * page_size\n", __func__);
            return false;
        }
        cache.page_table.resize(cache.page_num_per_seq * cache.seq_num_max);
        cache.page_used.resize(cache.page_num);
        cache.page_owner.resize(cache.page_num);
        cache.seq_lengths.resize(cache.seq_num_max);
        cache.seq_page_used.resize(cache.seq_num_max);
        for (int i = 0; i < cache.seq_num_max; i++) {
            for (int j = 0; j < cache.page_num_per_seq; j++) {
                cache.page_table[i * cache.page_num_per_seq + j] = 0;
            }
            cache.seq_lengths[i] = 0;
            cache.seq_page_used[i] = 0;
        }
        for (int i = 0; i < cache.page_num; i++) {
            cache.page_used[i] = 0;
            cache.page_owner[i] = -1;
        }
        if (cache.page_num < 2) {
            LLAMA_LOG_ERROR("%s: page_num must be greater than 2\n", __func__);
            return false;
        }
        // reserve page zero
        cache.page_owner[0] = -2;
    }

    cache.type_k = type_k;
    cache.type_v = type_v;

    cache.cells.clear();
    cache.cells.resize(kv_size);

    // create a context for each buffer type
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            struct ggml_init_params params = {
                /*.mem_size   =*/size_t(4u * n_layer * ggml_tensor_overhead()),
                /*.mem_buffer =*/NULL,
                /*.no_alloc   =*/true,
            };
            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }
            ctx_map[buft] = ctx;
            cache.ctxs.emplace_back(ctx);
            return ctx;
        }
        return it->second;
    };

    cache.k_l.reserve(n_layer);
    cache.v_l.reserve(n_layer);

    for (int i = 0; i < n_layer; i++) {
        const uint32_t n_embd_k_cache = hparams.n_embd_k_cache(i);
        const uint32_t n_embd_v_cache = hparams.n_embd_v_cache(i);

        LLAMA_LOG_DEBUG("%s: layer %d: n_embd_k_cache = %d, n_embd_v_cache = %d\n", __func__, i, n_embd_k_cache,
                        n_embd_v_cache);

        ggml_backend_buffer_type_t buft;
        if (offload) {
            auto * dev = model.dev_layer(i);
            buft       = ggml_backend_dev_buffer_type(dev);
        } else {
            buft = ggml_backend_cpu_buffer_type();
        }
        ggml_context * ctx = ctx_for_buft(buft);

        if (!ctx) {
            LLAMA_LOG_ERROR("%s: failed to create ggml context for kv cache\n", __func__);
            return false;
        }

        ggml_tensor * k = ggml_new_tensor_1d(ctx, type_k, n_embd_k_cache * kv_size);
        ggml_tensor * v = ggml_new_tensor_1d(ctx, type_v, n_embd_v_cache * kv_size);
        ggml_format_name(k, "cache_k_l%d", i);
        ggml_format_name(v, "cache_v_l%d", i);
        cache.k_l.push_back(k);
        cache.v_l.push_back(v);
    }
    // buffer on device for cann flash attention.
    if (cparams.flash_attn && !hparams.enable_mla) {
        cache.kq_mask_l.reserve(n_layer);
        std::map<ggml_backend_buffer_type_t, ggml_tensor *> kq_mask_per_buffer;
        for (int i = 0; i < n_layer; i++) {
            LLAMA_LOG_DEBUG("%s: layer %d: cache_size = %d\n", __func__, i, kv_size);

            ggml_backend_buffer_type_t buft;
            if (offload) {
                auto * dev = model.dev_layer(i);
                buft       = ggml_backend_dev_buffer_type(dev);
            } else {
                buft = ggml_backend_cpu_buffer_type();
            }
            ggml_context * ctx = ctx_for_buft(buft);

            if (!ctx) {
                LLAMA_LOG_ERROR("%s: failed to create ggml context for kv cache\n", __func__);
                return false;
            }

            if (cparams.enable_ge) {
                ggml_tensor * kq_mask;
                if (kq_mask_per_buffer.find(buft) == kq_mask_per_buffer.end()) {
                    const int64_t n_tokens    = GGML_PAD(cparams.n_ubatch, GGML_KQ_MASK_PAD);
                    kq_mask                   = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, kv_size * n_tokens);
                    ggml_tensor * kq_mask_tmp = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, kv_size * n_tokens);
                    ggml_set_name(kq_mask, "cache_kqmask");
                    ggml_set_name(kq_mask_tmp, "cache_kqmask_tmp");
                    kq_mask_per_buffer[buft] = kq_mask;
                    cache.kq_masks.push_back(kq_mask);
                    cache.kq_masks_tmp.push_back(kq_mask_tmp);
                } else {
                    kq_mask = kq_mask_per_buffer[buft];
                }
                cache.kq_mask_l.push_back(kq_mask);

            } else {
                ggml_tensor * kq_mask = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, kv_size * 64);
                ggml_format_name(kq_mask, "cache_kqmask_l%d", i);
                cache.kq_mask_l.push_back(kq_mask);
            }
        }
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto it : ctx_map) {
        auto * buft = it.first;
        auto * ctx  = it.second;

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buf) {
            LLAMA_LOG_ERROR("%s: failed to allocate buffer for kv cache\n", __func__);
            return false;
        }
        ggml_backend_buffer_clear(buf, 0);
        LLAMA_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf),
                       ggml_backend_buffer_get_size(buf) / 1024.0 / 1024.0);
        cache.bufs.emplace_back(buf);
    }

    return true;
}

void llama_kv_cache_clear(struct llama_kv_cache & cache) {
    for (int32_t i = 0; i < (int32_t) cache.size; ++i) {
        cache.cells[i].pos = -1;
        cache.cells[i].seq_id.reset();
        cache.cells[i].src  = -1;
        cache.cells[i].tail = -1;
    }
    cache.head = 0;
    cache.used = 0;
    if (cache.page_attention) {
        for (int i = 0; i < cache.page_num_per_seq * cache.seq_num_max; i++) {
            cache.page_table[i] = 0;
        }
        for (int i = 0; i < cache.seq_num_max; i++) {
            cache.seq_lengths[i] = 0;
            cache.seq_page_used[i] = 0;
        }
        for (int i = 0; i < cache.page_num; i++) {
            cache.page_used[i] = 0;
            cache.page_owner[i] = -1;
        }
        cache.page_owner[0] = -2;
    }

    for (auto & buf : cache.bufs) {
        ggml_backend_buffer_clear(buf.get(), 0);
    }
}

static void update_page_status(struct llama_kv_cache & cache, llama_seq_id seq_id) {
    int last_page = -1;
    int relative_blocks = 0;
    int cached_length = 0;
    GGML_ASSERT(seq_id < cache.seq_num_max);
    for (int page_id = 0; page_id < cache.page_num; page_id++) {
        int rep_id = page_id * cache.page_size;
        llama_kv_cell rep_cell = cache.cells[rep_id];
        if (!rep_cell.has_seq_id(seq_id)) {
            for (int off = 0; off < cache.page_size; off++) {
                if (cache.cells[rep_id + off].has_seq_id(seq_id)) {
                    GGML_ABORT("not implemented: head deleted.\n");
                }
            }
            // page is cleared.
            if (cache.page_owner[page_id] == seq_id) {
                cache.page_owner[page_id] = -1;
                cache.page_used[page_id] = 0;
            }
        } else {
            int used = cache.page_size;
            for (int off = 0; off < cache.page_size; off++) {
                if (cache.cells[rep_id + off].has_seq_id(seq_id)) {
                    if (used < cache.page_size) {
                        GGML_ABORT("not implemented: unexpected gap.\n");
                    }
                } else {
                    if (used >= cache.page_size) {
                        used = off;
                    }
                }
            }
            cache.page_owner[page_id] = seq_id;
            cache.page_used[page_id] = used;
            if (used < cache.page_size) {
                if (last_page >= 0) {
                    GGML_ABORT("not implemented: too much incomplete blocks.\n");
                }
                last_page = page_id;
            } else {
                if (relative_blocks >= cache.page_num_per_seq) {
                    GGML_ABORT("not implemented: too much blocks.\n");
                }
                cache.page_table[seq_id * cache.page_num_per_seq + relative_blocks] = page_id;
                relative_blocks++;
            }
            cached_length += used;
        }
    }
    if (last_page >= 0) {
        if (relative_blocks >= cache.page_num_per_seq) {
            GGML_ABORT("not implemented: too much blocks.\n");
        }
        cache.page_table[seq_id * cache.page_num_per_seq + relative_blocks] = last_page;
        cache.seq_page_used[seq_id] = relative_blocks + 1;
    } else {
        cache.seq_page_used[seq_id] = relative_blocks;
    }
    cache.seq_lengths[seq_id] = cached_length;
}
static int assign_cell(struct llama_kv_cache & cache, llama_seq_id seq_id) {
    GGML_ASSERT(seq_id < cache.seq_num_max);
    int32_t last_page = -1;
    if (cache.seq_page_used[seq_id] > 0) {
        last_page = cache.page_table[seq_id * cache.page_num_per_seq + cache.seq_page_used[seq_id] - 1];
    }
    if (last_page != -1 && cache.page_used[last_page] < cache.page_size) {
        cache.page_used[last_page]++;
        cache.seq_lengths[seq_id]++;
        return last_page * cache.page_size + cache.page_used[last_page] - 1;
    }
    for (int i = 0; i < cache.page_num; i++) {
        if (cache.page_owner[i] == -1) {
            cache.page_owner[i] = seq_id;
            cache.page_used[i] = 1;
            cache.seq_lengths[seq_id]++;
            cache.seq_page_used[seq_id]++;
            cache.page_table[seq_id * cache.page_num_per_seq + cache.seq_page_used[seq_id] - 1] = i;
            return i * cache.page_size;
        }
    }
    GGML_ABORT("not implemented: too much pages.");
}

bool llama_kv_cache_seq_rm(struct llama_kv_cache & cache, llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    uint32_t new_head = cache.size;

    p0 = std::max(p0, 0);
    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // models like Mamba or RWKV can't have a state partially erased
    if (cache.recurrent) {
        if (cache.page_attention) {
            GGML_ABORT("not implemented");
        }
        if (seq_id >= (int64_t) cache.size) {
            // could be fatal
            return false;
        }
        if (0 <= seq_id) {
            int32_t & tail_id = cache.cells[seq_id].tail;
            if (tail_id >= 0) {
                const llama_kv_cell & cell = cache.cells[tail_id];
                // partial intersection is invalid
                if ((0 < p0 && p0 <= cell.pos) || (0 < p1 && p1 <= cell.pos)) {
                    return false;
                }
                // invalidate tails which will be cleared
                if (p0 <= cell.pos && cell.pos < p1) {
                    tail_id = -1;
                }
            }
        } else {
            // seq_id is negative, then the range should include everything or nothing
            if (p0 != p1 && (p0 != 0 || p1 != std::numeric_limits<llama_pos>::max())) {
                return false;
            }
        }
    }

    for (uint32_t i = 0; i < cache.size; ++i) {
        if (cache.cells[i].pos >= p0 && cache.cells[i].pos < p1) {
            if (seq_id < 0) {
                cache.cells[i].seq_id.reset();
            } else if (cache.cells[i].has_seq_id(seq_id)) {
                cache.cells[i].seq_id.reset(seq_id);
            } else {
                continue;
            }
            if (cache.cells[i].is_empty()) {
                // keep count of the number of used cells
                if (cache.cells[i].pos >= 0) {
                    cache.used--;
                }

                cache.cells[i].pos = -1;
                cache.cells[i].src = -1;
                if (new_head == cache.size) {
                    new_head = i;
                }
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != cache.size && new_head < cache.head) {
        cache.head = new_head;
    }

    if (cache.page_attention) {
        update_page_status(cache, seq_id);
    }

    return true;
}

void llama_kv_cache_seq_cp(struct llama_kv_cache & cache, llama_seq_id seq_id_src, llama_seq_id seq_id_dst,
                           llama_pos p0, llama_pos p1) {
    p0 = std::max(p0, 0);
    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if (cache.recurrent) {
        if ((uint32_t) seq_id_dst < cache.size && (uint32_t) seq_id_src < cache.size) {
            llama_kv_cell & tail_src = cache.cells[seq_id_src];
            llama_kv_cell & tail_dst = cache.cells[seq_id_dst];
            if (tail_dst.tail >= 0) {
                // clear destination seq_id if it wasn't empty
                llama_kv_cell & cell_dst = cache.cells[tail_dst.tail];

                cell_dst.seq_id.reset(seq_id_dst);
                tail_dst.tail = -1;
                if (cell_dst.seq_id.none()) {
                    cell_dst.pos   = -1;
                    cell_dst.delta = -1;
                    cell_dst.src   = -1;
                    cache.used -= 1;
                }
            }
            if (tail_src.tail >= 0) {
                llama_kv_cell & cell_src = cache.cells[tail_src.tail];

                cell_src.seq_id.set(seq_id_dst);
                tail_dst.tail = tail_src.tail;
            }
        }

        return;
    }
    // otherwise, this is the KV cache of a Transformer-like model

    cache.head = 0;

    for (uint32_t i = 0; i < cache.size; ++i) {
        if (cache.cells[i].has_seq_id(seq_id_src) && cache.cells[i].pos >= p0 && cache.cells[i].pos < p1) {
            cache.cells[i].seq_id.set(seq_id_dst);
        }
    }
}

void llama_kv_cache_seq_add(struct llama_kv_cache & cache, llama_seq_id seq_id, llama_pos p0, llama_pos p1,
                            llama_pos delta) {
    uint32_t new_head = cache.size;

    if (p0 < 0) {
        p0 = 0;
    }
    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }
    // If there is no range then return early to avoid looping over the cache.
    if (p0 == p1) {
        return;
    }

    if (cache.recurrent) {
        // for Mamba-like or RWKV models, only the pos needs to be shifted
        if (0 <= seq_id && seq_id < (int64_t) cache.size) {
            const int32_t tail_id = cache.cells[seq_id].tail;
            if (tail_id >= 0) {
                llama_kv_cell & cell = cache.cells[tail_id];
                if (cell.has_seq_id(seq_id) && p0 <= cell.pos && cell.pos < p1) {
                    cell.pos += delta;
                }
            }
        }
        return;
    }

    for (uint32_t i = 0; i < cache.size; ++i) {
        if (cache.cells[i].has_seq_id(seq_id) && cache.cells[i].pos >= p0 && cache.cells[i].pos < p1) {
            cache.has_shift = true;
            cache.cells[i].pos += delta;
            cache.cells[i].delta += delta;

            if (cache.cells[i].pos < 0) {
                if (!cache.cells[i].is_empty()) {
                    cache.used--;
                }
                cache.cells[i].pos = -1;
                cache.cells[i].seq_id.reset();
                if (new_head == cache.size) {
                    new_head = i;
                }
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    // Otherwise we just start the next search from the beginning.
    cache.head = new_head != cache.size ? new_head : 0;

    if (cache.page_attention) {
        update_page_status(cache, seq_id);
    }
}

struct llama_kv_cache_slot_info llama_kv_cache_find_slot(struct llama_kv_cache &     cache,
                                                         const struct llama_ubatch & ubatch) {
    const uint32_t n_tokens     = ubatch.n_tokens;
    const uint32_t n_seqs       = ubatch.n_seqs;
    const uint32_t n_seq_tokens = ubatch.n_seq_tokens;

    // simplified: only support non-recurrent models.
    GGML_ASSERT(!cache.recurrent);

    // otherwise, one cell per token.
    if (n_tokens > cache.size) {
        LLAMA_LOG_ERROR("%s: n_tokens=%d > cache.size=%d\n", __func__, n_tokens, cache.size);
        return llama_kv_cache_slot_info_failed;
    }

    uint32_t n_tested = 0;

    while (true) {
        if (cache.head + n_tokens > cache.size) {
            n_tested += cache.size - cache.head;
            cache.head = 0;
            continue;
        }

        bool found = true;
        for (uint32_t i = 0; i < n_tokens; i++) {
            if (cache.cells[cache.head + i].pos >= 0) {
                found = false;
                cache.head += i + 1;
                n_tested += i + 1;
                break;
            }
        }

        if (found) {
            break;
        }

        if (n_tested >= cache.size) {
            return llama_kv_cache_slot_info_failed;
        }
    }

    for (uint32_t s = 0; s < n_seqs; s++) {
        for (uint32_t i = 0; i < n_seq_tokens; ++i) {
            uint32_t k                      = s * n_seq_tokens + i;
            cache.cells[cache.head + k].pos = ubatch.pos[k];

            for (int32_t j = 0; j < ubatch.n_seq_id[s]; j++) {
                cache.cells[cache.head + k].seq_id.set(ubatch.seq_id[s][j]);
            }
        }
    }

    cache.used += n_tokens;

    return llama_kv_cache_slot_info(cache.head, cache.head + n_tokens);
}

struct llama_kv_cache_slot_info llama_kv_cache_find_page_slot(struct llama_kv_cache &     cache,
                                                         const struct llama_ubatch & ubatch) {
    const uint32_t n_tokens     = ubatch.n_tokens;
    const uint32_t n_seqs       = ubatch.n_seqs;
    const uint32_t n_seq_tokens = ubatch.n_seq_tokens;

    // simplified: only support non-recurrent models.
    GGML_ASSERT(!cache.recurrent);

    // otherwise, one cell per token.
    if (n_tokens > cache.size) {
        LLAMA_LOG_ERROR("%s: n_tokens=%d > cache.size=%d\n", __func__, n_tokens, cache.size);
        return llama_kv_cache_slot_info_failed;
    }
    std::vector<int> slot_ids;

    // for page attention, multiple seq_id in a cell is not allowed.
    for (uint32_t s = 0; s < n_seqs; s++) {
        for (uint32_t i = 0; i < n_seq_tokens; ++i) {
            GGML_ASSERT(ubatch.n_seq_id[s] == 1);
            llama_seq_id seq_id = ubatch.seq_id[s][0];

            int32_t cell_id = assign_cell(cache, seq_id);
            slot_ids.push_back(cell_id);

            cache.cells[cell_id].pos = ubatch.pos[s * n_seq_tokens + i];
            cache.cells[cell_id].seq_id.set(seq_id);
        }
    }
    cache.used += n_tokens;

    return llama_kv_cache_slot_info(slot_ids);
}

struct llama_kv_cache_slot_info llama_kv_cache_find_scatter_slot(struct llama_kv_cache &     cache,
                                                                 const struct llama_ubatch & ubatch,
                                                                 int                         require_slots) {
    const uint32_t n_tokens     = ubatch.n_tokens;
    const uint32_t n_seqs       = ubatch.n_seqs;
    const uint32_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(!cache.recurrent);

    if (require_slots + cache.used > cache.size) {
        LLAMA_LOG_ERROR("%s: require_slots=%d + cache.used=%d > cache.size=%d\n", __func__, require_slots, cache.used,
                        cache.size);
        return llama_kv_cache_slot_info_failed;
    }

    uint32_t         n_tested = 0;
    std::vector<int> slot_ids;

    while (true) {
        if (cache.head >= cache.size) {
            cache.head = 0;
        }

        if (cache.cells[cache.head].pos < 0) {
            require_slots -= 1;
            slot_ids.push_back(cache.head);
        }
        cache.head += 1;
        n_tested += 1;

        if (require_slots == 0) {
            break;
        }

        if (n_tested >= cache.size) {
            return llama_kv_cache_slot_info_failed;
        }
    }

    for (uint32_t s = 0; s < n_seqs; s++) {
        for (uint32_t i = 0; i < n_seq_tokens; ++i) {
            uint32_t k                   = s * n_seq_tokens + i;
            cache.cells[slot_ids[k]].pos = ubatch.pos[k];

            for (int32_t j = 0; j < ubatch.n_seq_id[s]; j++) {
                cache.cells[slot_ids[k]].seq_id.set(ubatch.seq_id[s][j]);
            }
        }
    }

    cache.used += n_tokens;

    return llama_kv_cache_slot_info(slot_ids);
}

uint32_t llama_kv_cache_cell_max(const struct llama_kv_cache & cache) {
    for (uint32_t i = cache.size; i > 0; --i) {
        const llama_kv_cell & cell = cache.cells[i - 1];
        if (cell.pos >= 0 && !cell.is_empty()) {
            return i;
        }
    }

    return 0;
}

void llama_kv_cache_defrag(struct llama_kv_cache & cache) {
    if (!cache.recurrent) {
        cache.do_defrag = true;
    }
}

int32_t llama_get_kv_cache_used_cells(const struct llama_kv_cache & kv) {
    return kv.used;
}

//
// kv cache view
//

struct llama_kv_cache_view llama_kv_cache_view_init(const struct llama_kv_cache & kv, int32_t n_seq_max) {
    struct llama_kv_cache_view result = {
        /*.n_cells            = */ 0,
        /*.n_seq_max          = */ n_seq_max,
        /*.token_count        = */ 0,
        /*.used_cells         = */ llama_get_kv_cache_used_cells(kv),
        /*.max_contiguous     = */ 0,
        /*.max_contiguous_idx = */ -1,
        /*.cells              = */ nullptr,
        /*.cells_sequences    = */ nullptr,
    };

    return result;
}

void llama_kv_cache_view_update(struct llama_kv_cache_view * view, const struct llama_kv_cache & kv) {
    if (uint32_t(view->n_cells) < kv.size || view->cells == nullptr) {
        view->n_cells = int32_t(kv.size);
        void * p      = realloc(view->cells, sizeof(struct llama_kv_cache_view_cell) * view->n_cells);
        GGML_ASSERT(p != nullptr && "Failed to alloc kv_cache_view cells");
        view->cells = (struct llama_kv_cache_view_cell *) p;
        p           = realloc(view->cells_sequences, sizeof(llama_seq_id) * view->n_seq_max * view->n_cells);
        GGML_ASSERT(p != nullptr && "Failed to alloc kv_cache_view cells sequences");
        view->cells_sequences = (llama_seq_id *) p;
    }

    const std::vector<llama_kv_cell> & kv_cells        = kv.cells;
    llama_kv_cache_view_cell *         c_curr          = view->cells;
    llama_seq_id *                     cs_curr         = view->cells_sequences;
    int32_t                            used_cells      = 0;
    int32_t                            token_count     = 0;
    int32_t                            curr_contig_idx = -1;
    uint32_t                           max_contig      = 0;
    int32_t                            max_contig_idx  = -1;

    for (int32_t i = 0; i < int32_t(kv.size); i++, c_curr++, cs_curr += view->n_seq_max) {
        const size_t curr_size = kv_cells[i].seq_id.size();
        token_count += curr_size;
        c_curr->pos = kv_cells[i].pos + kv_cells[i].delta;

        if (curr_size > 0) {
            if (curr_contig_idx >= 0 && uint32_t(i - curr_contig_idx) > max_contig) {
                max_contig     = i - curr_contig_idx;
                max_contig_idx = curr_contig_idx;
            }
            curr_contig_idx = -1;
        } else if (curr_contig_idx < 0) {
            curr_contig_idx = i;
        }

        int seq_idx = 0;
        for (int32_t bit_idx = 0; seq_idx < view->n_seq_max && bit_idx < (int32_t) kv_cells[i].seq_id.size();
             bit_idx++) {
            if (kv_cells[i].seq_id.test(bit_idx)) {
                llama_seq_id seq = bit_idx;
                cs_curr[seq_idx] = seq;
                seq_idx++;
            }
        }
        if (seq_idx != 0) {
            used_cells++;
        }
        for (; seq_idx < view->n_seq_max; seq_idx++) {
            cs_curr[seq_idx] = -1;
        }
    }
    if (curr_contig_idx >= 0 && kv_cells.size() - curr_contig_idx > max_contig) {
        max_contig_idx = curr_contig_idx;
        max_contig     = kv_cells.size() - curr_contig_idx;
    }
    view->max_contiguous     = max_contig;
    view->max_contiguous_idx = max_contig_idx;
    view->token_count        = token_count;
    view->used_cells         = used_cells;
    if (uint32_t(used_cells) != kv.used) {
        LLAMA_LOG_ERROR("%s: used cells mismatch. kv_cache says %d but we calculated %d\n", __func__, kv.used,
                        used_cells);
    }
}
