#ifndef LLAMA_KV_CACHE_H
#define LLAMA_KV_CACHE_H

#include <algorithm>
#include <bitset>
#include <vector>

#include "ggml-cpp.h"
#include "llama-cparams.h"
#include "llama.h"

struct llama_kv_cell {
    llama_pos pos   = -1;
    llama_pos delta = 0;
    int32_t   src   = -1;  // used by recurrent state models to copy states
    int32_t   tail  = -1;

    std::bitset<32> seq_id;

    inline bool has_seq_id(const llama_seq_id & id) const { return seq_id.test(id); }

    inline bool is_empty() const { return seq_id.none(); }

    bool is_same_seq(const llama_kv_cell & other) const { return seq_id == other.seq_id; }
};

// ring-buffer of cached KV data
struct llama_kv_cache {
    bool has_shift = false;
    bool do_defrag = false;
    bool recurrent = false;  // with recurrent state models, a cell can hold the state for more than one past token
    bool v_trans   = true;   // the value tensor is transposed
    bool can_shift = false;

    // Note: The value of head isn't only used to optimize searching
    // for a free KV slot. llama_decode_impl also uses it, so it
    // cannot be freely changed after a slot has been allocated.
    uint32_t head = 0;
    uint32_t size = 0;
    uint32_t used = 0;  // used cells (i.e. at least one seq_id)

    // Page Attention
    bool                      page_attention   = false;
    uint32_t                  page_size        = 128;
    uint32_t                  page_num         = 0;
    uint32_t                  page_num_per_seq = 0;
    uint32_t                  seq_num_max      = 0;
    std::vector<int32_t>      page_table;
    std::vector<int32_t>      seq_page_used;
    std::vector<int64_t>      seq_lengths;
    std::vector<int32_t>      page_used;
    std::vector<llama_seq_id> page_owner;  // only one onwer now.

    // computed before each graph build
    uint32_t n = 0;

    ggml_type type_k = GGML_TYPE_F16;
    ggml_type type_v = GGML_TYPE_F16;

    std::vector<llama_kv_cell> cells;

    std::vector<struct ggml_tensor *> k_l;  // per layer
    std::vector<struct ggml_tensor *> v_l;
    std::vector<struct ggml_tensor *> kq_mask_l;
    std::vector<struct ggml_tensor *> kq_masks;
    std::vector<struct ggml_tensor *> kq_masks_tmp;

    struct ggml_tensor * cos_cache;
    struct ggml_tensor * sin_cache;

    // Store original sin/cos cache data for restoration
    std::vector<uint8_t> sin_cache_data;
    std::vector<uint8_t> cos_cache_data;

    std::vector<ggml_context_ptr>        ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    size_t total_size() const {
        size_t size = 0;
        for (const auto & buf : bufs) {
            size += ggml_backend_buffer_get_size(buf.get());
        }

        return size;
    }

    // TODO: better data structures to reduce the cost of this operation
    llama_pos max_pos() const {
        llama_pos max_pos = -1;
        for (const auto & cell : cells) {
            max_pos = std::max(max_pos, cell.pos);
        }

        return max_pos;
    }
};

// a structure holds information about the slot found in llama_kv_cache_find_slot
struct llama_kv_cache_slot_info {
    std::pair<uint32_t, uint32_t> boundaries;     // slot boundaries [begin, end)
    std::vector<int32_t>          slot_ids;       // indices of the slots
    bool                          found = false;  // the slot was found

    explicit llama_kv_cache_slot_info(bool found_) : found{ found_ } {}

    llama_kv_cache_slot_info(uint32_t begin, uint32_t end) : boundaries{ begin, end }, found{ true } {}

    llama_kv_cache_slot_info(const std::vector<int32_t> & slot_ids_) : slot_ids{ slot_ids_ }, found{ true } {}

    operator bool() const { return found; }
};

// TODO: maybe not needed
uint32_t llama_kv_cache_get_padding(const struct llama_cparams & cparams);

void llama_kv_cache_clear(struct llama_kv_cache & cache);

// 等价于can_shift
bool llama_kv_cache_can_shift(const struct llama_kv_cache & kv);

// find an empty slot of size "n_tokens" in the cache
// updates the cache head
// returns a structure holding information about the slot found
// Note: On success, it's important that cache.head points
// to the first cell of the slot.
struct llama_kv_cache_slot_info llama_kv_cache_find_slot(struct llama_kv_cache &     cache,
                                                         const struct llama_ubatch & batch);

struct llama_kv_cache_slot_info llama_kv_cache_find_scatter_slot(struct llama_kv_cache &     cache,
                                                                 const struct llama_ubatch & batch, int require_slots);

struct llama_kv_cache_slot_info llama_kv_cache_find_page_slot(struct llama_kv_cache &     cache,
                                                              const struct llama_ubatch & batch);

// find how many cells are currently in use
uint32_t llama_kv_cache_cell_max(const struct llama_kv_cache & cache);

bool llama_kv_cache_init(struct llama_kv_cache & cache, const llama_model & model, const llama_cparams & cparams,
                         ggml_type type_k, ggml_type type_v, uint32_t kv_size, bool offload);

// Restore sin/cos cache data from stored vectors (for MLA)
void llama_kv_cache_restore_rope_cache(struct llama_kv_cache & cache);

bool llama_kv_cache_seq_rm(struct llama_kv_cache & cache, llama_seq_id seq_id, llama_pos p0, llama_pos p1);

void llama_kv_cache_seq_cp(struct llama_kv_cache & cache, llama_seq_id seq_id_src, llama_seq_id seq_id_dst,
                           llama_pos p0, llama_pos p1);

void llama_kv_cache_seq_add(struct llama_kv_cache & cache, llama_seq_id seq_id, llama_pos p0, llama_pos p1,
                            llama_pos delta);

void llama_kv_cache_defrag(struct llama_kv_cache & cache);

int32_t llama_get_kv_cache_used_cells(const struct llama_kv_cache & kv);

// saves the kv_cache state for future recovery.
// used to rollback llama_kv_cache_find_slot changes.
struct llama_kv_slot_restorer {
    struct llama_kv_cache_state_local {
        uint32_t head = 0;
        uint32_t n    = 0;
    } old_state;

    // for non-recurrent models only
    // list of slots to restore
    std::vector<std::pair<uint32_t, uint32_t>> slot_boundaries;

    bool do_restore = false;

    explicit llama_kv_slot_restorer(const struct llama_kv_cache & cache) {
        old_state.head = cache.head;
        old_state.n    = cache.n;
    }

    // saves a slot information for future restoration
    void save(const struct llama_kv_cache_slot_info & slot) {
        if (slot) {
            do_restore = true;
            if (slot.boundaries.first != slot.boundaries.second) {
                slot_boundaries.push_back(slot.boundaries);
            }
        }
    }
};

//
// kv cache view
//

struct llama_kv_cache_view llama_kv_cache_view_init(const struct llama_kv_cache & kv, int32_t n_seq_max);

void llama_kv_cache_view_update(struct llama_kv_cache_view * view, const struct llama_kv_cache & kv);

#endif  // LLAMA_KV_CACHE_H
