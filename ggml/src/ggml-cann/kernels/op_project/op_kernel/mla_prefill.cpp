/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * AscendOpCommonLib is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

 #include "mlapo.utils/common.h"
 #include "mlapo.utils/common_func.h"
 #include "mlapo.utils/simd.h"
 #include "mlapo.utils/iterator.h"
 #include "mlapo.utils/mma.h"
 #include "mlapo.utils/utils.h"
 #include "mlapo.utils/fa_common.cpp"
 #include "kernel_operator.h"
 #include "lib/matmul_intf.h"
 
 #ifdef __CCE_KT_TEST__
 #define __aicore__
 #else
 #define __aicore__ [aicore]
 #endif
 
 // FFTS Flag
 constexpr int32_t QK_READY = 1;
 constexpr int32_t SOFTMAX_READY = 2;
 constexpr int32_t UPDATE_READY = 3;
 constexpr int32_t BIT_SHIFT = 8;
 constexpr int32_t SOFTMAX_MAX_LENGTH = 256;
 
 constexpr int32_t L0AB_HALF_BUF_SIZE = 16384;  // 128 * 128
 constexpr int32_t CUBE_MATRIX_SIZE = 256;         // 16 * 16
 constexpr int32_t L0AB_UINT8_BLOCK_SIZE = 32768;  // 128 * 128 * 2B
 constexpr int32_t KV_DB_SIZE = 65536;  // 128 * 128 * 2B
 constexpr int32_t TMP_SIZE = 32768 * 16;               // 128 * 256
 constexpr int32_t UB_UINT8_BLOCK_SIZE = 16384;  // 64 * 128 * 2B
 constexpr int32_t UB_HALF_BUF_SIZE = 8192;
 constexpr int32_t UB_UINT8_LINE_SIZE = 512;     // 128 * 4B
 constexpr int32_t UB_FLOAT_LINE_SIZE = 128;     // 128
 constexpr int32_t UB_HALF_LINE_SIZE = 128;       // UB_FLOAT_sLINE_SIZE * 2
 constexpr int32_t BLOCK_QK = 128;
 constexpr int32_t S_BLOCK_STACK = 4;
 

 #ifdef __DAV_C220_CUBE__
 
 template<typename IN_DATA_TYPE, typename QKV_DT = IN_DATA_TYPE, typename O_DT = IN_DATA_TYPE, int32_t MASK_TYPE = 0>
 class MLAPrefillOp_C {
 public:
    __aicore__ __attribute__((always_inline)) inline MLAPrefillOp_C() {
    }

    __aicore__ __attribute__((always_inline)) inline void Init(
        __gm__ uint8_t *__restrict__ query,
        __gm__ uint8_t *__restrict__ qRope,
        __gm__ uint8_t *__restrict__ key,
        __gm__ uint8_t *__restrict__ kRope,
        __gm__ uint8_t *__restrict__ value,
        __gm__ uint8_t *__restrict__ workspace,
        __gm__ const MLAPrefillTilingData* __restrict__ tiling)
     {

        SetPadding<uint64_t>(0);
        SetAtomicnone();
        SetNdpara(1, 0, 0);
        SetMasknorm();

        query_gm = reinterpret_cast<__gm__ QKV_DT *>(query);
        qRope_gm = reinterpret_cast<__gm__ QKV_DT *>(qRope);
        key_gm = reinterpret_cast<__gm__ QKV_DT *>(key);
        kRope_gm = reinterpret_cast<__gm__ QKV_DT *>(kRope);
        value_gm = reinterpret_cast<__gm__ QKV_DT *>(value);

        q_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ QKV_DT *>(this->query_gm));
        q_rope_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ QKV_DT *>(this->qRope_gm));

        k_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ QKV_DT *>(this->key_gm));
        k_rope_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ QKV_DT *>(this->kRope_gm));
        v_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ QKV_DT *>(this->value_gm));
        
        tiling_gm = const_cast<__gm__ MLAPrefillTilingData*>(tiling);

        __gm__ const MLAPrefillHeadParams* __restrict headparams = &(tiling_gm->MLAPrefillHead);

        this->batch_size = (uint32_t)(headparams->batch);
        this->max_seqlen = (uint32_t)(headparams->maxSeqLen);
        this->q_heads = (uint32_t)(headparams->numHeads);
        this->embd = (uint32_t)(headparams->embeddingSize);
        this->embdv = (uint32_t)(headparams->embeddingSizeV);
        this->kv_heads = (uint32_t)(headparams->kvRealHeads);
        this->total_q_blk_num = (uint32_t)(headparams->totalQBlkNum);
        this->tiling_head_size = (uint32_t)(headparams->headSize);
        this->tiling_para_size = (uint32_t)(headparams->paraSize);
        this->max_kv_seqlen = (uint32_t)(headparams->maxKvSeqLen);
        this->window_size = (uint32_t)(headparams->windowSize);
        this->block_dim = (uint32_t)(headparams->blockDim);

        if constexpr(MASK_TYPE == 1) {
            this->is_triu_mask = 1;
        }

        this->data_shape_type = 0;
        this->group_num = q_heads / kv_heads;
        this->stride_qo = q_heads * 128;
        this->stride_o = q_heads * embdv;
        this->stride_k = kv_heads * 128;
        this->stride_v = kv_heads * embdv;

        if (data_shape_type == 1) {
            this->stride_qo = embdv;
            this->stride_o = embdv;
            this->stride_k = embdv;
            this->stride_v = embdv;
        }

        this->batch_stride_kv = batch_size * max_kv_seqlen * kv_heads * embd * sizeof(QKV_DT);
        this->__k = embd;
        this->round_k = (__k + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

        uint64_t dataLenFloat = sizeof(float);
        uint64_t sSize = static_cast<uint64_t>(block_dim) * static_cast<uint64_t>(32768) * 16 * dataLenFloat;

        __gm__ float * o_tmp_gm = reinterpret_cast<__gm__ float *>(workspace);
        __gm__ QKV_DT * s_out_gm = reinterpret_cast<__gm__ QKV_DT *>(workspace + sSize);
        __gm__ QKV_DT * p_out_gm = reinterpret_cast<__gm__ QKV_DT *>(workspace + sSize * 2);
        o_tmp_gm_tensor.SetGlobalBuffer(o_tmp_gm, sSize);
        s_gm_tensor.SetGlobalBuffer(s_out_gm, sSize);
        p_gm_tensor.SetGlobalBuffer(p_out_gm, sSize);
        

        SET_FLAG(MTE1, MTE2, EVENT_ID0);
        SET_FLAG(MTE1, MTE2, EVENT_ID1);
        SET_FLAG(MTE1, MTE2, EVENT_ID2);
        SET_FLAG(MTE1, MTE2, EVENT_ID3);
        SET_FLAG(MTE1, MTE2, EVENT_ID4);
        SET_FLAG(MTE1, MTE2, EVENT_ID5);
        SET_FLAG(MTE1, MTE2, EVENT_ID6);
        SET_FLAG(MTE1, MTE2, EVENT_ID7);
        SET_FLAG(M, MTE1, EVENT_ID0);
        SET_FLAG(M, MTE1, EVENT_ID1);
        SET_FLAG(M, MTE1, EVENT_ID2);
        SET_FLAG(M, MTE1, EVENT_ID3);
        SET_FLAG(FIX, M, EVENT_ID0);
        SET_FLAG(FIX, M, EVENT_ID1);
    }
 
     __aicore__ __attribute__((always_inline)) inline ~MLAPrefillOp_C()
     {
         WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
         WAIT_FLAG(MTE1, MTE2, EVENT_ID1);
         WAIT_FLAG(MTE1, MTE2, EVENT_ID2);
         WAIT_FLAG(MTE1, MTE2, EVENT_ID3);
         WAIT_FLAG(MTE1, MTE2, EVENT_ID4);
         WAIT_FLAG(MTE1, MTE2, EVENT_ID5);
         WAIT_FLAG(MTE1, MTE2, EVENT_ID6);
         WAIT_FLAG(MTE1, MTE2, EVENT_ID7);
         WAIT_FLAG(M, MTE1, EVENT_ID0);
         WAIT_FLAG(M, MTE1, EVENT_ID1);
         WAIT_FLAG(M, MTE1, EVENT_ID2);
         WAIT_FLAG(M, MTE1, EVENT_ID3);
         WAIT_FLAG(FIX, M, EVENT_ID0);
         WAIT_FLAG(FIX, M, EVENT_ID1);
         PIPE_BARRIER(ALL);
     }
 
     __aicore__ __attribute__((always_inline)) inline void LoadDataToCa(
         AscendC::LocalTensor<QKV_DT> dst_tensor, AscendC::LocalTensor<QKV_DT> src_tensor,
         uint32_t round_k, uint32_t qk_round_m, uint32_t qk_m)
     {
         uint32_t round_row = RoundUp<uint32_t>(round_k, 32 / sizeof(QKV_DT));
         if (qk_m == 1) {
             l1_to_l0_a<ArchType::ASCEND_V220, QKV_DT, false, DataFormat::VECTOR, DataFormat::VECTOR>(
                 dst_tensor,
                 src_tensor,
                 0,
                 NumMatrixsRoundUp<QKV_DT>(round_row), // repeat
                 0,
                 1,                                                   // srcStride
                 0,
                 0                                                   // dstStride
             );
         } else {
             l1_to_l0_a<ArchType::ASCEND_V220, QKV_DT, false, DataFormat::NZ, DataFormat::ZZ>(
                 dst_tensor, src_tensor, qk_round_m, round_row, 0, 0, 0, 0);
         }
     }
 
     __aicore__ __attribute__((always_inline)) inline void Run()
     {
         __gm__ const MLAPrefillTiling* __restrict prefillTiling = &(tiling_gm->prefillTiling);
         uint64_t cur_batch = 0;
         uint32_t pre_total_q_blk_num = 0;
         uint32_t offset_tiling = tiling_head_size + tiling_para_size * cur_batch;
         uint32_t cur_total_q_blk_num = (uint32_t)(prefillTiling->totalQBlkNum[cur_batch]);
         uint32_t process_num = total_q_blk_num * q_heads;
         uint32_t next_process = 0;
         bool swa_flag = (window_size > 0 && MASK_TYPE == 2);
         
         for (uint32_t process = block_idx; process < process_num; process = next_process) {

             while (process >= cur_total_q_blk_num * q_heads) {
                 cur_batch++;
                 pre_total_q_blk_num = cur_total_q_blk_num;
                 offset_tiling += tiling_para_size;
                 cur_total_q_blk_num = (uint32_t)(prefillTiling->totalQBlkNum[cur_batch]);
             }
             next_process = process + block_num;
             if (is_triu_mask) {
                 uint32_t curr_iter = process / block_num;
                 next_process = curr_iter % 2 == 1 ? (curr_iter + 1) * block_num + block_idx : (curr_iter + 2) * block_num - 1 - block_idx;
             }
             // get tiling args
             uint32_t q_seqlen = (uint32_t)(prefillTiling->qSeqLen[cur_batch]);
             uint32_t kv_seqlen = (uint32_t)(prefillTiling->kvSeqlen[cur_batch]);

             if (q_seqlen == 0 || kv_seqlen == 0) {
                 continue;
             }
             uint32_t pp_m_scalar = (uint32_t)(prefillTiling->mUbd[cur_batch]);
             uint32_t pp_n_scalar = (uint32_t)(prefillTiling->indexNN[cur_batch]);
             uint32_t addr_q_high32 = (uint32_t)(prefillTiling->qSeqHigh[cur_batch]);
             uint32_t addr_q_loww32 = (uint32_t)(prefillTiling->qSeqLow[cur_batch]);
             uint64_t addr_q_scalar = (uint64_t)(((uint64_t)addr_q_high32) << 32 | addr_q_loww32);
             uint32_t addr_k_high32 = (uint32_t)(prefillTiling->kSeqHigh[cur_batch]);
             uint32_t addr_k_loww32 = (uint32_t)(prefillTiling->kSeqLow[cur_batch]);
             uint64_t addr_k_scalar = (uint64_t)(((uint64_t)addr_k_high32) << 32 | addr_k_loww32);
             uint32_t addr_v_high32 = (uint32_t)(prefillTiling->vSeqHigh[cur_batch]);
             uint32_t addr_v_loww32 = (uint32_t)(prefillTiling->vSeqLow[cur_batch]);
             uint64_t addr_v_scalar = (uint64_t)(((uint64_t)addr_v_high32) << 32 | addr_v_loww32);

             uint32_t process_idx = process - pre_total_q_blk_num * q_heads;
             uint32_t m_idx = process_idx / q_heads;
             uint64_t head_idx = process_idx % q_heads;
 
             uint32_t m_loop = (q_seqlen + pp_m_scalar - 1) / pp_m_scalar;
             uint32_t n_loop = (kv_seqlen + pp_n_scalar - 1) / pp_n_scalar;
 
             uint32_t qk_m = (m_idx == (m_loop - 1)) ? (q_seqlen - m_idx * pp_m_scalar) : pp_m_scalar;
             uint32_t qk_round_m = (qk_m + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
 
             /**************** pre_load *****************/
             uint32_t qk_n = n_loop == 1 ? kv_seqlen : pp_n_scalar;
             uint32_t qk_round_n = (qk_n + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
 
             uint32_t pingpong_flag = 0;
             uint32_t offset = pingpong_flag * L0AB_HALF_BUF_SIZE;
 
             uint64_t q_offset = addr_q_scalar + head_idx * 128 + (uint64_t)m_idx * pp_m_scalar * q_heads * 128;
             uint64_t k_offset = addr_k_scalar + (head_idx / group_num) * 128;
             uint64_t k_rope_offset = addr_k_scalar / 2 + (head_idx / group_num) * 64;
             uint64_t q_rope_offset = addr_q_scalar / 2 + head_idx * 64 + (uint64_t)m_idx * pp_m_scalar * q_heads * 64;
             if (data_shape_type == 1) {
                 q_offset = addr_q_scalar + head_idx * embdv * max_seqlen +(uint64_t) m_idx * pp_m_scalar * stride_qo;//  + 17592186044416 / sizeof(QKV_DT);
                 k_offset = addr_k_scalar + (head_idx / group_num) * embdv * max_kv_seqlen;//  + 17592186044416 / sizeof(QKV_DT);
             }
 
             // Only need load Q once
             if (qk_m == 1) {
                 gm_to_l1<ArchType::ASCEND_V220, QKV_DT, DataFormat::ND, DataFormat::ND>(
                     l1q_buf_addr_tensor,
                     q_gm_tensor[q_offset],
                     1,
                     0,
                     0,
                     128,              // lenBurst
                     0,
                     0
                 );
                 gm_to_l1<ArchType::ASCEND_V220, QKV_DT, DataFormat::ND, DataFormat::ND>(
                     l1q_buf_addr_tensor[16 * 128],
                     q_rope_gm_tensor[q_rope_offset],
                     1,
                     0,
                     0,
                     64,              // lenBurst
                     0,
                     0
                 );
             } else {
                 gm_to_l1<ArchType::ASCEND_V220, QKV_DT, DataFormat::ND, DataFormat::NZ>(
                     l1q_buf_addr_tensor,
                     q_gm_tensor[q_offset],
                     qk_m,       // nValue
                     qk_round_m, // dstNzC0Stride
                     0,
                     128,        // dValue
                     0,
                     stride_qo  // srcDValue
                 );
                 gm_to_l1<ArchType::ASCEND_V220, QKV_DT, DataFormat::ND, DataFormat::NZ>(
                     l1q_buf_addr_tensor[qk_round_m * 128],
                     q_rope_gm_tensor[q_rope_offset],
                     qk_m,       // nValue
                     qk_round_m, // dstNzC0Stride
                     0,
                     64,        // dValue
                     0,
                     stride_qo / 2
                 );
             }
             SET_FLAG(MTE2, MTE1, pingpong_flag);
             WAIT_FLAG(MTE2, MTE1, pingpong_flag);
             uint32_t loopQK = (this->embd + BLOCK_QK - 1) / BLOCK_QK;
             uint32_t loopV = (this->embdv + BLOCK_QK - 1) / BLOCK_QK;
             uint32_t qk_k = BLOCK_QK;
             uint32_t qk_round_k = BLOCK_QK;
 
             uint32_t sv_n = n_loop == 1 ? kv_seqlen : pp_n_scalar;
             uint32_t sv_round_n = (sv_n + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
             uint64_t v_offset = addr_v_scalar + (head_idx / group_num) * embdv;//  + 17592186044416 / sizeof(QKV_DT);
             if (data_shape_type == 1) {
                 v_offset = addr_v_scalar + (head_idx / group_num) * embdv * max_kv_seqlen;//  + 17592186044416 / sizeof(QKV_DT);
             }
 
             uint32_t n_end = n_loop;
             uint32_t window_start = (window_size + pp_n_scalar - 1) / pp_n_scalar;
             uint32_t n_start = 0;
             if (is_triu_mask || window_size > 0) {
                 n_end = m_idx + 1;
             }
             if (swa_flag) {
                 if (window_size > 0 && window_size < kv_seqlen){
                     n_start = (m_idx < window_start) ? 0 : m_idx - window_start;
                     k_offset += n_start * stride_k * pp_n_scalar;
                     v_offset += n_start * stride_v * pp_n_scalar;
                     k_rope_offset += n_start * stride_k * pp_n_scalar / 2;
                 }
             }
 
             uint32_t launch_delay = S_BLOCK_STACK * 2;
             uint32_t vect_mod = 2 * launch_delay;
             uint32_t kv_pingpong_flag = 0;
             uint64_t kv_pingpong_offset = kv_pingpong_flag * KV_DB_SIZE;
             uint32_t split_m = 128;
             uint32_t m_inner_loop = (qk_m + 127) / 128;
             for (uint32_t n_idx = n_start; n_idx < n_end + launch_delay; n_idx += S_BLOCK_STACK) {
                 if (n_idx < n_end) {
                     uint32_t sv_n_triu = n_end * pp_n_scalar;
                     uint32_t l0c_pingpong_flag = 0;
                     uint32_t l0c_offset = l0c_pingpong_flag * L0AB_HALF_BUF_SIZE;
                     if (n_idx + S_BLOCK_STACK > n_end - 1) {
                         sv_n = sv_n_triu > kv_seqlen ? kv_seqlen - n_idx * pp_n_scalar : sv_n_triu - n_idx * pp_n_scalar;
                     } else {
                         sv_n = pp_n_scalar * S_BLOCK_STACK;
                     }
                     uint32_t sv_round_n = (sv_n + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
                     WAIT_FLAG(M, MTE1, EVENT_ID0);
                     WAIT_FLAG(M, MTE1, EVENT_ID1);
                     LoadDataToCa(l0a_buf_tensor, l1q_buf_addr_tensor, 128, qk_round_m, qk_m);
                     LoadDataToCa(l0a_buf_tensor[L0AB_HALF_BUF_SIZE], l1q_buf_addr_tensor[qk_round_m * 128], 64, qk_round_m, qk_m);
                     // *** Prepare K to L1
                     SET_FLAG(MTE1, M, EVENT_ID0);
                     WAIT_FLAG(MTE1, M, EVENT_ID0);
                     for (uint32_t split_idx = 0; split_idx < S_BLOCK_STACK && n_idx + split_idx < n_end; split_idx++) {
                         pingpong_flag = (n_idx + split_idx - n_start) % 2;
                         offset = pingpong_flag * 128 * 192;
                         if (n_idx + split_idx == (n_loop - 1)) {
                             qk_n = (kv_seqlen - (n_idx + split_idx) * pp_n_scalar);
                             qk_round_n = (qk_n + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
                         }
                         bool last_split = split_idx == S_BLOCK_STACK - 1 || n_idx + split_idx == n_end - 1;
 
                         WAIT_FLAG(MTE1, MTE2, pingpong_flag + 2 * kv_pingpong_flag);
                         gm_to_l1<ArchType::ASCEND_V220, QKV_DT, DataFormat::ND, DataFormat::NZ>(
                             l1k_buf_addr_tensor[kv_pingpong_offset + offset],
                             k_gm_tensor[k_offset],
                             qk_n,        // nValue
                             qk_round_n,  // dstNzC0Stride
                             0,            // dstNzMatrixStride, unused
                             128,         // dValue
                             0,            // dstNzMatrixStride, unused
                             stride_k   // srcDValue
                         );
                         gm_to_l1<ArchType::ASCEND_V220, QKV_DT, DataFormat::ND, DataFormat::NZ>(
                             l1k_buf_addr_tensor[kv_pingpong_offset + offset + qk_round_n * 128],
                             k_rope_gm_tensor[k_rope_offset],
                             qk_n,        // nValue
                             qk_round_n,  // dstNzC0Stride
                             0,            // dstNzMatrixStride, unused
                             64,         // dValue
                             0,            // dstNzMatrixStride, unused
                             stride_k / 2   // srcDValue
                         );
                         SET_FLAG(MTE2, MTE1, pingpong_flag);
                         WAIT_FLAG(MTE2, MTE1, pingpong_flag);
                         for (int embed_idx = 0; embed_idx < loopQK; embed_idx++) {
                             uint32_t embedk = (embed_idx == 0) ? 128 : 64;
                             uint32_t last_embed_loop = (embed_idx == (loopQK - 1)) ? 1 : 0;
                             uint32_t initc = (embed_idx == 0) ? 1: 0;
                             uint32_t embed_flag = embed_idx % 2;
                             WAIT_FLAG(M, MTE1, embed_flag + 2);
                             l1_to_l0_b<ArchType::ASCEND_V220, QKV_DT, false, DataFormat::VECTOR, DataFormat::VECTOR>(
                                 l0b_buf_tensor[embed_flag * L0AB_HALF_BUF_SIZE],
                                 l1k_buf_addr_tensor[kv_pingpong_offset + offset + embed_idx * qk_round_n * 128],
                                 0,
                                 NumMatrixsRoundUp<QKV_DT>(RoundUp<uint32_t>(embedk, 32 / sizeof(QKV_DT)) * qk_round_n), // repeat
                                 0,
                                 1,                                        // srcStride
                                 0,
                                 0                                        // dstStride
                             );
 
                             if (last_embed_loop) {
                                 SET_FLAG(MTE1, MTE2, pingpong_flag + 2 * kv_pingpong_flag);
                             }
 
                             SET_FLAG(MTE1, M, pingpong_flag + 2);
                             WAIT_FLAG(MTE1, M, pingpong_flag + 2);
 
                             if (embed_idx == 0) {
                                 WAIT_FLAG(FIX, M, l0c_pingpong_flag);
                             }
 
                             mmad<ArchType::ASCEND_V220, QKV_DT, QKV_DT, float, false>(
                                 l0c_buf_tensor[l0c_offset],
                                 l0a_buf_tensor[embed_idx * L0AB_HALF_BUF_SIZE],
                                 l0b_buf_tensor[embed_flag * L0AB_HALF_BUF_SIZE],
                                 qk_m,  // m
                                 qk_n,  // n
                                 embedk,   // k
                                 initc      // cmatrixInitVal
                             );
                             PIPE_BARRIER(M);
 
                             SET_FLAG(M, MTE1, embed_flag + 2);
 
                             if (last_embed_loop) {
                                 SET_FLAG(M, FIX, l0c_pingpong_flag);
                                 WAIT_FLAG(M, FIX, l0c_pingpong_flag);
 
                                 l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, half, float>(
                                     s_gm_tensor[(uint64_t)block_idx * TMP_SIZE + (n_idx - n_start) % vect_mod * TMP_SIZE / vect_mod + split_idx * pp_n_scalar],
                                     l0c_buf_tensor[l0c_offset],
                                     qk_m,        // MSize
                                     qk_round_n,  // NSize
                                     qk_round_m,  // srcStride
                                     sv_round_n  // dstStride_dst_D
                                 );
                                 SET_FLAG(FIX, M, l0c_pingpong_flag);
                             }
                         }
                         k_offset += pp_n_scalar * stride_k;
                         k_rope_offset += pp_n_scalar * stride_k / 2;
                         l0c_pingpong_flag = 1 - l0c_pingpong_flag;
                         l0c_offset = l0c_pingpong_flag * L0AB_HALF_BUF_SIZE;
                     }
                     SET_FLAG(M, MTE1, EVENT_ID0);
                     SET_FLAG(M, MTE1, EVENT_ID1);
                     FftsCrossCoreSync<PIPE_FIX, 2>(QK_READY);
                     kv_pingpong_flag = 1 - kv_pingpong_flag;
                     kv_pingpong_offset = kv_pingpong_flag * KV_DB_SIZE;
                 }
                 if (n_idx >= launch_delay + n_start) {
                     uint32_t l0c_pingpong_flag = 0;
                     uint32_t l0c_offset = l0c_pingpong_flag * L0AB_HALF_BUF_SIZE;
                     uint32_t sv_n_triu = n_end * pp_n_scalar;
                     if (n_idx + S_BLOCK_STACK > n_end + launch_delay - 1) {
                         sv_n = sv_n_triu > kv_seqlen ? kv_seqlen - (n_idx - launch_delay) * pp_n_scalar : sv_n_triu - (n_idx - launch_delay) * pp_n_scalar;
                     } else {
                         sv_n = pp_n_scalar * S_BLOCK_STACK;
                     }
                     uint32_t sv_round_n = (sv_n + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
                     uint32_t n_slice = pp_n_scalar * ((S_BLOCK_STACK + 1) / 2);
                     // n_slice = 256;
                     uint32_t l1_split_loop = (sv_n + n_slice - 1) / n_slice;
                     WAIT_FLAG(MTE1, MTE2, kv_pingpong_flag * 2);
                     WAIT_FLAG(MTE1, MTE2, kv_pingpong_flag * 2 + 1);
                     gm_to_l1<ArchType::ASCEND_V220, QKV_DT, DataFormat::ND, DataFormat::NZ>(
                         l1v_buf_addr_tensor[kv_pingpong_offset],
                         v_gm_tensor[v_offset],
                         sv_n,       // nValue
                         sv_round_n, // dstNzC0Stride
                         0,          // dstNzMatrixStride, unused
                         embdv,        // dValue
                         0,          // dstNzMatrixStride, unused
                         stride_v   // srcDValue
                     );
                     v_offset += sv_n * stride_v;
 
                     WaitFlagDev(SOFTMAX_READY);
                     for (uint32_t gm_split_idx = 0; gm_split_idx < m_inner_loop; gm_split_idx++) {
                         WAIT_FLAG(FIX, M, l0c_pingpong_flag);
                         bool m_last_split = gm_split_idx == m_inner_loop - 1;
                         uint64_t gm_p_offset = gm_split_idx * split_m * sv_round_n;
                         uint32_t nowM = m_last_split ? qk_m - gm_split_idx * split_m : split_m;
                         uint32_t nowMRound = (nowM + 15) / 16 * 16;
                         uint32_t qk_round_k = 128;
                         for (uint32_t l1_k_split_idx = 0; l1_k_split_idx < l1_split_loop; l1_k_split_idx++) {
                             uint32_t l1_pingpong_flag = l1_k_split_idx % 2;
                             uint32_t l1_offset = l1_pingpong_flag * 128 * 256;
                             bool l1_last_split = l1_k_split_idx == l1_split_loop - 1;
                             uint32_t d = l1_last_split ? sv_n - l1_k_split_idx * n_slice : n_slice;
                             WAIT_FLAG(MTE1, MTE2, l1_pingpong_flag + 4);
                             if (nowM == 1) {
                                 gm_to_l1<ArchType::ASCEND_V220, QKV_DT, DataFormat::ND, DataFormat::ND>(
                                     l1p_buf_addr_tensor[l1_offset],
                                     p_gm_tensor[((uint64_t)block_idx * TMP_SIZE +
                                                 (n_idx - launch_delay - n_start) % vect_mod * TMP_SIZE / vect_mod) *
                                                 2 / sizeof(QKV_DT) + l1_k_split_idx * n_slice + gm_p_offset],
                                     1, 0, 0, RoundUp<uint64_t>(sv_round_n, BlockSize<QKV_DT>()), // lenBurst
                                     0, 0);
                             } else {
                                 gm_to_l1<ArchType::ASCEND_V220, QKV_DT, DataFormat::ND, DataFormat::NZ>(
                                     l1p_buf_addr_tensor[l1_offset],
                                     p_gm_tensor[((uint64_t)block_idx * TMP_SIZE +
                                                 (n_idx - launch_delay - n_start) % vect_mod * TMP_SIZE / vect_mod) *
                                                 2 / sizeof(QKV_DT) + l1_k_split_idx * n_slice + gm_p_offset],
                                     nowM,                           // nValue
                                     nowMRound,                     // dstNzC0Stride
                                     0,                              // dstNzMatrixStride, unused
                                     d,                           // dValue
                                     0,                              // dstNzMatrixStride, unused
                                     sv_round_n * 2 / sizeof(QKV_DT) // srcDValue
                                 );
                             }
                             SET_FLAG(MTE2, MTE1, l1_pingpong_flag + 4);
                             WAIT_FLAG(MTE2, MTE1, l1_pingpong_flag + 4);
                             uint32_t d_split_loop = (d + 127) / 128;
                             for (uint32_t l0_k_split_idx = 0; l0_k_split_idx < d_split_loop; l0_k_split_idx++) {
                                 uint32_t l0_pingpong_flag = l0_k_split_idx % 2;
                                 uint32_t l0_offset = l0_pingpong_flag * L0AB_HALF_BUF_SIZE;
                                 bool l0_last_split = l0_k_split_idx == d_split_loop - 1;
                                 int32_t l0_p_offset = nowM == 1 ? l0_k_split_idx * 128 : l0_k_split_idx * 128 * nowMRound;
                                 bool initC = l0_k_split_idx== 0 && l1_k_split_idx == 0;
                                 uint32_t reduce_d = l0_last_split ? d - l0_k_split_idx * 128 : 128;
                                 uint32_t round_reduce_d = (reduce_d + 15) / 16 * 16;
                                 WAIT_FLAG(M, MTE1, l0_pingpong_flag);
                                 LoadDataToCa(l0a_buf_tensor[l0_offset], l1p_buf_addr_tensor[l1_offset + l0_p_offset],
                                             RoundUp<uint64_t>(round_reduce_d, BlockSize<QKV_DT>()), nowMRound, nowM);
                                 if (l0_last_split){
                                     SET_FLAG(MTE1, MTE2, l1_pingpong_flag + 4);
                                 }
                                 WAIT_FLAG(M, MTE1, l0_pingpong_flag + 2);
                                 for (uint32_t l0b_load_idx = 0; l0b_load_idx < 128 / BLOCK_SIZE; ++l0b_load_idx) {
                                     l1_to_l0_b<ArchType::ASCEND_V220, QKV_DT, true, DataFormat::VECTOR, DataFormat::VECTOR>(
                                         l0b_buf_tensor[l0_offset + l0b_load_idx * qk_round_k * BLOCK_SIZE],
                                         l1v_buf_addr_tensor[kv_pingpong_offset + l0b_load_idx * CUBE_MATRIX_SIZE +
                                                             l1_k_split_idx * n_slice * BLOCK_SIZE +
                                                             l0_k_split_idx * 128 * BLOCK_SIZE],
                                         0,
                                         qk_round_k / BLOCK_SIZE, // repeat
                                         0,
                                         sv_round_n / BLOCK_SIZE, // srcStride
                                         0,
                                         0 // dstStride
                                     );
                                 }
                                 if (l0_last_split && l1_last_split && m_last_split){
                                     SET_FLAG(MTE1, MTE2, kv_pingpong_flag * 2);
                                     SET_FLAG(MTE1, MTE2, kv_pingpong_flag * 2 + 1);
                                 }
                                 SET_FLAG(MTE1, M, l0_pingpong_flag + 6);
                                 WAIT_FLAG(MTE1, M, l0_pingpong_flag + 6);
                                 mmad<ArchType::ASCEND_V220, QKV_DT, QKV_DT, float, false>(
                                     l0c_buf_tensor[l0c_offset],
                                     l0a_buf_tensor[l0_offset],
                                     l0b_buf_tensor[l0_offset],
                                     nowM, // m
                                     qk_round_k,  // n
                                     reduce_d, // k
                                     initC     // cmatrixInitVal
                                 );
                                 PIPE_BARRIER(M);
                                 SET_FLAG(M, MTE1, l0_pingpong_flag);
                                 SET_FLAG(M, MTE1, l0_pingpong_flag + 2);
                             }
                         }
                         SET_FLAG(M, FIX, l0c_pingpong_flag);
                         WAIT_FLAG(M, FIX, l0c_pingpong_flag);
                         // // copy O to gm
                         l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, float, float>(
                             o_tmp_gm_tensor[(uint64_t)block_idx * TMP_SIZE + (n_idx - launch_delay - n_start) % vect_mod * TMP_SIZE / vect_mod + gm_split_idx * split_m * qk_round_k],
                             l0c_buf_tensor[l0c_offset],
                             nowM,        // MSize
                             qk_round_k,     // NSize
                             nowMRound,  // srcStride
                             qk_round_k     // dstStride_dst_D
                         );
                         SET_FLAG(FIX, M, l0c_pingpong_flag);
                     }
                     FftsCrossCoreSync<PIPE_FIX, 2>(UPDATE_READY);
                     kv_pingpong_flag = 1 - kv_pingpong_flag;
                     kv_pingpong_offset = kv_pingpong_flag * KV_DB_SIZE;
                 }
             }
         }
     }
 
 private:
     __gm__ QKV_DT *__restrict__ query_gm{nullptr};
     __gm__ QKV_DT *__restrict__ qRope_gm{nullptr};
     __gm__ QKV_DT *__restrict__ key_gm{nullptr};
     __gm__ QKV_DT *__restrict__ kRope_gm{nullptr};
     __gm__ QKV_DT *__restrict__ value_gm{nullptr};
     __gm__ MLAPrefillTilingData* __restrict__ tiling_gm{nullptr};
 
     const uint32_t l1q_buf_addr_offset = 0;
     const uint32_t l1k_buf_addr_offset = 4 * L0AB_UINT8_BLOCK_SIZE;
     const uint32_t l1v_buf_addr_offset = 4 * L0AB_UINT8_BLOCK_SIZE;
     const uint32_t l1p_buf_addr_offset = 12 * L0AB_UINT8_BLOCK_SIZE;
 
     AsdopsBuffer<ArchType::ASCEND_V220> buf;
 
     AscendC::LocalTensor<QKV_DT> l1q_buf_addr_tensor = buf.GetBuffer<BufferType::ASCEND_CB, QKV_DT>(l1q_buf_addr_offset);
     AscendC::LocalTensor<QKV_DT> l1k_buf_addr_tensor = buf.GetBuffer<BufferType::ASCEND_CB, QKV_DT>(l1k_buf_addr_offset);
     AscendC::LocalTensor<QKV_DT> l1p_buf_addr_tensor = buf.GetBuffer<BufferType::ASCEND_CB, QKV_DT>(l1p_buf_addr_offset);
     AscendC::LocalTensor<QKV_DT> l1v_buf_addr_tensor = buf.GetBuffer<BufferType::ASCEND_CB, QKV_DT>(l1v_buf_addr_offset);
 
     AscendC::GlobalTensor<QKV_DT> q_gm_tensor;
     AscendC::GlobalTensor<QKV_DT> q_rope_gm_tensor;
     AscendC::GlobalTensor<QKV_DT> k_gm_tensor;
     AscendC::GlobalTensor<QKV_DT> k_rope_gm_tensor;
     AscendC::GlobalTensor<QKV_DT> v_gm_tensor;
     AscendC::GlobalTensor<QKV_DT> s_gm_tensor;
     AscendC::GlobalTensor<QKV_DT> p_gm_tensor;
     AscendC::GlobalTensor<float> o_tmp_gm_tensor;
     AscendC::GlobalTensor<float> upo_gm_tensor;
 
     AscendC::LocalTensor<QKV_DT> l0a_buf_tensor = buf.GetBuffer<BufferType::ASCEND_L0A, QKV_DT>(0);
     AscendC::LocalTensor<QKV_DT> l0b_buf_tensor = buf.GetBuffer<BufferType::ASCEND_L0B, QKV_DT>(0);
     AscendC::LocalTensor<float> l0c_buf_tensor = buf.GetBuffer<BufferType::ASCEND_L0C, float>(0);
 
     uint32_t batch_size{0};
     uint32_t max_seqlen{0};
     uint32_t max_kv_seqlen{0};
     uint32_t q_heads{0};
     uint32_t embd{0};
     uint32_t embdv{0};
     uint32_t kv_heads{0};
     uint32_t is_triu_mask{0};
     uint32_t total_q_blk_num{0};
     uint32_t group_num{0};
     uint64_t stride_qo{0};
     uint64_t stride_o{0};
     uint64_t stride_v{0};
     uint64_t stride_k{0};
     uint64_t batch_stride_kv{0};
     uint32_t __k{0};
     uint32_t round_k{0};
     uint32_t data_shape_type{0};
 
     uint32_t tiling_head_size{0};
     uint32_t tiling_para_size{0};
     uint32_t window_size{0};
     uint32_t block_dim{0};
 };

 extern "C" __global__ __aicore__ void mla_prefill(
    __gm__ uint8_t *__restrict__ query,
    __gm__ uint8_t *__restrict__ qRope,
    __gm__ uint8_t *__restrict__ key,
    __gm__ uint8_t *__restrict__ kRope,
    __gm__ uint8_t *__restrict__ value,
    __gm__ uint8_t *__restrict__ qSeqLen,
    __gm__ uint8_t *__restrict__ kvSeqLen,
    __gm__ uint8_t *__restrict__ mask,
    __gm__ uint8_t *__restrict__ attenOut,
    __gm__ uint8_t *__restrict__ workspace,
    __gm__ uint8_t *__restrict__ tiling)
 {
    GET_TILING_DATA_WITH_STRUCT(MLAPrefillTilingData, tiling_data_in, tiling);

    __gm__ const MLAPrefillTilingData* __restrict tiling_data = reinterpret_cast<__gm__ const MLAPrefillTilingData*>(tiling);

    __gm__ uint8_t* user = AscendC::GetUserWorkspace(workspace);

     if (TILING_KEY_IS(1)) {
         MLAPrefillOp_C<half, half, half, 0> fa_cube{};
         fa_cube.Init(query, qRope, key, kRope, value, user, tiling_data);
         fa_cube.Run();
     } else if (TILING_KEY_IS(2)) {
        MLAPrefillOp_C<half, half, half, 1> fa_cube{};
        fa_cube.Init(query, qRope, key, kRope, value, user, tiling_data);
        fa_cube.Run();
     } else if (TILING_KEY_IS(3)) {
        MLAPrefillOp_C<half, half, half, 2> fa_cube{};
        fa_cube.Init(query, qRope, key, kRope, value, user, tiling_data);
        fa_cube.Run();
     }
 }

 #endif

 #ifdef __DAV_C220_VEC__

__aicore__ __attribute__((always_inline)) inline void __set_mask(int32_t len)
{
    uint64_t mask = 0;
    uint64_t one = 1;
    uint64_t temp = len % FLOAT_VECTOR_SIZE;
    for (int64_t i = 0; i < temp; i++) {
        mask |= one << i;
    }

    if (len == VECTOR_SIZE || len == 0) {
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    } else if (len >= FLOAT_VECTOR_SIZE) {
        SetVectorMask<int8_t>(mask, (uint64_t)-1);
    } else {
        SetVectorMask<int8_t>(0x0, mask);
    }
}

template<typename S_DTYPE, typename EXP_DTYPE, typename P_DTYPE, typename MASK_DTYPE, uint32_t MASK_TYPE>
__aicore__ __attribute__((always_inline)) void OnlineSoftmaxStage1FP16(
    const AscendC::LocalTensor<S_DTYPE> &s_ub,
    const AscendC::LocalTensor<S_DTYPE> &mask_processed_ub,
    const AscendC::LocalTensor<S_DTYPE> &local_rowmax_ub,
    const AscendC::LocalTensor<S_DTYPE> &hat_rowmax_ub,
    const AscendC::LocalTensor<S_DTYPE> &global_rowmax_ub,
    const AscendC::LocalTensor<S_DTYPE> &diff16_rowmax_ub,
    const AscendC::LocalTensor<EXP_DTYPE> &diff_rowmax_ub,
    const AscendC::LocalTensor<EXP_DTYPE> &s_exp_ub,
    const AscendC::LocalTensor<EXP_DTYPE> &local_rowsum_ub,
    const AscendC::LocalTensor<EXP_DTYPE> &global_rowsum_ub,
    const AscendC::LocalTensor<P_DTYPE> &p_ub,
    const AscendC::LocalTensor<EXP_DTYPE> &tmp_ub,
    const AscendC::LocalTensor<S_DTYPE> &tmp16_ub,
    const AscendC::GlobalTensor<S_DTYPE> &s_gm,
    const AscendC::GlobalTensor<P_DTYPE> &p_gm,
    bool first_n_iter, half tor,
    uint32_t m, uint32_t n_real, uint32_t n_stride, uint32_t pingpong_flag,
    bool move_flag)
{
    uint32_t round_m = (m + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    uint32_t round_m_8 = (m + FLOAT_BLOCK_SIZE - 1) / FLOAT_BLOCK_SIZE * FLOAT_BLOCK_SIZE;

    WAIT_FLAG(MTE3, MTE2, pingpong_flag);
    gm_to_ub<ArchType::ASCEND_V220, S_DTYPE>(
        s_ub,
        s_gm,
        0,                                // sid
        m,                                // nBurst
        n_stride / BLOCK_SIZE,  // lenBurst
        0,                                // srcGap
        0                                 // dstGap
    );
    SET_FLAG(MTE2, V, pingpong_flag);
    WAIT_FLAG(MTE2, V, pingpong_flag);
    // *** ls = tor * ls
    muls_v<ArchType::ASCEND_V220, S_DTYPE>(
        s_ub,
        s_ub,
        (S_DTYPE)tor,
        (m * n_stride + VECTOR_SIZE - 1) / VECTOR_SIZE, // repeat
        1,                                                                // dstBlockStride
        1,                                                                // srcBlockStride
        8,                                                                // dstRepeatStride
        8                                                                 // srcRepeatStride
    );

    PIPE_BARRIER(V);

    if constexpr(MASK_TYPE == 1) {
        if (move_flag) {
            WAIT_FLAG(MTE2, V, EVENT_ID6);
            add_v<ArchType::ASCEND_V220, S_DTYPE>(
                s_ub,
                s_ub,
                mask_processed_ub,
                (m * n_stride + VECTOR_SIZE - 1) / VECTOR_SIZE, // repeat
                1,                                                                // dstBlockStride
                1,                                                                // src0BlockStride
                1,                                                                // src1BlockStride
                8,                                                                // dstRepeatStride
                8,                                                                // src0RepeatStride
                8                                                                 // src1RepeatStride
            );
            SET_FLAG(V, MTE2, EVENT_ID6);
            PIPE_BARRIER(V);
        }
    } else {
        if constexpr(MASK_TYPE == 2) {
            WAIT_FLAG(MTE2, V, EVENT_ID6);
            add_v<ArchType::ASCEND_V220, S_DTYPE>(
                s_ub,
                s_ub,
                mask_processed_ub,
                (m * n_stride + VECTOR_SIZE - 1) / VECTOR_SIZE, // repeat
                1,                                                                // dstBlockStride
                1,                                                                // src0BlockStride
                1,                                                                // src1BlockStride
                8,                                                                // dstRepeatStride
                8,                                                                // src0RepeatStride
                8                                                                 // src1RepeatStride
            );
            SET_FLAG(V, MTE2, EVENT_ID6);
            PIPE_BARRIER(V);
        }
    }
    PIPE_BARRIER(ALL);

    if (n_real == 512) {
        Rowmax<S_DTYPE, RowCalcTile::SPEC_TILE_512>(
            s_ub,
            local_rowmax_ub,
            tmp16_ub,
            round_m, n_real, n_stride
        );
    } else if (n_real == 256) {
        Rowmax<S_DTYPE, RowCalcTile::SPEC_TILE_256>(
            s_ub,
            local_rowmax_ub,
            tmp16_ub,
            round_m, n_real, n_stride
        );
    } else {
        Rowmax<S_DTYPE, RowCalcTile::TAIL_TILE>(
            s_ub,
            local_rowmax_ub,
            tmp16_ub,
            round_m, n_real, n_stride
        );
    }
    if (first_n_iter) {
        // *** hm = lm
        ub_to_ub<ArchType::ASCEND_V220, S_DTYPE>(
            hat_rowmax_ub,
            local_rowmax_ub,
            0,                              // sid
            1,                              // nBurst
            round_m / BLOCK_SIZE, // lenBurst
            0,                              // srcGap
            0                               // dstGap
        );
        PIPE_BARRIER(V);
    } else {
        __set_mask(m);
        // *** hm = vmax(lm, gm)
        max_v<ArchType::ASCEND_V220, S_DTYPE>(
            hat_rowmax_ub,
            local_rowmax_ub,
            global_rowmax_ub,
            1,         // repeat
            1,         // dstBlockStride
            1,         // src0BlockStride
            1,         // src1BlockStride
            8,         // dstRepeatStride
            8,         // src0RepeatStride
            8          // src1RepeatStride
        );
        PIPE_BARRIER(V);
        // *** dm = gm - hm
        sub_v<ArchType::ASCEND_V220, S_DTYPE>(
            diff16_rowmax_ub,
            global_rowmax_ub,
            hat_rowmax_ub,
            1,         // repeat
            1,         // dstBlockStride
            1,         // src0BlockStride
            1,         // src1BlockStride
            8,         // dstRepeatStride
            8,         // src0RepeatStride
            8          // src1RepeatStride
        );
        PIPE_BARRIER(V);
        // dm fp16 cast to fp32
        conv_v<ArchType::ASCEND_V220, S_DTYPE, EXP_DTYPE>(
            diff_rowmax_ub,
            diff16_rowmax_ub,
            1,  // repeat
            1,                                                                 // dstBlockStride
            1,                                                                 // srcBlockStride
            8,                                                                 // dstRepeatStride
            4                                                                  // srcRepeatStride
        );
        PIPE_BARRIER(V);
        // *** dm = exp(dm)
        exp_v<ArchType::ASCEND_V220, EXP_DTYPE>(
            diff_rowmax_ub,
            diff_rowmax_ub,
            1,         // repeat
            1,         // dstBlockStride
            1,         // srcBlockStride
            8,         // dstRepeatStride
            8          // srcRepeatStride
        );
    }
    SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    PIPE_BARRIER(V);
    // *** gm = hm
    ub_to_ub<ArchType::ASCEND_V220, S_DTYPE>(
        global_rowmax_ub,
        hat_rowmax_ub,
        0,                              // sid
        1,                              // nBurst
        round_m / BLOCK_SIZE, // lenBurst
        0,                              // srcGap
        0                               // dstGap
    );
    PIPE_BARRIER(V);
    // *** hm_block = expand_to_block(hm), 存放于 tv
    brcb_v<ArchType::ASCEND_V220, uint16_t>(
        tmp_ub.template ReinterpretCast<uint16_t>(),
        hat_rowmax_ub.template ReinterpretCast<uint16_t>(),
        1,                             // dstBlockStride
        8,                             // dstRepeatStride
        round_m / FLOAT_BLOCK_SIZE // repeat
    );
    PIPE_BARRIER(V);
    for (uint32_t vsub_idx = 0; vsub_idx < n_real / VECTOR_SIZE; ++vsub_idx) {
        sub_v<ArchType::ASCEND_V220, S_DTYPE>(
            s_ub[vsub_idx * VECTOR_SIZE],
            s_ub[vsub_idx * VECTOR_SIZE],
            tmp_ub.template ReinterpretCast<S_DTYPE>(),
            m,                         // repeat
            1,                             // dstBlockStride
            1,                             // src0BlockStride
            0,                             // src1BlockStride
            n_stride / BLOCK_SIZE, // dstRepeatStride
            n_stride / BLOCK_SIZE, // src0RepeatStride
            1                              // src1RepeatStride
        );
    }
    if (n_real % VECTOR_SIZE > 0) {
        __set_mask(n_real % VECTOR_SIZE);
        sub_v<ArchType::ASCEND_V220, S_DTYPE>(
            s_ub[n_real / VECTOR_SIZE * VECTOR_SIZE],
            s_ub[n_real / VECTOR_SIZE * VECTOR_SIZE],
            tmp_ub.template ReinterpretCast<S_DTYPE>(),
            m,                         // repeat
            1,                             // dstBlockStride
            1,                             // src0BlockStride
            0,                             // src1BlockStride
            n_stride / BLOCK_SIZE, // dstRepeatStride
            n_stride / BLOCK_SIZE, // src0RepeatStride
            1                              // src1RepeatStride
        );
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }
    PIPE_BARRIER(V);

    conv_v<ArchType::ASCEND_V220, S_DTYPE, EXP_DTYPE>(
        s_exp_ub,
        s_ub,
        (m * n_stride + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,  // repeat
        1,                                                                 // dstBlockStride
        1,                                                                 // srcBlockStride
        8,                                                                 // dstRepeatStride
        4                                                                  // srcRepeatStride
    );
    PIPE_BARRIER(V);
    // *** ls = exp(ls)
    exp_v<ArchType::ASCEND_V220, EXP_DTYPE>(
        s_exp_ub,
        s_exp_ub,
        (m * n_stride + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE, // repeat
        1,                                                                // dstBlockStride
        1,                                                                // srcBlockStride
        8,                                                                // dstRepeatStride
        8                                                                 // srcRepeatStride
    );
    PIPE_BARRIER(V);
    // *** ll = rowsum(ls32)
    if (n_real == 512) {
        Rowsum<EXP_DTYPE, RowCalcTile::SPEC_TILE_512>(
            s_exp_ub,
            local_rowsum_ub,
            tmp_ub,
            round_m_8, n_real, n_stride
        );
    } else if (n_real == 256) {
        Rowsum<EXP_DTYPE, RowCalcTile::SPEC_TILE_256>(
            s_exp_ub,
            local_rowsum_ub,
            tmp_ub,
            round_m_8, n_real, n_stride
        );
    } else {
        Rowsum<EXP_DTYPE, RowCalcTile::TAIL_TILE>(
            s_exp_ub,
            local_rowsum_ub,
            tmp_ub,
            round_m_8, n_real, n_stride
        );
    }

    // *** lp = castfp32to16(ls)
    conv_v<ArchType::ASCEND_V220, EXP_DTYPE, P_DTYPE>(
        p_ub,
        s_exp_ub,
        (m * n_stride + FLOAT_VECTOR_SIZE - 1) /
            FLOAT_VECTOR_SIZE, // repeat
        1,                     // dstBlockStride
        1,                     // srcBlockStride
        4,                     // dstRepeatStride
        8                      // srcRepeatStride
    );
    SET_FLAG(V, MTE3, pingpong_flag);
    WAIT_FLAG(V, MTE3, pingpong_flag);
    ub_to_gm<ArchType::ASCEND_V220, P_DTYPE>(
        p_gm,
        p_ub,
        0,                                    // sid
        m,                              // nBurst
        n_stride * 2 / BlockSize<int8_t>(), // lenBurst
        0,                                    // srcGap
        0                                     // dstGap
    );
    SET_FLAG(MTE3, MTE2, pingpong_flag);
    if (first_n_iter) {
        // *** gl = ll
        ub_to_ub<ArchType::ASCEND_V220, EXP_DTYPE>(
            global_rowsum_ub,
            local_rowsum_ub,
            0,                              // sid
            1,                              // nBurst
            round_m_8 / FLOAT_BLOCK_SIZE, // lenBurst
            0,                              // srcGap
            0                               // dstGap
        );
        PIPE_BARRIER(V);
    } else {
        __set_mask(m);
        // *** gl = dm * gl
        mul_v<ArchType::ASCEND_V220, EXP_DTYPE>(
            global_rowsum_ub,
            diff_rowmax_ub,
            global_rowsum_ub,
            1, // repeat
            1,         // dstBlockStride
            1,         // src0BlockStride
            1,         // src1BlockStride
            8,         // dstRepeatStride
            8,         // src0RepeatStride
            8          // src1RepeatStride
        );
        PIPE_BARRIER(V);
        // *** gl = ll + gl
        add_v<ArchType::ASCEND_V220, EXP_DTYPE>(
            global_rowsum_ub,
            global_rowsum_ub,
            local_rowsum_ub,
            1, // repeat
            1,         // dstBlockStride
            1,         // src0BlockStride
            1,         // src1BlockStride
            8,         // dstRepeatStride
            8,         // src0RepeatStride
            8          // src1RepeatStride
        );
        PIPE_BARRIER(V);
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }
}

template<typename IN_DATA_TYPE, typename QKV_DT = IN_DATA_TYPE, typename O_DT = IN_DATA_TYPE, int32_t MASK_TYPE = 0>
class MLAPrefillOp_V {
public:
    __aicore__ __attribute__((always_inline)) inline MLAPrefillOp_V() {
    }

    __aicore__ __attribute__((always_inline)) inline void Init(
        __gm__ uint8_t *__restrict__ mask,  
        __gm__ uint8_t *__restrict__ attenOut,
        __gm__ uint8_t *__restrict__ workspace,
        __gm__ const MLAPrefillTilingData* __restrict__ tiling)
    {

        SetAtomicnone();
        SetMasknorm();
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);

        this->mask_gm = mask;
        this->o_gm = attenOut;
        mask_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ IN_DATA_TYPE *>(this->mask_gm));
        o_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ O_DT *>(this->o_gm));

        this->sub_block_idx = GetSubBlockidx();

        tiling_gm = const_cast<__gm__ MLAPrefillTilingData*>(tiling);
        __gm__ const MLAPrefillHeadParams* __restrict headparams = &(tiling_gm->MLAPrefillHead);

        this->batch_size = (uint32_t)(headparams->batch);
        this->max_seqlen = (uint32_t)(headparams->maxSeqLen);
        this->q_heads = (uint32_t)(headparams->numHeads);
        this->embd = (uint32_t)(headparams->embeddingSize);
        this->embdv = (uint32_t)(headparams->embeddingSizeV);
        this->tor = (half)(headparams->tor);
        this->head_stride = (uint32_t)(headparams->headStride);
        this->mask_stride = (uint32_t)(headparams->maskStride);
        this->total_q_blk_num = (uint32_t)(headparams->totalQBlkNum);
        this->tiling_head_size = (uint32_t)(headparams->headSize);
        this->tiling_para_size = (uint32_t)(headparams->paraSize);
        this->window_size = (uint32_t)(headparams->windowSize);
        this->block_dim = (uint32_t)(headparams->blockDim);
        this->long_seq = (uint32_t)(0);
        
        this->data_shape_type = 0;
        this->stride_qo = q_heads * embdv;

        uint64_t dataLenFloat = sizeof(float);
        uint64_t sSize = static_cast<uint64_t>(block_dim) * static_cast<uint64_t>(32768) * 16 * dataLenFloat;
        // kernelInfo.GetScratchSizes() = {sSize, sSize, sSize, sSize * 2}; // oTmp/S/P

        __gm__ float * o_tmp_gm = reinterpret_cast<__gm__ float *>(workspace);
        __gm__ QKV_DT * s_out_gm = reinterpret_cast<__gm__ QKV_DT *>(workspace + sSize);
        __gm__ QKV_DT * p_out_gm = reinterpret_cast<__gm__ QKV_DT *>(workspace + sSize * 2);
        o_tmp_gm_tensor.SetGlobalBuffer(o_tmp_gm, sSize);
        s_gm_tensor.SetGlobalBuffer(s_out_gm, sSize);
        p_gm_tensor.SetGlobalBuffer(p_out_gm, sSize);

        if constexpr(MASK_TYPE == 1) {
            this->is_triu_mask = 1;
        }

        if (this->data_shape_type == 1) {
            this->stride_qo = embdv;
        }

        this->__k = embd;
        this->round_k = (__k + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        SET_FLAG(MTE3, MTE2, EVENT_ID0);
        SET_FLAG(MTE3, MTE2, EVENT_ID1);
        SET_FLAG(MTE3, MTE2, EVENT_ID2);
        SET_FLAG(V, MTE2, EVENT_ID0);
        SET_FLAG(V, MTE2, EVENT_ID1);
        SET_FLAG(V, MTE2, EVENT_ID2);
        SET_FLAG(V, MTE2, EVENT_ID6);
        SET_FLAG(V, MTE2, EVENT_ID7);
        SET_FLAG(MTE3, V, EVENT_ID0);
    }
    __aicore__ __attribute__((always_inline)) inline ~MLAPrefillOp_V()
    {
        WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
        WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
        WAIT_FLAG(MTE3, MTE2, EVENT_ID2);
        WAIT_FLAG(V, MTE2, EVENT_ID0);
        WAIT_FLAG(V, MTE2, EVENT_ID1);
        WAIT_FLAG(V, MTE2, EVENT_ID2);
        WAIT_FLAG(V, MTE2, EVENT_ID6);
        WAIT_FLAG(V, MTE2, EVENT_ID7);
        WAIT_FLAG(MTE3, V, EVENT_ID0);
        PIPE_BARRIER(ALL);
    }

    template<typename Dtype>
    __aicore__ __attribute__((always_inline)) inline uint32_t VectorSize()
    {
        return 256 / sizeof(Dtype);
    }

    template<typename Dtype>
    __aicore__ __attribute__((always_inline)) inline uint64_t NumVectorsRoundUp(uint64_t num)
    {
        return (num + VectorSize<Dtype>() - 1) / VectorSize<Dtype>();
    }

    __aicore__ __attribute__((always_inline)) inline void Run()
    {

        __gm__ const MLAPrefillTiling* __restrict prefillTiling = &(tiling_gm->prefillTiling);
        uint64_t cur_batch = 0;
        uint32_t pre_total_q_blk_num = 0;
        uint32_t offset_tiling = tiling_head_size + tiling_para_size * cur_batch;
        uint32_t cur_total_q_blk_num = (uint32_t)(prefillTiling->totalQBlkNum[cur_batch]);
    
        uint32_t process_num = total_q_blk_num * q_heads;
        float alibi_coeff = 1;

        uint32_t next_process = 0;
        bool swa_flag = (window_size > 0 && MASK_TYPE == 2);

        for (uint32_t process = block_idx; process < process_num; process = next_process) {
            while (process >= cur_total_q_blk_num * q_heads) {
                cur_batch++;
                pre_total_q_blk_num = cur_total_q_blk_num;
                offset_tiling += tiling_para_size;
                cur_total_q_blk_num = (uint32_t)(prefillTiling->totalQBlkNum[cur_batch]);
            }
            next_process = process + block_num;
            if (is_triu_mask) {
                uint32_t curr_iter = process / block_num;
                next_process = curr_iter % 2 == 1 ? (curr_iter + 1) * block_num + block_idx : (curr_iter + 2) * block_num - 1 - block_idx;
            }

            // get tiling args
            uint32_t q_seqlen = (uint32_t)(prefillTiling->qSeqLen[cur_batch]);
            uint32_t kv_seqlen = (uint32_t)(prefillTiling->kvSeqlen[cur_batch]);
            if (q_seqlen == 0 || kv_seqlen == 0) {
                continue;
            }
            uint32_t pp_m_scalar = (uint32_t)(prefillTiling->mUbd[cur_batch]);
            uint32_t pp_n_scalar = (uint32_t)(prefillTiling->indexNN[cur_batch]);
            uint32_t addr_o_high32 = (uint32_t)(prefillTiling->oSeqHigh[cur_batch]);
            uint32_t addr_o_loww32 = (uint32_t)(prefillTiling->oSeqLow[cur_batch]);
            uint64_t addr_o_scalar = (uint64_t)(((uint64_t)addr_o_high32) << 32 | addr_o_loww32);

            uint32_t process_idx = process - pre_total_q_blk_num * q_heads;
            uint32_t m_idx = process_idx / q_heads;
            uint64_t head_idx = process_idx % q_heads;

            uint32_t m_loop = (q_seqlen + pp_m_scalar - 1) / pp_m_scalar;
            uint32_t n_loop = (kv_seqlen + pp_n_scalar - 1) / pp_n_scalar;

            uint32_t qk_m = (m_idx == (m_loop - 1)) ? (q_seqlen - m_idx * pp_m_scalar) : pp_m_scalar;
            uint32_t sub_m = (sub_block_idx == 1) ? (qk_m - qk_m / 2) : qk_m / 2;
            uint32_t sub_m_d128 = (sub_m + VECTOR_SIZE - 1) / VECTOR_SIZE;            // up aligned to 128
            uint32_t sub_m_d64 = (sub_m + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE; // up aligned to 64
            uint32_t round_sub_m = (sub_m + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

            /******** pre_load *******/
            uint32_t qk_n = n_loop == 1 ? kv_seqlen : pp_n_scalar;
            uint32_t qk_round_n = (qk_n + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

            uint64_t mask_batch_offset = cur_batch * mask_stride * max_seqlen;
            uint64_t mask_head_offset = head_idx * head_stride * max_seqlen;
            uint64_t mask_offset = mask_batch_offset + mask_head_offset;
            uint32_t delta_uint = 0;
            float base_y = -128;
            float delta = 0;

            uint64_t o_offset = addr_o_scalar + head_idx * embdv + (uint64_t)m_idx * pp_m_scalar * stride_qo;
            if (data_shape_type == 1) {
                o_offset = addr_o_scalar + head_idx * embdv * max_seqlen + (uint64_t)m_idx * pp_m_scalar * stride_qo;
            }

            uint32_t n_end = n_loop;
            if (is_triu_mask || window_size > 0) {
                n_end = m_idx + 1;
            }
            uint32_t window_start = (window_size + pp_n_scalar - 1) / pp_n_scalar;
            uint32_t n_start = 0;
            if (swa_flag) {
                if (window_size > 0 && window_size < kv_seqlen){
                    n_start = (m_idx < window_start) ? 0 : m_idx - window_start;
                    mask_offset += n_start * pp_n_scalar;
                }
            }
            uint32_t qk_n_triu = n_end * pp_n_scalar;
            uint32_t pv_stage = 3;
            uint32_t launch_delay = S_BLOCK_STACK * 2;
            uint32_t vect_mod = 2 * launch_delay;
            uint32_t m_slice = FLOAT_VECTOR_SIZE / S_BLOCK_STACK;
            uint32_t m_end = (sub_m + m_slice - 1) / m_slice;
            if (long_seq == 0) {
                mask_offset += m_idx * pp_m_scalar * max_seqlen;
            }
            for (uint32_t n_idx = n_start; n_idx < n_end + launch_delay; n_idx += S_BLOCK_STACK) {
                if (n_idx < n_end) {
                    if (sub_m == 0) {
                        WaitFlagDev(QK_READY);
                        FftsCrossCoreSync<PIPE_MTE3, 2>(SOFTMAX_READY);
                    } else {
                    uint32_t p_scale_offset = n_idx / S_BLOCK_STACK % pv_stage * pp_m_scalar;
                    if (n_idx + S_BLOCK_STACK > n_end - 1) {
                        qk_n = qk_n_triu > kv_seqlen ? kv_seqlen - n_idx * pp_n_scalar : qk_n_triu - n_idx * pp_n_scalar;
                    } else {
                        qk_n = pp_n_scalar * S_BLOCK_STACK;
                    }
                    qk_round_n = (qk_n + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

                    uint32_t pingpong_flag = 0;
                    bool move_flag = 0;
                    if (m_idx - n_idx >= 0 && m_idx - n_idx < 4) {
                        move_flag = 1;
                    }
                    for (uint32_t m_ind = 0; m_ind < m_end; m_ind++) {
                        uint32_t curr_m = m_ind == m_end - 1 ? sub_m - m_ind * m_slice : m_slice;
                        uint32_t s_ub_offset = pingpong_flag * S_DB_SIZE;
                        uint32_t row_offset = m_ind * m_slice;
                        if constexpr(MASK_TYPE == 1) {
                            if (move_flag) {
                                uint64_t cmp_mask_offset = (m_idx - n_idx) * 512 * 128;
                                WAIT_FLAG(V, MTE2, EVENT_ID6);
                                // 搬16 * 512, 压缩mask 512,512
                                gm_to_ub_align<ArchType::ASCEND_V220, IN_DATA_TYPE>(
                                    mask_ubuf_tensor,
                                    mask_gm_tensor[cmp_mask_offset + (sub_block_idx * qk_m / 2 + m_ind * m_slice) * 512],
                                    0,                       // sid
                                    curr_m,                   // nBurst
                                    qk_n * 2,                // lenBurst
                                    0,                       // leftPaddingNum
                                    0,                       // rightPaddingNum
                                    (512 - qk_n) * 2, // srcGap
                                    0                        // dstGap
                                );
                                SET_FLAG(MTE2, V, EVENT_ID6);
                            }
                        } else {
                            if constexpr(MASK_TYPE == 2) {
                                WAIT_FLAG(V, MTE2, EVENT_ID6);
                                gm_to_ub_align<ArchType::ASCEND_V220, IN_DATA_TYPE>(
                                    mask_ubuf_tensor,
                                    mask_gm_tensor[mask_offset + (sub_block_idx * qk_m / 2 + m_ind * m_slice) * max_seqlen],
                                    0,                       // sid
                                    curr_m,                   // nBurst
                                    qk_n * 2,                // lenBurst
                                    0,                       // leftPaddingNum
                                    0,                       // rightPaddingNum
                                    (max_seqlen - qk_n) * 2, // srcGap
                                    0                        // dstGap
                                );
                                SET_FLAG(MTE2, V, EVENT_ID6);
                            }
                        }
                        if (m_ind == 0) {
                            WaitFlagDev(QK_READY);
                        }
                        if (curr_m == 0) {
                            continue;
                        }
                        OnlineSoftmaxStage1FP16<half, float, QKV_DT, IN_DATA_TYPE, MASK_TYPE> (
                            ls_ubuf_tensor[pingpong_flag * S_DB_SIZE],
                            mask_ubuf_tensor,
                            lm_ubuf_tensor[m_ind * m_slice],
                            hm_ubuf_tensor[m_ind * m_slice],
                            gm_ubuf_tensor[m_ind * m_slice],
                            dm16_ubuf_tensor[(((n_idx - n_start) / S_BLOCK_STACK) % 4) * UB_FLOAT_LINE_SIZE + m_ind * m_slice],
                            dm_ubuf_tensor[(((n_idx - n_start) / S_BLOCK_STACK) % 4) * UB_FLOAT_LINE_SIZE + m_ind * m_slice],
                            ls32_ubuf_tensor,
                            ll_ubuf_tensor[m_ind * m_slice],
                            gl_ubuf_tensor[m_ind * m_slice],
                            lp_ubuf_tensor[pingpong_flag * S_DB_SIZE].ReinterpretCast<QKV_DT>(),
                            tv_ubuf_tensor,
                            tv16_ubuf_tensor,
                            s_gm_tensor[(uint64_t)block_idx * TMP_SIZE + (n_idx - n_start) % vect_mod * TMP_SIZE / vect_mod +
                                    (uint64_t)sub_block_idx * qk_m / 2 * qk_round_n + m_ind * m_slice * qk_round_n],
                            p_gm_tensor[((uint64_t)block_idx * TMP_SIZE + (n_idx - n_start) % vect_mod * TMP_SIZE / vect_mod +
                                            ((uint64_t)sub_block_idx * qk_m / 2 + m_ind * m_slice) * qk_round_n) * 2 / sizeof(QKV_DT)],
                            n_idx == n_start, this->tor,
                            curr_m, qk_n, qk_round_n, pingpong_flag, move_flag
                        );
                        pingpong_flag = 1 - pingpong_flag;
                    }
                    FftsCrossCoreSync<PIPE_MTE3, 2>(SOFTMAX_READY);
                    mask_offset += qk_n;
                    }
                }
                if (n_idx >= launch_delay + n_start) {
                    uint32_t qk_round_k = 128;
                    if (sub_m == 0) {
                        WaitFlagDev(UPDATE_READY); // 4
                        continue;
                    }
                    uint32_t p_scale_offset = (n_idx - launch_delay) / S_BLOCK_STACK % pv_stage * pp_m_scalar;
                    uint32_t sub_km = 64;
                    uint32_t loop_m = (sub_m + sub_km - 1) / sub_km;
                    if (n_idx != launch_delay + n_start) {
                        // *** dm_block = expand_to_block(dm), 存放于 tv
                        brcb_v<ArchType::ASCEND_V220, uint32_t>(
                            tv_ubuf_tensor.ReinterpretCast<uint32_t>(),
                            dm_ubuf_tensor[((n_idx - launch_delay - n_start) / S_BLOCK_STACK % 4) * UB_FLOAT_LINE_SIZE].ReinterpretCast<uint32_t>(),
                            1,                             // dstBlockStride
                            8,                             // dstRepeatStride
                            round_sub_m / FLOAT_BLOCK_SIZE // repeat
                        );
                        PIPE_BARRIER(V);
                        // *** go = go * dm_block
                        for (uint32_t vmul_idx = 0; vmul_idx < qk_round_k / FLOAT_VECTOR_SIZE; ++vmul_idx) {
                            mul_v<ArchType::ASCEND_V220, float>(
                                go_ubuf_tensor[vmul_idx * FLOAT_VECTOR_SIZE],
                                go_ubuf_tensor[vmul_idx * FLOAT_VECTOR_SIZE],
                                tv_ubuf_tensor,
                                sub_m,                      // repeat
                                1,                          // dstBlockStride
                                1,                          // src0BlockStride
                                0,                          // src1BlockStride
                                qk_round_k / FLOAT_BLOCK_SIZE, // dstRepeatStride
                                qk_round_k / FLOAT_BLOCK_SIZE, // src0RepeatStride
                                1                           // src1RepeatStride
                            );
                        }
                        if (qk_round_k % FLOAT_VECTOR_SIZE > 0) {
                            __set_mask(qk_round_k % FLOAT_VECTOR_SIZE);
                            mul_v<ArchType::ASCEND_V220, float>(
                                go_ubuf_tensor[qk_round_k / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                go_ubuf_tensor[qk_round_k / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                tv_ubuf_tensor,
                                sub_m,                      // repeat
                                1,                          // dstBlockStride
                                1,                          // src0BlockStride
                                0,                          // src1BlockStride
                                qk_round_k / FLOAT_BLOCK_SIZE, // dstRepeatStride
                                qk_round_k / FLOAT_BLOCK_SIZE, // src0RepeatStride
                                1                           // src1RepeatStride
                            );
                            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                        }
                        PIPE_BARRIER(V);
                    }
                    WaitFlagDev(UPDATE_READY); // 4
                    // *** 更新 L 和 O
                    if (n_idx != launch_delay + n_start) {
                        for (uint32_t ms_idx = 0; ms_idx < loop_m; ms_idx++){
                            uint32_t nowm = (ms_idx == (loop_m - 1))? sub_m - ms_idx * sub_km : sub_km;
                            uint32_t now_roundm = (nowm + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
                            uint32_t resm = ms_idx * sub_km;
                            WAIT_FLAG(V, MTE2, EVENT_ID2);
                            gm_to_ub<ArchType::ASCEND_V220, float>(
                                lo_ubuf_tensor,
                                o_tmp_gm_tensor[(uint64_t)block_idx * TMP_SIZE + (n_idx - launch_delay - n_start) % vect_mod * TMP_SIZE / vect_mod +
                                                (uint64_t)sub_block_idx * qk_m / 2 * qk_round_k + resm * qk_round_k],
                                0,                                  // sid
                                1,                                  // nBurst
                                nowm * qk_round_k / FLOAT_BLOCK_SIZE, // lenBurst
                                0,                                  // srcGap
                                0                                   // dstGap
                            );
                            SET_FLAG(MTE2, V, EVENT_ID2);
                            // *** go = lo + go
                            WAIT_FLAG(MTE2, V, EVENT_ID2);
                            add_v<ArchType::ASCEND_V220, float>(go_ubuf_tensor[resm * qk_round_k], go_ubuf_tensor[resm * qk_round_k], lo_ubuf_tensor,
                                (nowm * qk_round_k + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE, // repeat
                                1,                                                             // dstBlockStride
                                1,                                                             // src0BlockStride
                                1,                                                             // src1BlockStride
                                8,                                                             // dstRepeatStride
                                8,                                                             // src0RepeatStride
                                8                                                              // src1RepeatStride
                            );
                            PIPE_BARRIER(V);
                            SET_FLAG(V, MTE2, EVENT_ID2);
                        }
                    } else {
                        WAIT_FLAG(MTE3, MTE2, EVENT_ID2);
                        gm_to_ub<ArchType::ASCEND_V220, float>(
                            go_ubuf_tensor,
                            o_tmp_gm_tensor[(uint64_t)block_idx * TMP_SIZE + (n_idx - launch_delay - n_start) % vect_mod * TMP_SIZE / vect_mod +
                                (uint64_t)sub_block_idx * qk_m / 2 * qk_round_k],
                            0,                                  // sid
                            1,                                  // nBurst
                            sub_m * qk_round_k / FLOAT_BLOCK_SIZE, // lenBurst
                            0,                                  // srcGap
                            0                                   // dstGap
                        );
                        SET_FLAG(MTE2, V, EVENT_ID3);
                        WAIT_FLAG(MTE2, V, EVENT_ID3);
                    }
                    if (n_idx + S_BLOCK_STACK > n_end + launch_delay - 1)  {
                        // *** gl_block = expand_to_block(gl), 存放于 tv
                        brcb_v<ArchType::ASCEND_V220, uint32_t>(
                            tv_ubuf_tensor.ReinterpretCast<uint32_t>(),
                            gl_ubuf_tensor.ReinterpretCast<uint32_t>(),
                            1,                             // dstBlockStride
                            8,                             // dstRepeatStride
                            round_sub_m / FLOAT_BLOCK_SIZE // repeat
                        );
                        PIPE_BARRIER(V);
                        // *** go = go / gl_block
                        for (uint32_t vdiv_idx = 0; vdiv_idx < qk_round_k / FLOAT_VECTOR_SIZE; ++vdiv_idx) {
                            div_v<ArchType::ASCEND_V220, float>(
                                go_ubuf_tensor[vdiv_idx * FLOAT_VECTOR_SIZE],
                                go_ubuf_tensor[vdiv_idx * FLOAT_VECTOR_SIZE],
                                tv_ubuf_tensor,
                                sub_m,                      // repeat
                                1,                          // dstBlockStride
                                1,                          // src0BlockStride
                                0,                          // src1BlockStride
                                qk_round_k / FLOAT_BLOCK_SIZE, // dstRepeatStride
                                qk_round_k / FLOAT_BLOCK_SIZE, // src0RepeatStride
                                1                           // src1RepeatStride
                            );
                        }
                        if (qk_round_k % FLOAT_VECTOR_SIZE > 0) {
                            __set_mask(qk_round_k % FLOAT_VECTOR_SIZE);
                            div_v<ArchType::ASCEND_V220, float>(
                                go_ubuf_tensor[qk_round_k / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                go_ubuf_tensor[qk_round_k / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                tv_ubuf_tensor,
                                sub_m,                      // repeat
                                1,                          // dstBlockStride
                                1,                          // src0BlockStride
                                0,                          // src1BlockStride
                                qk_round_k / FLOAT_BLOCK_SIZE, // dstRepeatStride
                                qk_round_k / FLOAT_BLOCK_SIZE, // src0RepeatStride
                                1                           // src1RepeatStride
                            );
                            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                        }
                        PIPE_BARRIER(V);
                        for (uint32_t ms_idx = 0; ms_idx < loop_m; ms_idx++){
                            uint32_t nowm = (ms_idx == (loop_m - 1))? sub_m - ms_idx * sub_km : sub_km;
                            uint32_t now_roundm = (nowm + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
                            uint32_t resm = ms_idx * sub_km;
                            conv_v<ArchType::ASCEND_V220, float, IN_DATA_TYPE>(
                                go_ubuf_tensor.ReinterpretCast<IN_DATA_TYPE>()[resm * qk_round_k],
                                go_ubuf_tensor[resm * qk_round_k],
                                (nowm * qk_round_k + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE, // repeat
                                1, // dstBlockStride
                                1, // srcBlockStride
                                4, // dstRepeatStride
                                8  // srcRepeatStride
                            );
                        }
                        PIPE_BARRIER(V);
                        // ********************* move O to GM ************************
                        SET_FLAG(V, MTE3, EVENT_ID0);
                        WAIT_FLAG(V, MTE3, EVENT_ID0);
                        ub_to_gm_align<ArchType::ASCEND_V220, IN_DATA_TYPE>(
                            o_gm_tensor[o_offset + (uint64_t)sub_block_idx * qk_m / 2 * stride_qo],
                            go_ubuf_tensor.ReinterpretCast<IN_DATA_TYPE>(),
                            0,                    // sid
                            sub_m,                // nBurst
                            qk_round_k * 2,              // lenBurst
                            0,                    // leftPaddingNum
                            0,                    // rightPaddingNum
                            0,                    // srcGap
                            (stride_qo - qk_round_k) * 2 // dstGap
                        );
                        SET_FLAG(MTE3, MTE2, EVENT_ID2);
                    }
                }
            }
        }
    }

private:
    __gm__ uint8_t *__restrict__ mask_gm{nullptr};
    __gm__ uint8_t *__restrict__ o_gm{nullptr};
    __gm__ uint8_t *__restrict__ s_gm{nullptr};
    __gm__ uint8_t *__restrict__ p_gm{nullptr};
    __gm__ uint8_t *__restrict__ o_tmp_gm{nullptr};
    __gm__ MLAPrefillTilingData* __restrict__ tiling_gm{nullptr};

    const uint32_t ls_ubuf_offset = 0;
    const uint32_t lp_ubuf_offset = 0;
    const uint32_t ls32_ubuf_offset = 2 * UB_UINT8_BLOCK_SIZE;
    const uint32_t mask_ubuf_offset = 4 * UB_UINT8_BLOCK_SIZE;
    const uint32_t go_ubuf_offset = 6 * UB_UINT8_BLOCK_SIZE;
    const uint32_t lm_ubuf_offset = 8 * UB_UINT8_BLOCK_SIZE;
    const uint32_t hm_ubuf_offset = 8 * UB_UINT8_BLOCK_SIZE + 1 * UB_UINT8_LINE_SIZE;
    const uint32_t gm_ubuf_offset = 8 * UB_UINT8_BLOCK_SIZE + 2 * UB_UINT8_LINE_SIZE;
    const uint32_t dm_ubuf_offset = 8 * UB_UINT8_BLOCK_SIZE + 3 * UB_UINT8_LINE_SIZE;
    const uint32_t ll_ubuf_offset = 8 * UB_UINT8_BLOCK_SIZE + 7 * UB_UINT8_LINE_SIZE;
    const uint32_t gl_ubuf_offset = 8 * UB_UINT8_BLOCK_SIZE + 8 * UB_UINT8_LINE_SIZE;
    const uint32_t tv_ubuf_offset = 8 * UB_UINT8_BLOCK_SIZE + 9 * UB_UINT8_LINE_SIZE;
    const uint32_t p_scale_ubuf_offset = 8 * UB_UINT8_BLOCK_SIZE + 21 * UB_UINT8_LINE_SIZE;
    const uint32_t dm16_ubuf_offset = 8 * UB_UINT8_BLOCK_SIZE + 21 * UB_UINT8_LINE_SIZE;
    const uint32_t log_ubuf_offset = 8 * UB_UINT8_BLOCK_SIZE + 30 * UB_UINT8_LINE_SIZE;
    const uint32_t log_ubuf_float_offset = 8 * UB_UINT8_BLOCK_SIZE + 30 * UB_UINT8_LINE_SIZE;
    const uint32_t lo_ubuf_offset = 9 * UB_UINT8_BLOCK_SIZE;
    const uint32_t mask16_ubuf_offset = 11 * UB_UINT8_BLOCK_SIZE;

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<half> ls_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(ls_ubuf_offset);
    AscendC::LocalTensor<half> lp_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(lp_ubuf_offset);
    AscendC::LocalTensor<float> ls32_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(ls32_ubuf_offset);
    AscendC::LocalTensor<half> mask_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(mask_ubuf_offset);
    AscendC::LocalTensor<float> lo_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(lo_ubuf_offset);
    AscendC::LocalTensor<half> lm_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(lm_ubuf_offset);
    AscendC::LocalTensor<half> hm_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(hm_ubuf_offset);
    AscendC::LocalTensor<half> gm_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(gm_ubuf_offset);
    AscendC::LocalTensor<float> dm_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(dm_ubuf_offset);
    AscendC::LocalTensor<float> ll_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(ll_ubuf_offset);
    AscendC::LocalTensor<float> gl_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(gl_ubuf_offset);
    AscendC::LocalTensor<float> tv_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(tv_ubuf_offset);
    AscendC::LocalTensor<half> tv16_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(tv_ubuf_offset);
    AscendC::LocalTensor<half> dm16_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(dm16_ubuf_offset);
    AscendC::LocalTensor<float> go_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(go_ubuf_offset);
    AscendC::LocalTensor<IN_DATA_TYPE> mask16_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, IN_DATA_TYPE>(mask16_ubuf_offset);

    AscendC::LocalTensor<IN_DATA_TYPE> log_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, IN_DATA_TYPE>(log_ubuf_offset);
    AscendC::LocalTensor<float> log_ubuf_float_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(log_ubuf_float_offset);


    AscendC::GlobalTensor<IN_DATA_TYPE> mask_gm_tensor;
    AscendC::GlobalTensor<QKV_DT> o_gm_tensor;
    AscendC::GlobalTensor<QKV_DT> s_gm_tensor;
    AscendC::GlobalTensor<QKV_DT> p_gm_tensor;
    AscendC::GlobalTensor<float> o_tmp_gm_tensor;
    uint32_t batch_size{0};
    uint32_t max_seqlen{0};
    uint32_t q_heads{0};
    uint32_t embd{0};
    half tor{0};
    uint32_t head_stride{0};
    uint32_t mask_stride{0};
    uint32_t is_triu_mask{0};
    uint32_t total_q_blk_num{0};
    uint32_t isClamp{0};
    float clampMin;
    float clampMax;
    uint64_t stride_qo{0};
    uint32_t __k{0};
    uint32_t embdv{0};
    uint32_t round_k{0};
    uint32_t go_flag_scalar{1};

    int32_t sub_block_idx{-1};
    uint32_t tiling_head_size{0};
    uint32_t tiling_para_size{0};
    uint32_t long_seq{0};
    uint32_t is_sqrt{0};
    uint32_t mask_type{0};
    uint32_t alibi_compress_offset{0};
    uint32_t alibi_left_align{0};
    uint32_t data_shape_type{0};
    uint32_t quantType{0};
    uint32_t window_size{0};
    uint32_t block_dim{0};
};
 
 extern "C" __global__ __aicore__ void mla_prefill(
    __gm__ uint8_t *__restrict__ query,
    __gm__ uint8_t *__restrict__ qRope,
    __gm__ uint8_t *__restrict__ key,
    __gm__ uint8_t *__restrict__ kRope,
    __gm__ uint8_t *__restrict__ value,
    __gm__ uint8_t *__restrict__ qSeqLen,
    __gm__ uint8_t *__restrict__ kvSeqLen,
    __gm__ uint8_t *__restrict__ mask,
    __gm__ uint8_t *__restrict__ attenOut,
    __gm__ uint8_t *__restrict__ workspace,
    __gm__ uint8_t *__restrict__ tiling)
 {
    GET_TILING_DATA_WITH_STRUCT(MLAPrefillTilingData, tiling_data_in, tiling);

    __gm__ const MLAPrefillTilingData* __restrict tiling_data = reinterpret_cast<__gm__ const MLAPrefillTilingData*>(tiling);

    __gm__ uint8_t* user = AscendC::GetUserWorkspace(workspace);

     if (TILING_KEY_IS(1)) {
         MLAPrefillOp_V<half, half, half, 0> fa_vec{};
         fa_vec.Init(mask, attenOut, user, tiling_data);
         fa_vec.Run();
     } else if (TILING_KEY_IS(2)) {
         MLAPrefillOp_V<half, half, half, 1> fa_vec{};
         fa_vec.Init(mask, attenOut, user, tiling_data);
         fa_vec.Run();
     } else if (TILING_KEY_IS(3)) {
         MLAPrefillOp_V<half, half, half, 2> fa_vec{};
         fa_vec.Init(mask, attenOut, user, tiling_data);
         fa_vec.Run();
     }
 }
 
  #endif