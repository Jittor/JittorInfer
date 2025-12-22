#include "jittor_infer_flash_attention_base.h"
#include "jittor_infer_flash_attention_s1s2_bns1_x910.h"
#include "kernel_operator.h"

#define INVOKE_PFA_GENERAL_OP_IMPL(templateClass, ...)                     \
    TPipe tPipe;                                                           \
    do {                                                                   \
        if (query == nullptr) {                                            \
            return;                                                        \
        }                                                                  \
        INVOKE_PFA_TILING_DATA(tiling);                                    \
        templateClass<__VA_ARGS__> op;                                     \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, \
                          op.bmm2, bmm2tiling);                            \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengths,  \
                actualSeqLengthsKV, blocktable, attentionOut, softmaxLse,  \
                user, tiling_data, tiling, &tPipe);                        \
        op.Process();                                                      \
    } while (0)

#ifdef __DAV_C220_CUBE__
#define INVOKE_PFA_TILING_DATA(tiling)                                  \
    GET_TILING_DATA_MEMBER(JittorInferFlashAttentionTilingData,         \
                           bmm1TilingDataRect, bmm1TilingData, tiling); \
    GET_TILING_DATA_MEMBER(JittorInferFlashAttentionTilingData,         \
                           bmm2TilingDataRect, bmm2TilingData, tiling); \
    const TCubeTiling* __restrict bmm1tiling = &bmm1TilingData;         \
    const TCubeTiling* __restrict bmm2tiling = &bmm2TilingData;         \
    const JittorInferFlashAttentionTilingData* __restrict tiling_data = nullptr
#else
#define INVOKE_PFA_TILING_DATA(tiling)                                  \
    GET_TILING_DATA_WITH_STRUCT(JittorInferFlashAttentionTilingData,    \
                                tiling_data_in, tiling);                \
    const JittorInferFlashAttentionTilingData* __restrict tiling_data = \
        &tiling_data_in;                                                \
    const TCubeTiling* __restrict bmm1tiling =                          \
        &(tiling_data->bmm1TilingDataRect);                             \
    const TCubeTiling* __restrict bmm2tiling =                          \
        &(tiling_data->bmm2TilingDataRect)
#endif
constexpr uint32_t FLOATBYTENUM = 8;
constexpr uint32_t FLOAT16BYTENUM = 16;
constexpr uint32_t INT8BYTENUM = 32;

extern "C" __global__ __aicore__ void jittor_infer_flash_attention_FIAS(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
    __gm__ uint8_t* pseShift, __gm__ uint8_t* attenMask,
    __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKV,
    __gm__ uint8_t* deq_scale1, __gm__ uint8_t* quant_scale1,
    __gm__ uint8_t* deq_scale2, __gm__ uint8_t* quant_scale2,
    __gm__ uint8_t* quant_offset2, __gm__ uint8_t* blocktable,
    __gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse,
    __gm__ uint8_t* workspace, __gm__ uint8_t* tiling) {
    GET_TILING_DATA_MEMBER(JittorInferFlashAttentionTilingData,
                           promptAttentionBaseParams, baseParams, tiling);
    auto maskByteNum = baseParams.maskTypeByteNum;

    __gm__ uint8_t* user = GetUserWorkspace(workspace);

    if (TILING_KEY_IS(1000000000000101612)) {
        // BSH layout HighPrecision
        INVOKE_PFA_GENERAL_OP_IMPL(JittorInferFlashAttentionS1s2Bns1X910,
                                   PFAType<PFALayout::BSH, half, bool, half,
                                           half, Mode::HighPrecision>);
    } else if (TILING_KEY_IS(1000000000002101612)) {
        // BSH layout HighPrecision, enable L1 reuse
        INVOKE_PFA_GENERAL_OP_IMPL(
            JittorInferFlashAttentionS1s2Bns1X910,
            PFAType<PFALayout::BSH, half, bool, half, half, Mode::HighPrecision,
                    MatMulType::MM_IBSHARE_NORM>);
    } else if (TILING_KEY_IS(1000000000000001612)) {
        // BNSD layout HighPrecision
        INVOKE_PFA_GENERAL_OP_IMPL(JittorInferFlashAttentionS1s2Bns1X910,
                                   PFAType<PFALayout::BNSD, half, bool, half,
                                           half, Mode::HighPrecision>);
    } else if (TILING_KEY_IS(1000000000001001612)) {
        // BNSD layout HighPrecision
        INVOKE_PFA_GENERAL_OP_IMPL(
            JittorInferFlashAttentionS1s2Bns1X910,
            PFAType<PFALayout::BNSD, half, bool, half, half,
                    Mode::HighPrecision, MatMulType::MM_NORM>);
    } else if (TILING_KEY_IS(1000000000002001612)) {
        // BNSD layout HighPrecision
        INVOKE_PFA_GENERAL_OP_IMPL(
            JittorInferFlashAttentionS1s2Bns1X910,
            PFAType<PFALayout::BNSD, half, bool, half, half,
                    Mode::HighPrecision, MatMulType::MM_IBSHARE_NORM>);
    } else if (TILING_KEY_IS(1000000000002101012)) {
        // no anti-quant path for CVDIFF-BSH, half in half out, enable L1 reuse
        INVOKE_PFA_GENERAL_OP_IMPL(
            JittorInferFlashAttentionS1s2Bns1X910,
            PFAType<PFALayout::BSH, half, uint8_t, half, half,
                    Mode::HighPerformance, MatMulType::MM_IBSHARE_NORM>);
    } else if (TILING_KEY_IS(1000000000001001012)) {
        INVOKE_PFA_GENERAL_OP_IMPL(
            JittorInferFlashAttentionS1s2Bns1X910,
            PFAType<PFALayout::BNSD, half, uint8_t, half, half,
                    Mode::HighPerformance, MatMulType::MM_NORM>);
    } else if (TILING_KEY_IS(1000000000002001012)) {
        INVOKE_PFA_GENERAL_OP_IMPL(
            JittorInferFlashAttentionS1s2Bns1X910,
            PFAType<PFALayout::BNSD, half, uint8_t, half, half,
                    Mode::HighPerformance, MatMulType::MM_IBSHARE_NORM>);
    }
}

extern "C" __global__ __aicore__ void jittor_infer_flash_attention(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
    __gm__ uint8_t* pseShift, __gm__ uint8_t* attenMask,
    __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKV,
    __gm__ uint8_t* deq_scale1, __gm__ uint8_t* quant_scale1,
    __gm__ uint8_t* deq_scale2, __gm__ uint8_t* quant_scale2,
    __gm__ uint8_t* quant_offset2, __gm__ uint8_t* attentionOut,
    __gm__ uint8_t* workspace, __gm__ uint8_t* tiling) {
    jittor_infer_flash_attention_FIAS(
        query, key, value, pseShift, attenMask, actualSeqLengths,
        actualSeqLengthsKV, deq_scale1, quant_scale1, deq_scale2, quant_scale2,
        quant_offset2, nullptr, attentionOut, nullptr, workspace, tiling);
}