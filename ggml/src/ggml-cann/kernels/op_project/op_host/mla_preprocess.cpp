
#include <cmath>
#include <string>

#include "mla_preprocess_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

constexpr size_t DIM_0 = 0;
constexpr size_t DIM_1 = 1;
constexpr size_t DIM_2 = 2;
constexpr size_t DIM_3 = 3;
constexpr size_t DIM_4 = 4;
constexpr size_t DIM_5 = 5;
constexpr size_t DIM_6 = 6;
constexpr size_t DIM_7 = 7;
constexpr size_t DIM_8 = 8;
constexpr size_t DIM_9 = 9;
constexpr size_t DIM_10 = 10;
constexpr size_t DIM_11 = 11;
constexpr size_t DIM_12 = 12;
constexpr size_t DIM_13 = 13;
constexpr size_t DIM_14 = 14;
constexpr size_t DIM_15 = 15;

constexpr uint32_t AXES_ALIGN_SIZE = 512;
constexpr uint32_t BASE_BLOCK_STEP = 2;
constexpr uint32_t CONST_16 = 16;
constexpr uint32_t CONST_32 = 32;
constexpr uint32_t CONST_128 = 128;
constexpr uint32_t CONST_256 = 256;
constexpr uint32_t CONST_512 = 512;
constexpr uint32_t L1_BUFFER_SIZE = 524288;
constexpr uint32_t L1_PINGPONG_BUFFER_LEN = 262144;
constexpr uint32_t L0AB_PINGPONG_BUFFER_LEN = 131072;
constexpr uint32_t L1_SCALE_SIZE = 4096;
constexpr uint32_t L1_BIAS_SIZE = 2048;
constexpr uint32_t L0C_SIZE = 128 * 1024;
constexpr uint32_t CONCAT_SIZE = 512;
constexpr uint32_t HIDDEN_STRATE = 7168;
constexpr uint32_t HIDDEN_STRATE_ROPE = 192;
constexpr uint32_t HIDDEN_STRATE_MM = 2112;
constexpr uint32_t HIDDEN_STRATE_RMS = 1536;
constexpr uint32_t UB_SIZE = 196352;
constexpr uint32_t HEADDIM = 64;
constexpr uint32_t FP32_REPEAT_MASK = 64;
constexpr uint32_t FP16_REPEAT_MASK = 128;

const int32_t NUM1 = 1;
const int32_t NUM2 = 2;
const int32_t NUM3 = 3;
const int32_t NUM4 = 4;
const int32_t NUM8 = 8;
const uint32_t INDEX_WDQKV = 5;
const uint32_t INDEX_WUQ = 16;
const uint32_t INDEX_WUK = 18;

enum class QuantMode : int32_t {
    PER_TENSOR_ASYMM_QUANT = 0,
    PER_TOKEN_SYMM_QUANT,
    PER_TOKEN_ASYMM_QUANT,
    NO_QUANT,
};

inline uint32_t CeilDiv(const uint32_t dividend, const uint32_t divisor) {
    if (divisor == 0) {
        return UINT32_MAX;
    }
    return (dividend + divisor - 1) / divisor;
}

inline uint32_t RoundUp(const uint32_t val, const uint32_t align = 16) {
    if (align == 0) {
        return 0;
    }
    return (val + align - 1) / align * align;
}

inline uint32_t RoundDown(const uint32_t val, const uint32_t align = 16) {
    if (align == 0) {
        return 0;
    }
    return val / align * align;
}

template <typename T = uint32_t>
inline T Max(const T a, const T b) {
    return a > b ? a : b;
}

template <typename T = uint32_t>
inline T Min(const T a, const T b) {
    return a < b ? a : b;
}

using QuantMode = QuantMode;
class PpMatmulTilingApi {
   public:
    PpMatmulTilingApi(uint32_t numBatch, uint32_t m, uint32_t k, uint32_t n,
                      bool transA, bool transB, bool enDequant,
                      bool deqOnTheFly, gert::TilingContext* context)
        : context_(context),
          numBatch_(numBatch),
          m_(m),
          k_(k),
          n_(n),
          transA_(transA),
          transB_(transB),
          enDequant_(enDequant),
          deqOnTheFly_(deqOnTheFly) {
        inDataSize_ = enDequant ? sizeof(uint8_t) : sizeof(uint16_t);
    }
    void GetTilingData(optiling::PpMatmulTilingData& tiling);

   private:
    void GetTileSize();
    float GetCost(const uint32_t m0, const uint32_t n0);
    void UpdateTileSize(const uint32_t m0, const uint32_t n0);
    void Swizzle();
    uint32_t ComputeL1AbSize();
    uint32_t ComputeK0ForABpingpong(uint32_t l1AbSize);
    bool IsLoadAllAmat(uint32_t l1AbSize);
    uint32_t ComputeK0ForOnlyBpingpong(uint32_t l1AbSize);

   private:
    gert::TilingContext* context_;
    uint32_t numBatch_{0};
    uint32_t m_{0};
    uint32_t k_{0};
    uint32_t n_{0};
    uint32_t m0_{0};
    uint32_t k0_{0};
    uint32_t n0_{0};
    uint32_t mLoop_{0};
    uint32_t kLoop_{0};
    uint32_t nLoop_{0};
    uint32_t coreLoop_{0};
    uint32_t swizzleCount_{0};
    uint32_t blockDim_{0};
    uint32_t swizzleDirect_{0};
    uint32_t inDataSize_{0};
    uint32_t b0matPingPongBufferLen_{L1_PINGPONG_BUFFER_LEN};
    bool transA_{false};
    bool transB_{false};
    bool enDequant_{false};
    bool enShuffleK_{false};
    bool enLoadAllAmat_{false};
    bool deqOnTheFly_{false};
};

void PpMatmulTilingApi::GetTilingData(optiling::PpMatmulTilingData& tiling) {
    GetTileSize();
    tiling.set_numBatch(numBatch_);
    tiling.set_m(m_);
    tiling.set_k(k_);
    tiling.set_n(n_);
    tiling.set_m0(m0_);
    tiling.set_k0(k0_);
    tiling.set_n0(n0_);
    tiling.set_mLoop(mLoop_);
    tiling.set_kLoop(kLoop_);
    tiling.set_nLoop(nLoop_);
    tiling.set_coreLoop(coreLoop_);
    tiling.set_swizzleCount(swizzleCount_);
    tiling.set_swizzleDirect(swizzleDirect_);
    tiling.set_enShuffleK(static_cast<uint32_t>(enShuffleK_));
    tiling.set_blockDim(blockDim_);
    tiling.set_enLoadAllAmat(static_cast<uint32_t>(enLoadAllAmat_));
    tiling.set_b0matPingPongBufferLen(b0matPingPongBufferLen_);
}

void PpMatmulTilingApi::GetTileSize() {
    bool priFlag = !(m_ < n_);
    uint32_t roundBase =
        pow(2, ceil(log(CeilDiv(priFlag ? n_ : m_, CONST_16)))) * CONST_16;
    uint32_t priAxes = RoundUp(priFlag ? m_ : n_, CONST_16);
    uint32_t subAxes = RoundUp(priFlag ? n_ : m_, roundBase);
    float minCost = __FLT_MAX__;
    uint32_t maxAxes0 = AXES_ALIGN_SIZE;
    uint32_t maxPriAxes0 = Min(maxAxes0, priAxes);
    uint32_t maxSubAxes0 = Min(maxAxes0, subAxes);
    auto ascendcPlatform =
        platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    uint64_t l0c_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C,
                                   l0c_size);
    for (uint32_t priAxes0 = CONST_16; priAxes0 <= maxPriAxes0;
         priAxes0 *= BASE_BLOCK_STEP) {
        for (uint32_t subAxes0 = CONST_16; subAxes0 <= maxSubAxes0;
             subAxes0 *= BASE_BLOCK_STEP) {
            // PlatformInfo::Instance().GetL0CSize() -> l0c_size
            if (priAxes0 * subAxes0 * sizeof(float) > l0c_size) {
                continue;
            }
            uint32_t newM0 = priFlag ? priAxes0 : subAxes0;
            uint32_t newN0 = priFlag ? subAxes0 : priAxes0;
            if (newN0 > CONST_256 && enDequant_) {
                continue;
            }
            float cost = GetCost(newM0, newN0);
            if (cost < minCost) {
                minCost = cost;
                UpdateTileSize(newM0, newN0);
            }
        }
    }

    Swizzle();

    uint32_t l1AbSize = ComputeL1AbSize();
    k0_ = ComputeK0ForABpingpong(l1AbSize);
    kLoop_ = CeilDiv(k_, k0_);
    // 对于MM1和MM2, 如果一个核一轮跑不完, 选择全载A, 并更新k0
    if (0) {  // IsLoadAllAmat(l1AbSize)
        k0_ = ComputeK0ForOnlyBpingpong(l1AbSize);
        kLoop_ = CeilDiv(k_, k0_);
    }
}

uint32_t PpMatmulTilingApi::ComputeK0ForOnlyBpingpong(uint32_t l1AbSize) {
    enLoadAllAmat_ = true;
    b0matPingPongBufferLen_ = static_cast<uint32_t>(static_cast<float>(
        (l1AbSize -
         RoundUp(m_, CONST_16) * RoundUp(k_, CONST_32) * inDataSize_) /
        DIM_2));
    uint32_t k0MaxB0 = static_cast<uint32_t>(static_cast<float>(
        b0matPingPongBufferLen_ / (RoundUp(n0_, CONST_16) * inDataSize_)));
    uint32_t k0B0 = k0MaxB0 < CONST_512 ? RoundDown(k0MaxB0, CONST_32)
                                        : RoundDown(k0MaxB0, CONST_512);
    return k0B0 > CONST_512 ? RoundDown(k0B0, CONST_512) : k0B0;
}

bool PpMatmulTilingApi::IsLoadAllAmat(uint32_t l1AbSize) {
    return (coreLoop_ > blockDim_) && enDequant_ && (kLoop_ > 1) &&
           (l1AbSize >
            RoundUp(m_, CONST_16) * RoundUp(k_, CONST_32) * inDataSize_) &&
           (mLoop_ == 1);
}

uint32_t PpMatmulTilingApi::ComputeK0ForABpingpong(uint32_t l1AbSize) {
    uint32_t k0Max = static_cast<uint32_t>(
        static_cast<float>(l1AbSize / DIM_2) / ((m0_ + n0_) * inDataSize_));
    uint32_t tmpK0;
    if (enDequant_) {
        tmpK0 = k0Max < CONST_512 ? RoundDown(k0Max, CONST_32)
                                  : RoundDown(k0Max, CONST_512);
    } else {
        tmpK0 = k0Max < CONST_256 ? RoundDown(k0Max, CONST_16)
                                  : RoundDown(k0Max, CONST_256);
    }
    if (tmpK0 > CONST_512) {
        tmpK0 = RoundDown(tmpK0, CONST_512);
    }
    return tmpK0;
}

uint32_t PpMatmulTilingApi::ComputeL1AbSize() {
    if (enDequant_ && deqOnTheFly_) {
        return L1_BUFFER_SIZE;
    }
    return enDequant_ ? (L1_BUFFER_SIZE - L1_BIAS_SIZE - L1_SCALE_SIZE)
                      : L1_BUFFER_SIZE;
}

float PpMatmulTilingApi::GetCost(const uint32_t m0, const uint32_t n0) {
    auto ascendcPlatform =
        platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    uint64_t l2_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2_size);
    float aCoef = 1.0;
    float bCoef = 1.0;
    float bwCoef = 5.0;
    uint32_t mLoop = CeilDiv(m_, m0);
    uint32_t nLoop = CeilDiv(n_, n0);
    if (mLoop == 0 || nLoop == 0) {
        return __FLT_MAX__;
    }
    uint32_t rqdNumCore = numBatch_ * mLoop * nLoop;
    uint32_t blockDim = Min(rqdNumCore, ascendcPlatform.GetCoreNumAic());
    uint32_t mOnce = blockDim < nLoop ? m0 : blockDim / nLoop * m0;
    uint32_t nOnce =
        blockDim < nLoop ? ascendcPlatform.GetCoreNumAic() * n0 : n_;
    // PlatformInfo::Instance().GetL2Size() -> l2_size
    if (mOnce * k_ * sizeof(uint16_t) > l2_size) {
        aCoef = bwCoef;
    }
    if (nOnce * k_ * sizeof(uint16_t) > l2_size) {
        bCoef = bwCoef;
    }
    if (transA_ && m0 % CONST_256 == 0) {
        aCoef *= NUM2;
    }
    if (!transB_ && n0 % CONST_256 == 0) {
        bCoef *= NUM2;
    }
    return 1 / (aCoef * static_cast<float>(n0)) +
           1 / (bCoef * static_cast<float>(m0));
}

void PpMatmulTilingApi::UpdateTileSize(const uint32_t m0, const uint32_t n0) {
    m0_ = m0;
    n0_ = n0;
    mLoop_ = CeilDiv(m_, m0_);
    nLoop_ = CeilDiv(n_, n0_);
    coreLoop_ = numBatch_ * mLoop_ * nLoop_;
    // const uint32_t maxNumCubeCore = ascendcPlatform.GetCoreNumAic();
    auto ascendcPlatform =
        platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    const uint32_t maxNumCubeCore = ascendcPlatform.GetCoreNumAic();
    if (mLoop_ == 1 && transB_ &&
        coreLoop_ % maxNumCubeCore < maxNumCubeCore / NUM4 * NUM3) {
        uint32_t tmpM0 = RoundUp(m_, CONST_16);
        uint32_t maxN0 = L0C_SIZE / (tmpM0 * sizeof(float));
        if (enDequant_) {
            maxN0 = maxN0 < CONST_256 ? maxN0 : CONST_256;
        }
        uint32_t x = CeilDiv(n_, maxNumCubeCore);
        uint32_t y = CeilDiv(x, maxN0);
        uint32_t tmpN0 = RoundUp(CeilDiv(x, y), CONST_16);
        uint32_t rqdL0cSize = tmpM0 * tmpN0 * sizeof(float);
        if (rqdL0cSize < L0C_SIZE &&
            (tmpM0 + tmpN0) * CONST_256 * inDataSize_ < L1_BUFFER_SIZE) {
            m0_ = tmpM0;
            n0_ = tmpN0;
            nLoop_ = CeilDiv(n_, n0_);
            coreLoop_ = numBatch_ * nLoop_;
        }
    }
    blockDim_ = Min(coreLoop_, maxNumCubeCore);
}

void PpMatmulTilingApi::Swizzle() {
    float minCost = m_ * k_ + k_ * n_;
    for (uint32_t i = 1; i <= blockDim_; ++i) {
        int c = static_cast<int32_t>((blockDim_ + i - 1) / i);
        float cost;
        // B0 + A < A0 + B
        if (i * n0_ + m_ < m0_ * c + n_) {
            swizzleDirect_ = 1;  // Nz
            cost = n0_ * i + m0_ * c;
            if (cost <= minCost) {
                minCost = cost;
                swizzleCount_ = i;
            }
        } else {
            swizzleDirect_ = 0;  // Zn
            cost = m0_ * i + n0_ * c;
            if (cost < minCost) {
                minCost = cost;
                swizzleCount_ = i;
            }
        }
    }
}

class MlaPreprocessTiling {
   public:
    optiling::MLAPreprocessTilingData tilingData;
    void Init(gert::TilingContext* context);
    void RmsNormQuantTiling(const uint32_t numTokens,
                            const uint32_t numVectorCore);
    void RopeConcatTiling(const int64_t ntokens, const int64_t headNum,
                          const uint32_t& aicNum);
    void EinSumQuantTiling(const int64_t ntokens, const int64_t headNum,
                           const int64_t quantMode, const uint32_t& aicNum,
                           const ge::DataType inDtype);
    void SetTilingKey(gert::TilingContext* context, int64_t cacheMode,
                      uint64_t quantMode);
    void SetMlapoWorkSpace(const ge::DataType inDtype, int64_t cacheMode,
                           uint64_t headNum, uint64_t quantMode,
                           gert::TilingContext* context);
};

void MlaPreprocessTiling::RmsNormQuantTiling(const uint32_t numTokens,
                                             const uint32_t numVectorCore) {
    // const uint32_t numVectorCore =
    // PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    tilingData.set_rmsNumCore1(numVectorCore);
    tilingData.set_rmsNumCol1(HIDDEN_STRATE);
    tilingData.set_rmsNumRow1(numTokens);
    tilingData.set_rmsQuantMin1(-CONST_128);
    tilingData.set_rmsNumCore2(numVectorCore);
    tilingData.set_rmsNumCol2(HIDDEN_STRATE_MM);
    tilingData.set_rmsNumRow2(numTokens);
    tilingData.set_rmsQuantMin2(-CONST_128);
}

void MlaPreprocessTiling::RopeConcatTiling(const int64_t ntokens,
                                           const int64_t headNum,
                                           const uint32_t& aicNum) {
    uint32_t hiddenSizeQ = HEADDIM * headNum;
    uint32_t headDim = HEADDIM;
    uint32_t headNumQ = hiddenSizeQ / headDim;
    uint32_t concatSize = CONCAT_SIZE;
    uint32_t maxCore = aicNum * 2;
    uint32_t maxUbSize = UB_SIZE;
    uint32_t allHeadNum = ntokens * headNumQ;

    uint32_t tempCore = (allHeadNum + maxCore - 1) / maxCore;
    uint32_t realCore = (allHeadNum + tempCore - 1) / tempCore;  // 实际运算核数
    uint32_t nlCoreRun =
        (allHeadNum + realCore - 1) / realCore;  // 前核运算head数
    uint32_t lCoreRun =
        allHeadNum - (realCore - 1) * nlCoreRun;  // 尾核运算head数

    uint32_t dataTypeSize = 2;

    // 计算一次能搬几行 q 4+2、reverseq 4、neg 4、sin 4+2、cos 4+2  + concat 2
    uint32_t allSize = headDim * (3 * (4 + dataTypeSize) + 2 * 4) +
                       concatSize * dataTypeSize;  // rope内部升精度计算
    uint32_t maxNPerLoopForUb =
        maxUbSize / allSize;  // ub每次能载入最大行数（包括所有计算数据）
    uint32_t preCoreLoopTime =
        (nlCoreRun + maxNPerLoopForUb - 1) / maxNPerLoopForUb;  // 前核循环次数
    uint32_t preCoreLoopNLast =
        nlCoreRun -
        (preCoreLoopTime - 1) * maxNPerLoopForUb;  // 前核最后一批处理数据行数
    uint32_t lastCoreLoopTime =
        (lCoreRun + maxNPerLoopForUb - 1) / maxNPerLoopForUb;  // 尾核循环次数
    uint32_t lastCoreLoopNLast =
        lCoreRun -
        (lastCoreLoopTime - 1) * maxNPerLoopForUb;  // 尾核最后一批处理数据行数

    tilingData.set_hiddenSizeQ(hiddenSizeQ);
    tilingData.set_headNumQ(headNumQ);
    tilingData.set_headDim(headDim);
    tilingData.set_concatSize(concatSize);
    tilingData.set_rotaryCoeff(NUM2);
    tilingData.set_ntokens(ntokens);
    tilingData.set_realCore(realCore);
    tilingData.set_nlCoreRun(nlCoreRun);
    tilingData.set_lCoreRun(nlCoreRun);
    tilingData.set_maxNPerLoopForUb(maxNPerLoopForUb);
    tilingData.set_preCoreLoopTime(preCoreLoopTime);
    tilingData.set_preCoreLoopNLast(preCoreLoopNLast);
    tilingData.set_lastCoreLoopTime(lastCoreLoopTime);
    tilingData.set_lastCoreLoopNLast(lastCoreLoopNLast);
}

void MlaPreprocessTiling::EinSumQuantTiling(const int64_t ntokens,
                                            const int64_t headNum,
                                            const int64_t quantMode,
                                            const uint32_t& aicNum,
                                            const ge::DataType inDtype) {
    uint32_t aivCore = aicNum * 2;
    uint32_t ubSize = UB_SIZE - 1024;

    // input shape
    uint32_t esqBatch = ntokens;           // tokenNum
    uint32_t esqHeadNum = headNum;         // headNum
    uint32_t esqColNum = AXES_ALIGN_SIZE;  // 512

    // split core
    uint32_t esqFrontCore = esqBatch % aivCore;
    uint32_t esqTailCore = aivCore - esqFrontCore;
    uint32_t esqFrontCoreBatch = CeilDiv(esqBatch, aivCore);
    uint32_t esqTailCoreBatch = esqBatch / aivCore;

    // split ub --> calc H' <-- 一次ub循环中搬运处理的行数
    uint32_t splitFactor = 0;
    uint32_t esqHeadPerLoop = 0;  // ub每次计算的head行数
    uint32_t repeatMask = 0;

    if (inDtype == ge::DT_BF16 ||
        static_cast<QuantMode>(quantMode) == QuantMode::PER_TOKEN_SYMM_QUANT) {
        // 将scale一次性搬入、广播、缓存 H * 32bytes
        uint32_t scaleUb = RoundUp(esqHeadNum) * CONST_32;
        // bf16 input [H', colNum](f16 + fp32 + int8), ub reuse
        splitFactor =
            esqColNum * (sizeof(uint16_t) + sizeof(float) + sizeof(uint8_t));
        splitFactor *= NUM2;
        esqHeadPerLoop = (ubSize - scaleUb) / splitFactor;  // 26
        repeatMask = FP32_REPEAT_MASK;
    } else {
        // fp16 input [H', cloNum](fp16*2 + int8) + [H', 1](fp16) + [H',
        // 16](fp16)
        splitFactor = esqColNum * (NUM2 * sizeof(uint16_t) + sizeof(uint8_t)) +
                      sizeof(uint16_t) + (CONST_16 * sizeof(uint16_t));
        esqHeadPerLoop = ubSize / splitFactor;
        repeatMask = FP16_REPEAT_MASK;
        esqHeadPerLoop = RoundDown(esqHeadPerLoop);  // 向下16对齐
    }
    uint32_t esqUbHeadLoop = esqHeadNum / esqHeadPerLoop;  // ub完整循环次数
    uint32_t esqHeadTail =
        esqHeadNum % esqHeadPerLoop;  // ub最后一次处理head的行数
    uint32_t esqColLoop =
        esqColNum / repeatMask;  // 每行按列计算要循环处理的次数
    uint32_t esqColTail =
        esqColNum % repeatMask;  // colNum非64/128对齐时，最后一次计算列数

    tilingData.set_esqFrontCore(esqFrontCore);
    tilingData.set_esqTailCore(esqTailCore);
    tilingData.set_esqFrontCoreBatch(esqFrontCoreBatch);
    tilingData.set_esqTailCoreBatch(esqTailCoreBatch);
    tilingData.set_esqHeadNum(esqHeadNum);
    tilingData.set_esqColNum(esqColNum);
    tilingData.set_esqUbHeadLoop(esqUbHeadLoop);
    tilingData.set_esqHeadPerLoop(esqHeadPerLoop);
    tilingData.set_esqHeadTail(esqHeadTail);
    tilingData.set_esqColLoop(esqColLoop);
    tilingData.set_esqColTail(esqColTail);
}

void MlaPreprocessTiling::SetTilingKey(gert::TilingContext* context,
                                       int64_t cacheMode, uint64_t quantMode) {
    // TODO: 确认一下是不是StorageFormat
    ge::Format formatWeight1 =
        context->GetInputTensor(INDEX_WDQKV)->GetStorageFormat();
    auto formatWeight2 = context->GetInputTensor(INDEX_WUQ)->GetStorageFormat();
    auto formatWeight3 = context->GetInputTensor(INDEX_WUK)->GetStorageFormat();
    uint64_t tilingKey = static_cast<uint64_t>(
        context->GetInputTensor(0)->GetDataType() == ge::DT_BF16);
    tilingKey = (tilingKey << 2) +
                static_cast<uint64_t>(cacheMode);  // 2bit for cacheMode.
    tilingKey = (tilingKey << 1) +
                static_cast<uint64_t>(formatWeight1 == ge::FORMAT_FRACTAL_NZ);
    tilingKey = (tilingKey << 1) +
                static_cast<uint64_t>(formatWeight2 == ge::FORMAT_FRACTAL_NZ);
    tilingKey = (tilingKey << 1) +
                static_cast<uint64_t>(formatWeight3 == ge::FORMAT_FRACTAL_NZ);
    tilingKey = (tilingKey << 2) +
                static_cast<uint64_t>(quantMode);  // 2bit for quantMode.
    context->SetTilingKey(tilingKey);
    std::cout << "tilingKey is: " << tilingKey << std::endl;
}

std::ostream& operator<<(std::ostream& os,
                         optiling::PpMatmulTilingData& tilingData) {
    os << "(bSize, mSize, kSize, nSize) = (" << tilingData.get_numBatch()
       << ", " << tilingData.get_m() << ", " << tilingData.get_k() << ", "
       << tilingData.get_n() << ")" << std::endl;

    os << "(m0, k0, n0) = (" << tilingData.get_m0() << ", "
       << tilingData.get_k0() << ", " << tilingData.get_n0() << ")"
       << std::endl;

    os << "(mLoop, kLoop, nLoop) = (" << tilingData.get_mLoop() << ", "
       << tilingData.get_kLoop() << ", " << tilingData.get_nLoop() << ")"
       << std::endl;

    os << "coreLoop = " << tilingData.get_coreLoop() << std::endl;

    os << "(SwizzleDirect, SwizzleCount) = (" << tilingData.get_swizzleDirect()
       << ", " << tilingData.get_swizzleCount() << ")" << std::endl;

    os << "blockDim = " << tilingData.get_blockDim() << std::endl;

    return os;
}

std::ostream& operator<<(std::ostream& os,
                         optiling::MLAPreprocessTilingData& tilingData) {
    os << "tiling data:" << std::endl;
    os << "numCore = " << tilingData.get_numCore() << std::endl;
    os << "n = " << tilingData.get_n() << std::endl;
    os << "perTaskNum = " << tilingData.get_perTaskNum() << std::endl;
    os << "resTaskNum = " << tilingData.get_resTaskNum() << std::endl;
    os << tilingData.mm1 << std::endl;
    os << tilingData.mm2 << std::endl;
    os << tilingData.mm3 << std::endl;
    os << "rmsNumCore1 = " << tilingData.get_rmsNumCore1() << std::endl;
    os << "rmsNumCol1 = " << tilingData.get_rmsNumCol1() << std::endl;
    os << "rmsNumRow1 = " << tilingData.get_rmsNumRow1() << std::endl;
    os << "rmsQuantMin1 = " << tilingData.get_rmsQuantMin1() << std::endl;
    os << "rmsNumCore2 = " << tilingData.get_rmsNumCore2() << std::endl;
    os << "rmsNumCol2 = " << tilingData.get_rmsNumCol2() << std::endl;
    os << "rmsNumRow2 = " << tilingData.get_rmsNumRow2() << std::endl;
    os << "rmsQuantMin2 = " << tilingData.get_rmsQuantMin2() << std::endl;
    os << "hiddenSizeQ = " << tilingData.get_hiddenSizeQ() << std::endl;
    os << "headNumQ = " << tilingData.get_headNumQ() << std::endl;
    os << "headDim = " << tilingData.get_headDim() << std::endl;
    os << "concatSize = " << tilingData.get_concatSize() << std::endl;
    os << "rotaryCoeff = " << tilingData.get_rotaryCoeff() << std::endl;
    os << "ntokens = " << tilingData.get_ntokens() << std::endl;
    os << "realCore = " << tilingData.get_realCore() << std::endl;
    os << "nlCoreRun = " << tilingData.get_nlCoreRun() << std::endl;
    os << "lCoreRun = " << tilingData.get_lCoreRun() << std::endl;
    os << "maxNPerLoopForUb = " << tilingData.get_maxNPerLoopForUb()
       << std::endl;
    os << "preCoreLoopTime = " << tilingData.get_preCoreLoopTime() << std::endl;
    os << "preCoreLoopNLast = " << tilingData.get_preCoreLoopNLast()
       << std::endl;
    os << "lastCoreLoopTime = " << tilingData.get_lastCoreLoopTime()
       << std::endl;
    os << "lastCoreLoopNLast = " << tilingData.get_lastCoreLoopNLast()
       << std::endl;
    os << "esqFrontCore      = " << tilingData.get_esqFrontCore() << std::endl;
    os << "esqTailCore       = " << tilingData.get_esqTailCore() << std::endl;
    os << "esqFrontCoreBatch = " << tilingData.get_esqFrontCoreBatch()
       << std::endl;
    os << "esqTailCoreBatch  = " << tilingData.get_esqTailCoreBatch()
       << std::endl;
    os << "esqHeadNum        = " << tilingData.get_esqHeadNum() << std::endl;
    os << "esqColNum         = " << tilingData.get_esqColNum() << std::endl;
    os << "esqUbHeadLoop     = " << tilingData.get_esqUbHeadLoop() << std::endl;
    os << "esqHeadPerLoop    = " << tilingData.get_esqHeadPerLoop()
       << std::endl;
    os << "esqHeadTail       = " << tilingData.get_esqHeadTail() << std::endl;
    os << "esqColLoop        = " << tilingData.get_esqColLoop() << std::endl;
    os << "esqColTail        = " << tilingData.get_esqColTail() << std::endl;
    return os;
}

void MlaPreprocessTiling::SetMlapoWorkSpace(const ge::DataType inDtype,
                                            int64_t cacheMode, uint64_t headNum,
                                            uint64_t quantMode,
                                            gert::TilingContext* context) {
    uint64_t s1wsFactor = static_cast<uint64_t>(
        cacheMode == 2 ? std::max(HIDDEN_STRATE * sizeof(int8_t),
                                  headNum * AXES_ALIGN_SIZE * sizeof(uint16_t))
                       : HIDDEN_STRATE * sizeof(int8_t));
    uint64_t workSizeS1 =
        static_cast<uint64_t>(tilingData.get_n()) * s1wsFactor;
    uint64_t workSizeS2 = static_cast<uint64_t>(tilingData.get_n()) * headNum *
                          HIDDEN_STRATE_ROPE * sizeof(uint16_t);
    uint64_t workSizeS3 = static_cast<uint64_t>(tilingData.get_n()) *
                          HIDDEN_STRATE_MM * sizeof(uint16_t);
    uint64_t workSizeS4 =
        static_cast<uint64_t>(tilingData.get_n()) *
        std::max((uint32_t)headNum * HIDDEN_STRATE_ROPE, HIDDEN_STRATE_MM) *
        sizeof(uint32_t);
    uint64_t pertokenWorkspace =
        static_cast<uint64_t>(tilingData.get_n()) * sizeof(float) * 2;
    uint64_t maxWorkspaceSize = 0;
    maxWorkspaceSize = std::max(maxWorkspaceSize, workSizeS1);
    maxWorkspaceSize = std::max(maxWorkspaceSize, workSizeS2);
    maxWorkspaceSize = std::max(maxWorkspaceSize, workSizeS3);
    maxWorkspaceSize = std::max(maxWorkspaceSize, workSizeS4);
    size_t usrSize = 0;
    if (inDtype == ge::DT_BF16 ||
        static_cast<QuantMode>(quantMode) == QuantMode::PER_TOKEN_SYMM_QUANT) {
        usrSize = maxWorkspaceSize + maxWorkspaceSize + maxWorkspaceSize +
                  maxWorkspaceSize + pertokenWorkspace;
        tilingData.set_maxWorksapceSize(maxWorkspaceSize);
        tilingData.set_pertokenWorksapce(pertokenWorkspace);
    } else {
        usrSize = maxWorkspaceSize + maxWorkspaceSize + maxWorkspaceSize;
        tilingData.set_maxWorksapceSize(maxWorkspaceSize);
    }
    auto ascendcPlatform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(
        1);  // 通过框架获取workspace的指针，GetWorkspaceSizes入参为所需workspace的块数。当前限制使用一块。
    // printf("usrSize: %lu, maxWorkspaceSize: %lu, pertokenWorkspace: %lu,
    // workSizeS1: %lu, workSizeS2: %lu, workSizeS3: %lu, workSizeS4: %lu\n",
    // usrSize, maxWorkspaceSize, pertokenWorkspace, workSizeS1, workSizeS2,
    // workSizeS3, workSizeS4);
    currentWorkspace[0] =
        usrSize +
        sysWorkspaceSize;  // 设置总的workspace的数值大小，总的workspace空间由框架来申请并管理。
}

void MlaPreprocessTiling::Init(gert::TilingContext* context) {
    auto inDtype = context->GetInputTensor(0)->GetDataType();
    auto ascendPlatform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t& aicNum = ascendPlatform.GetCoreNumAic();
    const int64_t* N = context->GetAttrs()->GetInt(0);
    const int64_t* headNum = context->GetAttrs()->GetInt(1);
    const int64_t* cacheMode = context->GetAttrs()->GetInt(2);
    const int64_t* quantMode = context->GetAttrs()->GetInt(3);

    tilingData.set_n(*N);
    tilingData.set_numCore(aicNum);
    bool deqOnTheFly = false;
    if (inDtype == ge::DT_BF16 ||
        static_cast<QuantMode>(*quantMode) == QuantMode::PER_TOKEN_SYMM_QUANT) {
        deqOnTheFly = true;
    }
    // MKI_LOG(INFO) << "tilingSize: " << kernelInfo.GetTilingSize();
    RmsNormQuantTiling(*N, ascendPlatform.GetCoreNumAiv());
    RopeConcatTiling(*N, *headNum, aicNum);
    EinSumQuantTiling(*N, *headNum, *quantMode, aicNum, inDtype);
    PpMatmulTilingApi mm1TilingApi(1,                      // numBatch
                                   *N,                     // m
                                   HIDDEN_STRATE,          // k
                                   HIDDEN_STRATE_MM,       // n
                                   false,                  // transA
                                   true,                   // transB
                                   true,                   // enDequant
                                   deqOnTheFly, context);  // in bf16.cce?
    mm1TilingApi.GetTilingData(tilingData.mm1);
    PpMatmulTilingApi mm2TilingApi(1,                              // numBatch
                                   *N,                             // m
                                   HIDDEN_STRATE_RMS,              // k
                                   *headNum * HIDDEN_STRATE_ROPE,  // n
                                   false,                          // transA
                                   true,                           // transB
                                   true,                           // enDequant
                                   deqOnTheFly, context);  // in bf16.cce?
    mm2TilingApi.GetTilingData(tilingData.mm2);
    PpMatmulTilingApi mm3TilingApi(*headNum,               // numBatch
                                   *N,                     // m
                                   CONST_128,              // k
                                   CONCAT_SIZE,            // n
                                   false,                  // transA
                                   false,                  // transB
                                   false,                  // enDequant
                                   deqOnTheFly, context);  // in bf16.cce?
    mm3TilingApi.GetTilingData(tilingData.mm3);
    std::cout << tilingData << std::endl;
    SetMlapoWorkSpace(inDtype, *cacheMode, *headNum, *quantMode, context);
    context->SetBlockDim(aicNum);
    SetTilingKey(context, *cacheMode, *quantMode);
    return;
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    MlaPreprocessTiling tiling;
    tiling.Init(context);
    tiling.tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                                   context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const int32_t Q_HEADDIM = 576;
    const int32_t SPLIT_SIZE_ONE = 512;
    const int32_t SPLIT_SIZE_TWO = 64;
    const gert::Shape* tensorQ = context->GetInputShape(0);
    const gert::Shape* tensorKEY = context->GetInputShape(14);
    gert::Shape* out0 = context->GetOutputShape(0);
    gert::Shape* out1 = context->GetOutputShape(1);
    gert::Shape* out2 = context->GetOutputShape(2);
    gert::Shape* out3 = context->GetOutputShape(3);
    const int64_t* N = context->GetAttrs()->GetInt(0);
    const int64_t* headNum = context->GetAttrs()->GetInt(1);
    const int64_t* cacheMode = context->GetAttrs()->GetInt(2);
    if (*cacheMode == 0) {
        *out0 = {*N, *headNum, Q_HEADDIM};
        *out1 = *tensorKEY;
        *out2 = {*N, *headNum, SPLIT_SIZE_TWO};
        *out3 = *tensorKEY;
        out3->SetDim(3, SPLIT_SIZE_TWO);
    } else {
        *out0 = {*N, *headNum, SPLIT_SIZE_ONE};
        *out1 = *tensorKEY;
        out1->SetDim(3, SPLIT_SIZE_ONE);
        *out2 = {*N, *headNum, SPLIT_SIZE_TWO};
        *out3 = *tensorKEY;
        out3->SetDim(3, SPLIT_SIZE_TWO);
    }
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    const auto tensorQDataType = context->GetInputDataType(DIM_0);
    const auto tensorKEYDataType = context->GetInputDataType(14);
    const int64_t* cacheMode = context->GetAttrs()->GetInt(2);
    if (*cacheMode == 0) {
        context->SetOutputDataType(DIM_0, tensorQDataType);
        context->SetOutputDataType(DIM_1, tensorKEYDataType);
        context->SetOutputDataType(DIM_2, tensorQDataType);
        context->SetOutputDataType(DIM_3, tensorKEYDataType);
    } else {
        context->SetOutputDataType(DIM_0, tensorQDataType);
        context->SetOutputDataType(DIM_1, tensorKEYDataType);
        context->SetOutputDataType(DIM_2, tensorQDataType);
        context->SetOutputDataType(DIM_3, tensorKEYDataType);
    }

    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class MLAPreprocess : public OpDef {
   public:
    explicit MLAPreprocess(const char* name) : OpDef(name) {
        this->Input("hiddenState")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("gamma1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("beta1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("quantScale1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("quantOffset1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("wdqkv")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_FRACTAL_NZ})
            .UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ});
        this->Input("bias1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("gamma2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("beta2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("quantScale2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("quantOffset2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("gamma3")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("sin1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("cos1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("keycache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("slotMapping")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("wuq")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_FRACTAL_NZ})
            .UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ});
        this->Input("bias2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("wuk")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("descale1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("descale2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("ctkvScale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("qnopeScale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("q1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("keycacheOut1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("q2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("keycacheOut2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("N").Int();
        this->Attr("headNum").Int();
        this->Attr("cacheMode").Int();
        this->Attr("quantMode").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MLAPreprocess);
}  // namespace ops
