#include "kernel_operator.h"
constexpr int32_t BUFFER_NUM = 2;
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
using namespace AscendC;

template <typename T>
class KernelRopeCache {
   public:
    __aicore__ inline KernelRopeCache() {}

    __aicore__ inline void Init(GM_ADDR dst_cos, GM_ADDR dst_sin, int32_t ne0,
                                int32_t ne1, int32_t s1, int32_t s2,
                                int32_t n_dims, float freq_scale,
                                float theta_scale, float ext_factor,
                                float attn_factor, float corr_dims_v_0,
                                float corr_dims_v_1, float logf_1_freq_scale,
                                int64_t offset) {
        // 计算当前块的工作范围
        int64_t blockNum = AscendC::GetBlockNum();
        int64_t blockIndex = AscendC::GetBlockIdx();

        // 设置参数
        this->ne0 = ne0;        // 向量维度
        this->ne1 = ne1;        // 行数
        this->s1 = s1;          // 行跨度
        this->s2 = s2;          // 块跨度
        this->n_dims = n_dims;  // RoPE维度
        this->freq_scale = freq_scale;
        this->ext_factor = ext_factor;
        this->attn_factor = attn_factor;
        this->theta_scale = theta_scale;
        this->corr_dims_v_0 = corr_dims_v_0;
        this->corr_dims_v_1 = corr_dims_v_1;
        this->logf_1_freq_scale = logf_1_freq_scale;

        this->blockLength = ne0 / 2;
        this->tileNum = 1;
        this->offset = offset;

        cosGm.SetGlobalBuffer(
            (__gm__ float *)dst_cos + blockIndex * this->blockLength,
            this->blockLength);
        sinGm.SetGlobalBuffer(
            (__gm__ float *)dst_sin + blockIndex * this->blockLength,
            this->blockLength);

        pipe.InitBuffer(outQueueDstCos, BUFFER_NUM,
                        this->blockLength * sizeof(T));
        pipe.InitBuffer(outQueueDstSin, BUFFER_NUM,
                        this->blockLength * sizeof(T));
        pipe.InitBuffer(tmpBuf_i0_2_float, this->blockLength * sizeof(float));
        pipe.InitBuffer(tmpBuf_theta_base, this->blockLength * sizeof(float));
        pipe.InitBuffer(tmpBuf_theta_interp, this->blockLength * sizeof(float));
        pipe.InitBuffer(tmpBuf_cos_theta, this->blockLength * sizeof(float));
        pipe.InitBuffer(tmpBuf_sin_theta, this->blockLength * sizeof(float));
        pipe.InitBuffer(tmpBuf_tmp0, this->blockLength * sizeof(float));
        pipe.InitBuffer(tmpBuf_tmp1, this->blockLength * sizeof(float));
        pipe.InitBuffer(tmpBuf_y, this->blockLength * sizeof(float));
        pipe.InitBuffer(tmpBuf_theta_extrap, this->blockLength * sizeof(float));
    }

    __aicore__ inline void Process() {
        auto row = GetBlockIdx();
        CopyIn(row);
        ComputeRope(row);
        CopyOut(row);
    }

   private:
    __aicore__ inline void CopyIn(int32_t row) {
        // 复制输入数据到UB
    }

    __aicore__ inline void ComputeRope(int32_t row) {
        // Assume the block vector is already aligned.
        AscendC::LocalTensor<T> dstCosLocal = outQueueDstCos.AllocTensor<T>();
        AscendC::LocalTensor<T> dstSinLocal = outQueueDstSin.AllocTensor<T>();
        uint32_t blockN = this->blockLength;  // blockN = blockLength = ne0 / 2
        auto row_dst = GetBlockIdx();         // row_dst: current block index
        int32_t pos_val = offset + row_dst;

        // Generate arithmetic progression: i0_2_float[i] = i, for i in [0,
        // blockN - 1]
        LocalTensor<float> /* i0_2_float */ tmp0 =
            tmpBuf_i0_2_float.Get<float>();
        AscendC::ArithProgression<float>(
            /* i0_2_float */ tmp0, 0.0f, 1.0f,
            blockN);  // Formula: i0_2_float[i] = 0.0 + 1.0 * i

        // Compute theta_base: theta_base[i] = (theta_scale)^(i0_2_float[i])
        LocalTensor<float> /* theta_base */ tmp1 =
            tmpBuf_theta_base.Get<float>();
        Power(/* theta_base */ tmp1, theta_scale, /* i0_2_float */ tmp0,
              blockN);  // theta_base = theta_scale^(i0_2_float)
        // Multiply theta_base by pos_val: theta_base[i] = theta_base[i] *
        // pos_val
        Muls(/* theta_base */ tmp1, /* theta_base */ tmp1,
             static_cast<float>(pos_val),
             blockN);  // theta_base = theta_base * pos_val

        // Set frequency factor (here 1.0f so no change)
        const float freq_factor = 1.0f;
        // Compute theta_extrap = theta_base * (1 / freq_factor)
        LocalTensor<float> theta_extrap = tmpBuf_theta_extrap.Get<float>();
        Muls(theta_extrap, /* theta_base */ tmp1, 1 / freq_factor,
             blockN);  // theta_extrap = theta_base * (1 / freq_factor)

        // Compute theta_interp = theta_extrap * freq_scale
        LocalTensor<float> theta_interp = tmpBuf_theta_interp.Get<float>();
        Muls(theta_interp, theta_extrap, freq_scale,
             blockN);  // theta_interp = theta_extrap * freq_scale

        // Temporary buffers for further calculations

        // If ext_factor is non-zero, perform additional interpolation
        // correction
        if (ext_factor != 0.0f) {
            // Compute constant: yy = max(0.001, corr_dims_v_1 - corr_dims_v_0)
            float yy = MAX(0.001f, corr_dims_v_1 - corr_dims_v_0);
            // Compute normalized index: y[i] = (i0_2_float[i] - corr_dims_v_0)
            LocalTensor<float> y = tmpBuf_y.Get<float>();
            Adds(y, /* i0_2_float */ tmp0, -corr_dims_v_0,
                 blockN);  // y = i0_2_float - corr_dims_v_0

            // Normalize: y[i] = y[i] / yy
            Muls(y, y, 1.0f / yy,
                 blockN);  // y = (i0_2_float - corr_dims_v_0) / yy

            // Clamp lower bound: y[i] = max(y[i], 0.0)
            Maxs(y, y, 0.0f, blockN);  // y = max(y, 0.0)

            tmp0 = tmpBuf_tmp0.Get<float>();

            // Clamp upper bound and store in tmp0: tmp0[i] = min(y[i], 1.0)
            Mins(tmp0, y, 1.0f, blockN);  // tmp0 = min(y, 1.0)

            // Compute ramp factor: y[i] = -tmp0[i] + 1.0  => y = 1.0 - tmp0
            Muls(y, tmp0, -1.0f, blockN);  // y = -tmp0
            Adds(y, y, 1.0f, blockN);      // y = 1.0 - tmp0

            // Scale ramp and clamped values by ext_factor
            Muls(y, y, ext_factor, blockN);  // y = ext_factor * (1.0 - tmp0)
            Muls(tmp0, tmp0, ext_factor, blockN);  // tmp0 = ext_factor * tmp0

            // Fuse theta_interp update:
            // tmp0 = tmp0 * theta_interp  and  tmp1 = theta_extrap * y, then:
            // theta_interp = tmp0 + tmp1
            Mul(tmp0, tmp0, theta_interp,
                blockN);  // tmp0 = (ext_factor * tmp0) * theta_interp

            tmp1 = tmpBuf_tmp1.Get<float>();
            Mul(tmp1, theta_extrap, y,
                blockN);  // tmp1 = theta_extrap * (ext_factor * (1.0 - tmp0))
            Add(theta_interp, tmp0, tmp1,
                blockN);  // theta_interp = tmp0 + tmp1

            // Adjust attention factor: attn_factor = attn_factor * (1 + 0.1 *
            // logf_1_freq_scale)
            attn_factor *= (1.0f + 0.1f * logf_1_freq_scale);
        }

        // Compute cosine and sine of theta_interp:
        // cos_theta[i] = cos(theta_interp[i])
        // sin_theta[i] = sin(theta_interp[i])
        LocalTensor<float> cos_theta = tmpBuf_cos_theta.Get<float>();
        LocalTensor<float> sin_theta = tmpBuf_sin_theta.Get<float>();
        Cos(cos_theta, theta_interp, blockN);  // cos_theta = cos(theta_interp)
        Sin(sin_theta, theta_interp, blockN);  // sin_theta = sin(theta_interp)
        // Scale both by attn_factor: cos_theta = cos_theta * attn_factor,
        // similarly for sin_theta
        Muls(dstCosLocal, cos_theta, attn_factor,
             blockN);  // cos_theta = cos_theta * attn_factor
        Muls(dstSinLocal, sin_theta, attn_factor,
             blockN);  // sin_theta = sin_theta * attn_factor
        outQueueDstCos.EnQue(dstCosLocal);
        outQueueDstSin.EnQue(dstSinLocal);
    }

    __aicore__ inline void CopyOut(int32_t row) {
        AscendC::LocalTensor<T> dstCosLocal = outQueueDstCos.DeQue<T>();
        AscendC::LocalTensor<T> dstSinLocal = outQueueDstSin.DeQue<T>();
        DataCopy(cosGm, dstCosLocal, this->blockLength);  // 输出cos_theta
        DataCopy(sinGm, dstSinLocal, this->blockLength);  // 输出sin_theta
        outQueueDstCos.FreeTensor(dstCosLocal);
        outQueueDstSin.FreeTensor(dstSinLocal);
    }

   private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueDstCos;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueDstSin;
    AscendC::GlobalTensor<int32_t> posGm;
    AscendC::GlobalTensor<float> cosGm;
    AscendC::GlobalTensor<float> sinGm;
    TBuf<AscendC::TPosition::VECCALC> tmpBuf_i0_2_float;
    TBuf<AscendC::TPosition::VECCALC> tmpBuf_theta_base;
    TBuf<AscendC::TPosition::VECCALC> tmpBuf_theta_interp;
    TBuf<AscendC::TPosition::VECCALC> tmpBuf_cos_theta;
    TBuf<AscendC::TPosition::VECCALC> tmpBuf_sin_theta;
    TBuf<AscendC::TPosition::VECCALC> tmpBuf_theta_extrap;
    TBuf<AscendC::TPosition::VECCALC> tmpBuf_tmp0;
    TBuf<AscendC::TPosition::VECCALC> tmpBuf_tmp1;
    TBuf<AscendC::TPosition::VECCALC> tmpBuf_y;

    uint32_t blockLength;
    uint32_t tileNum;

    int32_t ne0;          // 向量维度
    int32_t ne1;          // 行数
    int32_t s1;           // 行跨度
    int32_t s2;           // 块跨度
    int32_t n_dims;       // RoPE维度
    float freq_scale;     // 频率缩放
    float ext_factor;     // 外推因子
    float attn_factor;    // 注意力因子
    float theta_scale;    // theta缩放因子
    float corr_dims_v_0;  // YaRN修正维度0
    float corr_dims_v_1;  // YaRN修正维度1
    float logf_1_freq_scale;
    int64_t offset;
};

extern "C" __global__ __aicore__ void ascendc_custom_rope_cache_ext(
    GM_ADDR dst_cos, GM_ADDR dst_sin, int32_t ne0, int32_t ne1, int32_t s1,
    int32_t s2, int32_t n_dims, float freq_scale, float theta_scale,
    float ext_factor, float attn_factor, float corr_dims_v_0,
    float corr_dims_v_1, float logf_1_freq_scale, int64_t offset) {
    KernelRopeCache<float> op;

    op.Init(dst_cos, dst_sin, ne0, ne1, s1, s2, n_dims, freq_scale, theta_scale,
            ext_factor, attn_factor, corr_dims_v_0, corr_dims_v_1,
            logf_1_freq_scale, offset);
    op.Process();
}
