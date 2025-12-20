#include "kernel_operator.h"
constexpr int32_t BUFFER_NUM = 2;
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
using namespace AscendC;

template <typename T>
class KernelRopeV2 {
   public:
    __aicore__ inline KernelRopeV2() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR cos, GM_ADDR sin,
                                GM_ADDR dst, int32_t ne0, int32_t ne1,
                                int32_t pos_len) {
        int64_t blockIndex = GetBlockIdx();
        this->ne0 = ne0;
        this->ne1 = ne1;
        this->pos_len = pos_len;
        this->blockLength = ne0 / 2;
        const int curr_pos = (blockIndex / ne1) % pos_len;

        xGm.SetGlobalBuffer((__gm__ T *)x + blockIndex * this->ne0, this->ne0);
        cosGm.SetGlobalBuffer(
            (__gm__ float *)cos + curr_pos * this->blockLength,
            this->blockLength);
        sinGm.SetGlobalBuffer(
            (__gm__ float *)sin + curr_pos * this->blockLength,
            this->blockLength);
        dstGm.SetGlobalBuffer((__gm__ T *)dst + blockIndex * this->ne0,
                              this->ne0);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->ne0 * sizeof(T));
        pipe.InitBuffer(inQueueCos, BUFFER_NUM,
                        this->blockLength * sizeof(float));
        pipe.InitBuffer(inQueueSin, BUFFER_NUM,
                        this->blockLength * sizeof(float));
        pipe.InitBuffer(outQueueDst, BUFFER_NUM, this->ne0 * sizeof(T));
        pipe.InitBuffer(tmpBuf_x0, this->blockLength * sizeof(float));
        pipe.InitBuffer(tmpBuf_x1, this->blockLength * sizeof(float));
        pipe.InitBuffer(tmpBuf_tmp0, this->blockLength * sizeof(float));
        pipe.InitBuffer(tmpBuf_tmp1, this->blockLength * sizeof(float));
    }

    __aicore__ inline void Process() {
        auto row = GetBlockIdx();
        CopyIn(row);
        ComputeRope(row);
        CopyOut(row);
    }

   private:
    __aicore__ inline void CopyIn(int32_t row) {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        LocalTensor<float> cosLocal = inQueueCos.AllocTensor<float>();
        LocalTensor<float> sinLocal = inQueueSin.AllocTensor<float>();

        DataCopy(xLocal, xGm, this->ne0);
        DataCopy(cosLocal, cosGm, this->blockLength);
        DataCopy(sinLocal, sinGm, this->blockLength);

        inQueueX.EnQue(xLocal);
        inQueueCos.EnQue(cosLocal);
        inQueueSin.EnQue(sinLocal);
    }

    __aicore__ inline void ComputeRope(int32_t row) {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<float> cosLocal = inQueueCos.DeQue<float>();
        LocalTensor<float> sinLocal = inQueueSin.DeQue<float>();
        LocalTensor<T> dstLocal = outQueueDst.AllocTensor<T>();

        uint32_t blockN = this->blockLength;

        LocalTensor<float> x0 = tmpBuf_x0.Get<float>();
        LocalTensor<float> x1 = tmpBuf_x1.Get<float>();
        LocalTensor<float> tmp0 = tmpBuf_tmp0.Get<float>();
        LocalTensor<float> tmp1 = tmpBuf_tmp1.Get<float>();

        if constexpr (std::is_same<T, float>::value) {
            DataCopy(x0, xLocal,
                     blockN);  // x0 = first blockN elements of xLocal
            DataCopy(x1, xLocal[blockN],
                     blockN);  // x1 = second blockN elements of xLocal
        } else {
            // For half precision, cast xLocal to float before copying.
            Cast(x0, xLocal, RoundMode::CAST_NONE,
                 blockN);  // x0 = cast(xLocal) for first blockN elements
            Cast(x1, xLocal[blockN], RoundMode::CAST_NONE,
                 blockN);  // x1 = cast(xLocal) for second blockN elements
        }
        // Compute first output half:
        // tmp0 = x0 * cos_theta and tmp1 = x1 * sin_theta, then tmp0 = tmp0 -
        // tmp1
        Mul(tmp0, x0, cosLocal, blockN);  // tmp0 = x0 * cos(theta_interp)
        Mul(tmp1, x1, sinLocal, blockN);  // tmp1 = x1 * sin(theta_interp)
        Sub(tmp0, tmp0, tmp1,
            blockN);  // tmp0 = (x0 * cos_theta) - (x1 * sin_theta)

        if constexpr (std::is_same<T, float>::value)
            DataCopy(dstLocal, tmp0,
                     blockN);  // dstLocal[0:blockN] = tmp0 (first half)
        else
            Cast(dstLocal, tmp0, RoundMode::CAST_NONE,
                 blockN);  // For half precision, cast result

        // Compute second output half:
        // tmp0 = x0 * sin_theta and tmp1 = x1 * cos_theta, then tmp0 = tmp0 +
        // tmp1
        Mul(tmp0, x0, sinLocal, blockN);  // tmp0 = x0 * sin(theta_interp)
        Mul(tmp1, x1, cosLocal, blockN);  // tmp1 = x1 * cos(theta_interp)
        Add(tmp0, tmp0, tmp1,
            blockN);  // tmp0 = (x0 * sin_theta) + (x1 * cos_theta)

        if constexpr (std::is_same<T, float>::value)
            DataCopy(dstLocal[blockN], tmp0,
                     blockN);  // dstLocal[blockN:2*blockN] = tmp0 (second half)
        else
            Cast(dstLocal[blockN], tmp0, RoundMode::CAST_NONE,
                 blockN);  // For half precision, cast result

        // Enqueue the result tensor and free local buffers
        outQueueDst.EnQue(dstLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueCos.FreeTensor(cosLocal);
        inQueueSin.FreeTensor(sinLocal);
    }

    __aicore__ inline void CopyOut(int32_t row) {
        LocalTensor<T> dstLocal = outQueueDst.DeQue<T>();
        DataCopy(dstGm, dstLocal, this->ne0);
        outQueueDst.FreeTensor(dstLocal);
    }

   private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueCos;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueSin;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueDst;
    GlobalTensor<T> xGm;
    GlobalTensor<T> dstGm;
    GlobalTensor<float> cosGm;
    GlobalTensor<float> sinGm;
    TBuf<TPosition::VECCALC> tmpBuf_x0;
    TBuf<TPosition::VECCALC> tmpBuf_x1;
    TBuf<TPosition::VECCALC> tmpBuf_tmp0;
    TBuf<TPosition::VECCALC> tmpBuf_tmp1;

    uint32_t blockLength;
    uint32_t tileNum;
    int32_t ne0;
    int32_t ne1;
    int32_t pos_len;
};

extern "C" __global__ __aicore__ void rope_ext_custom_v2(GM_ADDR x, GM_ADDR cos,
                                                         GM_ADDR sin,
                                                         GM_ADDR dst,
                                                         GM_ADDR workspace,
                                                         GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelRopeV2<DTYPE_X> op;
    op.Init(x, cos, sin, dst, tiling_data.ne0, tiling_data.ne1,
            tiling_data.pos_len);
    op.Process();
}
