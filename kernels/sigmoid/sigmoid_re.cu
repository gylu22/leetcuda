#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>   

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&value)[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value)(reinterpret_cast<__nv_bfloat162*>(&(value))[0])
//将value的地址重新解释为float4*，然后取第一个元素。
#define LDST128BITS(value)(reinterpret_cast<float4*>(&(value))[0])

#define MAX_EXP_F32 88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f
#define MAX_EXP_F16 __float2half(11.089866488461016f)
#define MIN_EXP_F16 __float2half(-9.704060527839234f)

// fp32 
__global__ void sigmoid_f32_kernel(float *x, float *y,int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<N){
        float v = x[idx];
        v = fminf(fmaxf(v,MIN_EXP_F32),MAX_EXP_F32);
        y[idx] = 1.0f / (1.0f + expf(-v));
    }
}

// 
__global__ void sigmoid_f32x4_kernel(float *x, float*y, int N){

    int idx = 4 *(blockIdx.x * blockDim.x + threadIdx.x);
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;

    reg_x.x = fminf(fmaxf(reg_x.x,MIN_EXP_F32),MAX_EXP_F32);
    reg_x.y = fminf(fmaxf(reg_x.y,MIN_EXP_F32),MAX_EXP_F32);
    reg_x.z = fminf(fmaxf(reg_x.z,MIN_EXP_F32),MAX_EXP_F32);
    reg_x.w = fminf(fmaxf(reg_x.w,MIN_EXP_F32),MAX_EXP_F32);

    reg_y.x = 1.0f/(1.0f + expf(-reg_x.x));
    reg_y.y = 1.0f/(1.0f + expf(-reg_x.y));
    reg_y.z = 1.0f/(1.0f + expf(-reg_x.z));
    reg_y.w = 1.0f/(1.0f + expf(-reg_x.w));

    if ((idx + 3 < N)){
        FLOAT4(y[idx]) = reg_y;
    }
}

__global__ void sigmoid_f16_kernel(half *x, half *y, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const half f = __float2half(1.0f);
    if (idx<N){
        half v = x[idx];
        v = __hmin(__hmax(v,MIN_EXP_F16),MAX_EXP_F16);
        y[idx] = f/(f+hexp(-v));
    }
}

__global__ void sigmoid_f16x2_kernel(half *x, half *y, int N){
    int idx = 2 *(blockIdx.x * blockDim.x + threadIdx.x);
    half2 reg_x = HALF2(x[idx]);
    half2 reg_y;
    const half f = __float2half(1.0f);
    reg_x.x  = __hmin(__hmax(reg_x.x,MIN_EXP_F16),MAX_EXP_F16);
    reg_x.y  = __hmin(__hmax(reg_x.y,MIN_EXP_F16),MAX_EXP_F16);
    reg_y.x = f/(f+hexp(-reg_x.x));
    reg_y.y = f/(f+hexp(-reg_x.y));
    if ((idx + 1 < N)){
        HALF2(y[idx]) = reg_y;
    }
}


__global__ void sigmoid_f16x8_kernel(half *x, half *y, int N){
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    const half f = __float2half(1.0f);

    half2 reg_x_0 = HALF2(x[idx + 0]);
    half2 reg_x_1 = HALF2(x[idx + 2]);
    half2 reg_x_2 = HALF2(x[idx + 4]);
    half2 reg_x_3 = HALF2(x[idx + 6]);

    reg_x_0.x = __hmin(__hmax(reg_x_0.x,MIN_EXP_F16),MAX_EXP_F16);
    reg_x_0.y = __hmin(__hmax(reg_x_0.y,MIN_EXP_F16),MAX_EXP_F16);
    reg_x_1.x = __hmin(__hmax(reg_x_1.x,MIN_EXP_F16),MAX_EXP_F16);
    reg_x_1.y = __hmin(__hmax(reg_x_1.y,MIN_EXP_F16),MAX_EXP_F16);
    reg_x_2.x = __hmin(__hmax(reg_x_2.x,MIN_EXP_F16),MAX_EXP_F16);
    reg_x_2.y = __hmin(__hmax(reg_x_2.y,MIN_EXP_F16),MAX_EXP_F16);
    reg_x_3.x = __hmin(__hmax(reg_x_3.x,MIN_EXP_F16),MAX_EXP_F16);
    reg_x_3.y = __hmin(__hmax(reg_x_3.y,MIN_EXP_F16),MAX_EXP_F16); 

    half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;
    reg_y_0.x = f/(f+hexp(-reg_x_0.x));
    reg_y_0.y = f/(f+hexp(-reg_x_0.y));
    reg_y_1.x = f/(f+hexp(-reg_x_1.x));
    reg_y_1.y = f/(f+hexp(-reg_x_1.y));
    reg_y_2.x = f/(f+hexp(-reg_x_2.x));
    reg_y_2.y = f/(f+hexp(-reg_x_2.y));
    reg_y_3.x = f/(f+hexp(-reg_x_3.x));
    reg_y_3.y = f/(f+hexp(-reg_x_3.y));

    if ((idx + 0) < N) {
        HALF2(y[idx + 0]) = reg_y_0;
    }
    if ((idx + 2) < N) {
        HALF2(y[idx + 2]) = reg_y_1;
    }
    if ((idx + 4) < N) {
        HALF2(y[idx + 4]) = reg_y_2;
    }
    if ((idx + 6) < N) {
        HALF2(y[idx + 6]) = reg_y_3;
}
}

// pack f16x8
__global__ void sigmoid_f16x8_pack_kernel(half *x, half *y,int N){
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    const half f = __float2half(1.0f);
    /*
    8个half（16位）在内存中连续存储：每个half是2字节，所以8个是16字节。
    float4是4个float（32位），也是128位（4*32=128），所以128位的内存可以同时表示为：
    8个half（16位每个）或4个float（32位每个）
    reinterpret_cast允许我们这样解释内存，但必须注意：
    内存必须对齐到float4的要求（16字节对齐）。
    */
    half pack_x[8],pack_y[8];
    // LDST128BITS(x[idx])：将&x[idx]解释为float4*，然后[0]表示取这个128位块作为float4。
    // 但x[idx]是half，所以&x[idx]是half*，但LDST128BITS将其视为float4*。
    // 在kernel中，pack_x[0]是half，但LDST128BITS(pack_x[0])是将pack_x[0]的地址（即&pack_x[0]）解释为float4*，然后取[0]。
    LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);

#pragma unroll
    for (int i=0;i<8;i++){
        pack_x[i] = __hmin(__hmax(pack_x[i],MIN_EXP_F16),MAX_EXP_F16);
        pack_y[i] = f/(f+hexp(-pack_x[i]));
    }   

    if ((idx + 0) < N) {
        LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
    }
}


#define STRINGFY(str) #str

#define TORCH_BINDING_COMMON_EXTENSION(func) \
    m.def(STRINGFY(func), &func, STRINGFY(func));\

# define CHECK_TORCH_TENSOR_DTYPE(x, type) \
    if ((x).options().dtype() != (type)) { \
        std::cout << "Tensor Info:" << (x).options() << std::endl; \
        throw std::invalid_argument("Tensor dtype mismatch"); \
    }  \

/* packed_type: fp16x8, fp16x4, fp32x4
    S: 矩阵行数 ，k 矩阵列数
    为什么要去判断 ((K / (n_elements))<=1024)
        1. 1024 是CUDA限制，CUDA限制一个块的线程数不能超过1024
        2. 如果 K / n_elements > 1024，则线程块大小会超过 1024，导致 编译失败或运行时错误。
    K / n_elements <= 1024，
        线程块大小不会超过 1024，直接设置block大小等于 K/elements
        grid 数量等行数 
    K / n_elements > 1024， 放弃按行分配，改用标准一维分块
        将整个张量视为一维数组（总元素数 N = S*K）
        每个线程块处理 256 个线程（block = 256 / n_elements）
        每个线程处理 n_elements 个元素 → 线程块总处理元素数 = 256 
*/
#define TORCH_BINDING_SIGMOID(packed_type,th_type,element_type,n_elements) \
    void sigmoid_##packed_type(torch::Tensor x, torch::Tensor y){       \
        CHECK_TORCH_TENSOR_DTYPE(x, (th_type));                         \
        CHECK_TORCH_TENSOR_DTYPE(y, (th_type));                         \
        const int ndim = x.dim();     \
        int N;                                   \
        if (ndim != 2){                                                \
            N = 1;                                                 \
            for (int i=0;i<ndim;++i){                                   \
                N *= x.size(i);}                                        \
            dim3 block(256 / (n_elements));                             \
            dim3 grid((N + 256 - 1) / 256);                             \
            sigmoid_##packed_type##_kernel<<<grid,block>>>(             \
            reinterpret_cast<element_type*>(x.data_ptr()),          \
            reinterpret_cast<element_type*>(y.data_ptr()), N);      \
        } else {                                                     \
            const int S = x.size(0);                                \
            const int K = x.size(1);                                \
            N = S * K;                                    \
            if ((K / (n_elements))<=1024){                          \
                dim3 block(K / (n_elements));                           \
                dim3 grid(S);                                           \
                sigmoid_##packed_type##_kernel<<<grid, block>>>(        \
                reinterpret_cast<element_type *>(x.data_ptr()),         \
                reinterpret_cast<element_type *>(y.data_ptr()), N);     \
            } else {                                                \
                int N = 1;                                          \
                for (int i = 0; i < ndim; ++i) {                    \
                N *= x.size(i);}                                    \
                dim3 block(256 / (n_elements));                     \
                dim3 grid((N + 256 - 1) / 256);                     \
                sigmoid_##packed_type##_kernel<<<grid, block>>>(    \
                reinterpret_cast<element_type *>(x.data_ptr()),     \
                reinterpret_cast<element_type *>(y.data_ptr()), N); \
            }                                                       \
        }                                                               \
    }                                                                   \


TORCH_BINDING_SIGMOID(f32, torch::kFloat32, float, 1)
TORCH_BINDING_SIGMOID(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_SIGMOID(f16, torch::kHalf, half, 1)
TORCH_BINDING_SIGMOID(f16x2, torch::kHalf, half, 2)
TORCH_BINDING_SIGMOID(f16x8, torch::kHalf, half, 8)
TORCH_BINDING_SIGMOID(f16x8_pack, torch::kHalf, half, 8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f32)
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f16)
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f16x8_pack)
}