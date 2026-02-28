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
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define MAX_EXP_F32 88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f
#define MAX_EXP_F16 __float2half(11.089866488461016f)
#define MIN_EXP_F16 __float2half(-9.704060527839234f)

#define HLAF_1 __float2half(1.0f)
#define HALF_2 __float2half(2.0f)
#define HALF_DIV2 __float2half(0.5f)

// M_SQRT2 M_2_SQRTPI math中定义的宏 
#define SQRT_2_PI =  M_2_SQRTPI * M_SQRT2 * 0.5f    

#define HALF_SQRT_2_PI __float2half(M_2_SQRTPI) * __float2half(M_SQRT2) * HALF_DIV2

#define HALF_GELU_OPS gelu_tanh_approximate
#define GELU_OPS gelu_tanh_approximate

// gelu_f32
__device__ __forceinline__ float gelu_tanh_approximate(float x) { 
    return 0.5f * x * (1.0f + tanhf(SQRT_2_PI *(x + 0.044715 * x * x * x)));
}

// gelu_f16
__device__ __forceinline__ half gelu_tanh_approximate(half,x){

    half x_cube = x * x * x;
    half inner = HALF_SQRT_2_PI * (x + __float2half(0.044715f) * x_cube);
    half app_tanh = (hexp(HALF_2 * inner)-HALF_1) / (hexp(HALF_2 * inner) + HALF_1);
    return HALF_DIV2 * x *(HALF_1 + app_tanh);
}
//gelu none approximate 
__device__ __forceinline__ float gelu_none_approximate(float x){
    return 0.5f* x *(1.0f + erff(x * M_SQRT1_2));
}


__device__ void gelu_f32_kernel(float *x ,float *y,int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
        float v = fminf(fmax(x[idx],MIN_EXP_F32),MAX_EXP_F32);
        y[idx] = GELU_OPS(v);
    }
}

__device__ void gelu_f32x4_kernel(half *x ,half *y,int N){
    int idx = 4*(blockIdx.x * blockDim.x + threadIdx.x);
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;

    reg_x.x = fminf(fmax(reg_x.x,MIN_EXP_F32),MAX_EXP_F32);
    reg_x.y = fminf(fmax(reg_x.y,MIN_EXP_F32),MAX_EXP_F32);
    reg_x.z = fminf(fmax(reg_x.z,MIN_EXP_F32),MAX_EXP_F32);
    reg_x.w = fminf(fmax(reg_x.w,MIN_EXP_F32),MAX_EXP_F32);

    reg_y.x = GELU_OPS(reg_x.x);
    reg_y.y = GELU_OPS(reg_x.y);
    reg_y.z = GELU_OPS(reg_x.z);
    reg_y.w = GELU_OPS(reg_x.w);
    
    if ((idx + 3) < N){
        FLOAT4(y[idx]) = reg_y;
    }
}