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

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])


__global__ void embedding_f32_kernel(const int *idx, float *weight
                                        float *output, int n, int emb_size){
    // 当前的处理逻辑，每一个block 处理一个embedding 
    // block中的thread处理embedding中的每一个元素
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    // int tid = bx * blockDim.x + tx; // global thread id
    int offset = idx[bx] * emb_size; //计算当前词在权重数组中的起始位置
    // bx * emb_size 计算当前词在输出数组中的起始位置 tx 代表在embedidng中处理的元素索引
    // offset + tx 计算当前词在embedding中处理的元素索引
    output[bx * emb_size + tx] = weight[offset + tx];
}                                        

__global__ void embedding_f32x4_kernel(const int *idx, float *weight
                                        float *output, int n, int emb_size){

        int tx = threadIdx.x * 4;
        int bx = blockIdx.x;
        int offset = idx[bx] * emb_size;
        output[bx * emb_size + tx] = weight[offset + tx];
        output[bx * emb_size + tx + 1] = weight[offset + tx +1];
        output[bx * emb_size + tx + 2] = weight[offset + tx +2];
        output[bx * emb_size + tx + 3] = weight[offset + tx +3];
}



__global__ void embedding_f16_kernel(const int *idx, half *weight, half *output,
                                     int n, int emb_size) {
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int tid = bx * blockDim.x + tx;
  int offset = idx[bx] * emb_size;
  output[bx * emb_size + tx] = weight[offset + tx];
}

__global__ void embedding_f16x8_kernel(const int *idx, half *weight,
                                       half *output, int n, int emb_size) {
  int tx = threadIdx.x * 8;
  int bx = blockIdx.x;
  int offset = idx[bx] * emb_size;
  output[bx * emb_size + tx] = weight[offset + tx];
  output[bx * emb_size + tx + 1] = weight[offset + tx + 1];
  output[bx * emb_size + tx + 2] = weight[offset + tx + 2];
  output[bx * emb_size + tx + 3] = weight[offset + tx + 3];
  output[bx * emb_size + tx + 4] = weight[offset + tx + 4];
  output[bx * emb_size + tx + 5] = weight[offset + tx + 5];
  output[bx * emb_size + tx + 6] = weight[offset + tx + 6];
  output[bx * emb_size + tx + 7] = weight[offset + tx + 7];
}

__global__ void embedding_f16x8_pack_kernel(const int *idx, half *weight,
                                            half *output, int n, int emb_size){

  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int offset = idx[bx] * emb_size;
  LDST128BITS(output[bx * emb_size + tx * 8]) = LDST128BITS(weight[offset + tx * 8]);

}