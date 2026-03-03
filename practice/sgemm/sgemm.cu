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


// block 32x32 大小
__global__ void sgemm_naive_f32_kernel(float *a, 
          float *b, float *c, int M,int N, int K){
  
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx < N && ty <M){
      float sum =0; 
      #pragma unroll
      for (int i=0; i<K; i++){
        sum += a[ty * K + i] * b[i * N + tx];
      }
      c[ty * N + tx] = sum;
    }
}

// shared memory + Block Tile 
template <const int BM = 32, const int BN = 32, const int BK = 32>
__global__ void sgemm_sliced_k_f32_kernel(float *a, float *b, float *c, int M,
                                          int N, int K) {

  float __shared__ tileA[BM][BK],tileB[BK][BN];
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  int row = by * blockDim.y + ty;
  int col = bx * blockDim.x + tx; 

  float *a_ptr = a + by * blockDim.y * K;
  float *b_ptr = b + bx * blockDim.x; 

  // 外部循环 （K + BK-1）/ BK次 
  float sum = 0.f;
  for (int k=0; k<K; k+= BK){
    // 将一个tile的数据导入到shared memory 
    if (row < M && k + tx < K ){
      tileA[ty][tx] = a_ptr[ty * K + k + tx];
    }
    else {
      tileA[ty][tx] = 0.f;
    }
    if (col < N && k + ty < K){
      tileB[ty][tx] = b_ptr[(k + ty) * N + tx];
    }
    else {
      tileB[ty][tx] = 0.f;
    }
    __syncthreads();

    #pragma unroll
    for (int i =0;i<BK;i++){
      sum += tileA[ty][i] * tileB[i][tx];
    }
    __syncthreads();
  }
  c[row * N + col] = sum;

}

// block size 16 x 16 Tile Size: 128 x 128 
// 单个线程计算 128 / 16 x 128  / 16 = 8x8 大小  
// shared memory 大小 128 x 8

template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8>
__global__ void sgemm_t_8x8_sliced_k_f32x4_kernel(float *a, float *b, float *c,
                                                  int M, int N, int K) {

    float __shared__ tileA[BM][BK], tileB[BK][BN];
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float *a_ptr = a + blockIdx.y * BM * K;
    float *b_ptr = b + blockIdx.x * BN;
    float *c_ptr = c + blockIdx.y * BM * N + blockIdx.x * BN;
    int tid = ty * blockDim.x + tx; // 0-127

    float temp[TM][TN] = {0.f};
    // 重新映射线程
    int ta_y = tid / (BK / 4); // 0-127
    int ta_x = (tid % (BK / 4)) * 4 ; // 0,4     
    int tb_y = tid / (BN / 4); // 0-7
    int tb_x = tid % (BN / 4) * 4; // 0,4,8...  

    // 外层循环（M+BK -1）/ BK
    for (int k=0; k<K; k+=BK){
  
      FLOAT4(tileA[ta_y][ta_x]) = FLOAT4(a_ptr[ta_y * K + ta_x + k]);
      FLOAT4(tileB[tb_y][tb_x]) = FLOAT4(b_ptr[(k + tb_y) * N + tb_x]);
      // float4 x_vec = FLOAT4(a_ptr[ta_y * K + ta_x + k]);
      // tileA[ta_y][ta_x] = x_vec.x;
      // tileA[ta_y][ta_x+1] = x_vec.y;
      // tileA[ta_y][ta_x+2] = x_vec.z;
      // tileA[ta_y][ta_x+3] = x_vec.w;
      
      // float4 b_vec = FLOAT4(b_ptr[(k + tb_y) * N + tb_x]);
      // tileB[tb_y][tb_x] = b_vec.x;
      // tileB[tb_y][tb_x+1] = b_vec.y;
      // tileB[tb_y][tb_x+2] = b_vec.z;
      // tileB[tb_y][tb_x+3] = b_vec.w;
      
      // 第一次同步，保证第一次循环计算之前所有元素都读取到shared memory
      __syncthreads();
      // 计算每个线程的BK x BK的区域
      // 内层循环BK次
      #pragma unroll
      for (int i = 0; i < BK; i++) {
        float tileA_BK[TM];
        float tileB_BK[TN];
        #pragma unroll
        for (int j = 0; j < TM; j++) {
          tileA_BK[j] = tileA[ty * TM + j][i];
        }
        #pragma unroll
        for (int j = 0; j < TN; j++) {
          tileB_BK[j] = tileB[i][tx * TN + j];
        }
        #pragma unroll
        for (int a_i = 0; a_i < TM; a_i++) {
          for (int b_j = 0; b_j < TN; b_j++) {
            temp[a_i][b_j] += tileA_BK[a_i] * tileB_BK[b_j];
          }
        }
      }
      // 第二次同步，保证下一次更新shared memory之前已经计算完成
      __syncthreads();
      }
      #pragma unroll
      for (int i=0;i<TM;i++){
        for (int j=0;j<TN;j++){
          c_ptr[(ty * TM + i) * N + tx * TN + j] = temp[i][j];
        }
      }
}


#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor size mismatch!");                         \
  }


void sgemm_naive_f32(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 32;
  constexpr int BN = 32;

  dim3 block(BN, BM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  sgemm_naive_f32_kernel<<<grid, block>>>(
      reinterpret_cast<float *>(a.data_ptr()),
      reinterpret_cast<float *>(b.data_ptr()),
      reinterpret_cast<float *>(c.data_ptr()), M, N, K);
}

void sgemm_sliced_k_f32(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 32;
  constexpr int BN = 32;
  constexpr int BK = 32;

  dim3 block(BN, BM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  sgemm_sliced_k_f32_kernel<BM, BN, BK>
      <<<grid, block>>>(reinterpret_cast<float *>(a.data_ptr()),
                        reinterpret_cast<float *>(b.data_ptr()),
                        reinterpret_cast<float *>(c.data_ptr()), M, N, K);
}

void sgemm_t_8x8_sliced_k_f32x4(torch::Tensor a, torch::Tensor b,
                                torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;

  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  sgemm_t_8x8_sliced_k_f32x4_kernel<BM, BN, BK, TM, TN>
      <<<grid, block>>>(reinterpret_cast<float *>(a.data_ptr()),
                        reinterpret_cast<float *>(b.data_ptr()),
                        reinterpret_cast<float *>(c.data_ptr()), M, N, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(sgemm_naive_f32)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_sliced_k_f32)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_t_8x8_sliced_k_f32x4)

}