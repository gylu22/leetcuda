#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

using namespace nvcuda;

#define WARP_SIZE 32 
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HOST_DEVICE_INLINE __device__ __host__ inline
#define PAD 8
HOST_DEVICE_INLINE
int div_ceil(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

// 一个block有32个线程， tile 大小是16x16 
// tensor core
template<const int WMMA_M=16,const int WMMA_N=16,const int WMMA_K=8>
__global__ void sgemm_wmma_naive_kernel(float* a,float *b,float *c, int M, int N, int K){

    // 计算循环的次数
    const int tile_k = div_ceil(K,WMMA_K);

    const int row = blockIdx.y * WMMA_M;
    const int col = blockIdx.x * WMMA_N;
    if (row >= M || col >= N) return;
    // 定义 fragment
    wmma::fragment<wmma::accumulator,WMMA_M, WMMA_N,WMMA_K,float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);
    
    #pragma unroll
    for (int k=0;k <tile_k;k++){
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,wmma::precision::tf32,wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,wmma::precision::tf32,wmma::row_major> b_frag;
        wmma::load_matrix_sync(a_frag, a + row * K + k * WMMA_K, K);
        wmma::load_matrix_sync(b_frag, b + k * WMMA_K * N + col, N);
        wmma::mma_sync(c_frag,a_frag,b_frag,c_frag);
    }
    wmma::store_matrix_sync(c + row * N + col, c_frag, N, wmma::mem_row_major);
}

// shared memory + warp tiling + wmma(tensor core)
// tile size 128 x128 
// block size 128 = 4 warp
// 每个 warp 负责 64x64的大小区域 
template<const int BM,const int BN,const int BK, const int WARP_M,const int WARP_N,
const int WMMA_M=16,const int WMMA_N=16,const int WMMA_K=8>
__global__ void sgemm_wmma_shared_warp_tiling_kernel(float *a ,float *b,float *c, const int M ,const int N,const int K){
    
    __shared__ float s_a[BM][BK],s_b[BK][BN];

    const int WARP_TILE_M = WARP_M / WMMA_M; // 64 / 16 = 4
    const int WARP_TILE_N = WARP_N / WMMA_N; 
    const int warp_id = threadIdx.x / 32;
    // warp排布是2x2 
    const int warp_row = warp_id / 2;
    const int warp_col = warp_id % 2;
    int tx = threadIdx.x;

    float *a_ptr = a + blockIdx.y * BM * K ;
    float *b_ptr = b + blockIdx.x * BN;

    // 每个 Warp 负责 64x64 输出，所以需要 4x4 = 16 个累加器 Fragment
    wmma::fragment<wmma::accumulator,WMMA_M,WMMA_N,WMMA_K,float>  c_frag[WARP_TILE_M][WARP_TILE_N];

    for (int i=0;i<WARP_TILE_M;i++){
        for (int j=0;j<WARP_TILE_N;j++){
            wmma::fill_fragment(c_frag[i][j],0.0f);
        }
    }

    // 外部循环
    for (int tile_idx =0; tile_idx <K; tile_idx+=BK){
        //load tile from global memory -> shared memory 
        // 对于A来说，使用向量化加载，每行使用32/4 = 8个线程，128 / 8 外部需要循环16次
        const int thread_col_sa = BK / 4; // 8
        const int thread_row_sa = BM / thread_col_sa; // 16 
        #pragma unroll
        for (int sa_idx=0; sa_idx<BM; sa_idx += thread_row_sa){
            int sa_row = tx / thread_col_sa + sa_idx; 
            int sa_col = (tx % thread_col_sa) * 4; // 0,4,8...
            FLOAT4(s_a[sa_row][sa_col]) = FLOAT4(a_ptr[sa_row * K + sa_col + tile_idx]);
        }

        const int thread_col_sb = BN / 4; // 32
        const int thread_row_sb = BN / thread_col_sb; // 128 / 32 = 4
        #pragma unroll
        for (int sb_idx =0; sb_idx<BK; sb_idx+=thread_row_sb){
          // 外层8次循环
          int sb_row = tx / thread_col_sb + sb_idx ;
          int sb_col = (tx % thread_col_sb) * 4;
          FLOAT4(s_b[sb_row][sb_col]) = FLOAT4(b_ptr[(sb_row + tile_idx) * N + sb_col]); 
        }
        __syncthreads();

        // 内存循环，在shared memory 内部做循环
        // 128 x 32 x 32 x128 
        // 单个 warp负责处理 64 x 64 的区域
        // 外部循环沿 BK 方向 总共循环 32 / 8 四次
        for (int k_inner = 0; k_inner < BK; k_inner += WMMA_K){
          // 
          wmma::fragment<wmma::matrix_a,WMMA_M,WMMA_N,WMMA_K,wmma::precision::tf32,wmma::row_major> a_frag[WARP_TILE_M];
          wmma::fragment<wmma::matrix_b,WMMA_M,WMMA_N,WMMA_K,wmma::precision::tf32,wmma::row_major> b_frag[WARP_TILE_N];

          // 
          #pragma unroll
          for (int i=0; i<WARP_TILE_M; i++){
            wmma::load_matrix_sync(a_frag[i],&s_a[(warp_row * WARP_M+i * WMMA_M)][k_inner],BK);
          }
          #pragma unroll
          for (int j=0;j<WARP_TILE_N;j++){
            wmma::load_matrix_sync(b_frag[j],&s_b[k_inner][WARP_N * warp_col + j * WMMA_N],BN);
          }
          // 开始计过程

          #pragma unroll
          for (int i = 0; i < WARP_TILE_M; i++) {
              #pragma unroll
              for (int j = 0; j < WARP_TILE_N; j++) {
                  wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
              }
          }
        }
        __syncthreads();

        
      }
      // 写入输出矩阵
      #pragma unroll 
        for (int i = 0; i < WARP_TILE_M; i++) {
          #pragma unroll
          for (int j = 0; j < WARP_TILE_N; j++) {
            int global_c_row = blockIdx.y * BM + warp_row * WARP_M + i * WMMA_M;
            int global_c_col = blockIdx.x * BN + warp_col * WARP_N + j * WMMA_N;
            
            // 确保不越界再写入
            if (global_c_row < M && global_c_col < N) {
                wmma::store_matrix_sync(&c[global_c_row * N + global_c_col], c_frag[i][j], N, wmma::mem_row_major);
            }
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


void sgemm_wmma_naive(torch::Tensor a,torch::Tensor b, torch::Tensor c){

    CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 8;

    dim3 block(WARP_SIZE);
    dim3 grid(div_ceil(N,WMMA_N),div_ceil(M,WMMA_M));

    sgemm_wmma_naive_kernel<WMMA_M,WMMA_N,WMMA_K><<<grid,block>>>(
        reinterpret_cast<float *>(a.data_ptr()),
        reinterpret_cast<float *>(b.data_ptr()),
        reinterpret_cast<float *>(c.data_ptr()), M, N, K);
}


void sgemm_wmma_shared_warp_tiling(torch::Tensor a,torch::Tensor b, torch::Tensor c){
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)

    const int BM = 128;
    const int BN = 128;
    const int BK = 32;
    
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 8;

    const int WARP_M = 64;
    const int WARP_N = 64;

    dim3 block(WARP_SIZE * 4);
    dim3 grid(div_ceil(N,BN), div_ceil(M,BM));
    sgemm_wmma_shared_warp_tiling_kernel<BM,BN,BK,WARP_M,WARP_N,WMMA_M,WMMA_N,WMMA_K>
    <<<grid,block>>>(reinterpret_cast<float *>(a.data_ptr()),
                     reinterpret_cast<float *>(b.data_ptr()),
                     reinterpret_cast<float *>(c.data_ptr()),M,N,K);

}