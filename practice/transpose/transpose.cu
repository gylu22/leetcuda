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

#define WARP_SIZE 256
#define WARP_SIZE_S 16
#define PAD 1
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define MAX_EXP_F32 88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f
#define MAX_EXP_F16 __float2half(11.089866488461016f)
#define MIN_EXP_F16 __float2half(-9.704060527839234f)


// fp32 1D grid  矩阵转置 column to row 线程ID对应输入矩阵的索引
// 最基础版本
__global__ void mat_transpose_f32_col2row_kernel(float *x, float *y, int row, int col) { 

    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_row = global_idx / col; 
    const int global_col = global_idx % col;  
    if (global_idx < row * col) {
        y[global_col * row + global_row] = x[global_idx];
    }
}

// fp32 1D grid   
__global__ void mat_transpose_f32_row2col_kernel(float *x, float *y, int row, int col) { 
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 反向计算[global_idx]位置的元素在输入x中的索引
    // 写入连续，读取不连续
    const int global_col = global_idx / row;
    const int global_row = global_idx % row;
    if (global_idx < row * col){
      y[global_idx] = x[global_row * col + global_col];
    }
}

// 1d 向量化 col2Row  f32x4
__global__ void mat_transpose_f32x4_col2row_kernel(float *x, float *y, int row, int col){

    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_row = global_idx * 4 / col; // 输入矩阵的行索引
    const int global_col = global_idx * 4 % col; // 输入矩阵的列索引
    // 边界条件
    if (global_row < row && global_col + 3 < col){
      float4 x_vec = FLOAT4(x[global_idx * 4]);
      y[global_col * row + global_row] = x_vec.x;
      y[(global_col+1) * row + global_row] = x_vec.y;
      y[(global_col+2) * row + global_row] = x_vec.z;
      y[(global_col+3) * row + global_row] = x_vec.w;
    }
}

// 1d 向量化 row2Col

__global__ void mat_transpose_f32x4_row2col_kernel(float *x, float *y, int row, int col){ 
  
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_col = global_idx * 4 / row; 
    const int global_row = global_idx * 4 % row; 
    // 边界条件
    if (global_row < row && global_col < col){
      float4 x_vec;
      x_vec.x = x[global_row * col + global_col];
      x_vec.y = x[(global_row + 1) * col + global_col];
      x_vec.z = x[(global_row + 2) * col + global_col];
      x_vec.w = x[(global_row + 3) * col + global_col];
      FLOAT4(y[global_idx*4]) = x_vec;
    }
}


// fp32 2D grid  矩阵转置
// 最基础版本
__global__ void mat_transpose_f32_col2row2d_kernel(float *x, float *y, int row, int col) { 

    const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (global_x < col && global_y < row){
      y[global_x * row + global_y] = x[global_y * col + global_x];
    }
}



__global__ void mat_transpose_f32_row2col2d_kernel(float *x, float *y, int row, int col) { 

    const int global_y = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_x = blockIdx.y * blockDim.y + threadIdx.y;
    if (global_y < col && global_x < row){
      y[global_y * row + global_x] = x[global_x * col + global_y];
    }
}

// 矩阵转置 + 向量化读取
__global__ void mat_transpose_f32x4_col2row2d_kernel(float *x, float *y, int row, int col) { 

  // 
  int tx = blockDim.x * blockIdx.x + threadIdx.x; // x 列索引
  int ty = blockDim.y * blockIdx.y + threadIdx.y; // x 行索引
  // 边界条件
  if (tx * 4 + 3  < col && ty < row){
    // 一个线程向量化读取四个元素
    float4 x_vec = FLOAT4(x[ty * col + tx * 4]);
    y[tx * 4  * row + ty] = x_vec.x;
    y[(tx * 4 + 1) * row + ty] = x_vec.y;
    y[(tx * 4 + 2) * row + ty] = x_vec.z;
    y[(tx * 4 + 3) * row + ty] = x_vec.w;
  }
}

// 矩阵转置  shared_memory + float4 
__global__ void mat_transpose_f32x4_shared_col2row2d_kernel(float *x, float *y,
                                                            const int row,
                                                            const int col) {
    __shared__ float tile[WARP_SIZE_S][WARP_SIZE_S * 4]; 

    // ====== 1. 加载阶段：从 Global 向量化加载到 Shared (保持不变) ======
    const int gx = blockDim.x * blockIdx.x + threadIdx.x; // x的局部列
    const int gy = blockDim.y * blockIdx.y + threadIdx.y; // x的局部行
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    // 每个block的起始地址
                                                         
    // 边界检查并向量化写入 shared memory
    if (gx * 4 + 3 < col && gy < row) {
        FLOAT4(tile[ty][tx * 4]) = FLOAT4(x[gy * col + gx * 4]);
        __syncthreads();
        // 此时有 16x16 个线程负责读取 16x64个元素 写入到输出矩阵中
        // 输出矩阵这部分的形状是 64 x 4 
        // 重新计算对共享内存的索引
        const int tid = ty * blockDim.x + tx; // 单个block中的全局索引
        int by =  tid / (WARP_SIZE_S * 4); // 0-3 y 
        int bx = tid % (WARP_SIZE_S * 4);
        // 输出 block的起始地址
        float *y_ptr = y + blockDim.x * blockIdx.x * 4 * row + blockDim.y * blockIdx.y; 
        float4 x_shared;
        x_shared.x = tile[by*4][bx];
        x_shared.y = tile[by*4+1][bx];
        x_shared.z = tile[by*4+2][bx];
        x_shared.w = tile[by*4+3][bx];
        FLOAT4(y_ptr[bx * row + by * 4]) = x_shared;
    }
}

// 矩阵转置  shared_memory + float4 + bank conflict  
__global__ void mat_transpose_f32x4_shared_bcf_col2row2d_kernel(float *x, float *y,
                                                            const int row,
                                                            const int col) {
    __shared__ float tile[WARP_SIZE_S][WARP_SIZE_S * 4 + 2 * PAD]; 

    // ====== 1. 加载阶段：从 Global 向量化加载到 Shared (保持不变) ======
    const int gx = blockDim.x * blockIdx.x + threadIdx.x; // x的局部列
    const int gy = blockDim.y * blockIdx.y + threadIdx.y; // x的局部行
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;                                                 
    // 边界检查并向量化写入 shared memory
    if (gx * 4 + 3 < col && gy < row) {
        float4 x_val = FLOAT4(x[gy * col + gx * 4]);
        tile[ty][tx * 4] = x_val.x;
        tile[ty][tx * 4 + 1] = x_val.y;
        tile[ty][tx * 4 + 2] = x_val.z;
        tile[ty][tx * 4 + 3] = x_val.w;
        __syncthreads();
        // 此时有 16x16 个线程负责读取 16x64个元素 写入到输出矩阵中
        // 输出矩阵这部分的形状是 64 x 4 
        // 重新计算对共享内存的索引
        const int tid = ty * blockDim.x + tx; // 单个block中的全局索引
        int by =  tid / (WARP_SIZE_S * 4); // 0-3 y 
        int bx = tid % (WARP_SIZE_S * 4);
        // 输出 block的起始地址
        float *y_ptr = y + blockDim.x * blockIdx.x * 4 * row + blockDim.y * blockIdx.y; 
        float4 x_shared;
        x_shared.x = tile[by*4][bx];
        x_shared.y = tile[by*4+1][bx];
        x_shared.z = tile[by*4+2][bx];
        x_shared.w = tile[by*4+3][bx];
        FLOAT4(y_ptr[bx * row + by * 4]) = x_shared;
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

#define TORCH_BINDING_MAT_TRANSPOSE(tag, th_type, element_type, n_pack)        \
  void mat_transpose_##tag(torch::Tensor x, torch::Tensor y) {                 \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                     \
    const int M = x.size(0);                                                   \
    const int N = x.size(1);                                                   \
    dim3 block(WARP_SIZE);                                                     \
    dim3 grid(((N * M + WARP_SIZE - 1) / n_pack / WARP_SIZE));                 \
    mat_transpose_##tag##_kernel<<<grid, block>>>(                             \
        reinterpret_cast<element_type *>(x.data_ptr()),                        \
        reinterpret_cast<element_type *>(y.data_ptr()), M, N);                 \
  }


#define TORCH_BINDING_MAT_TRANSPOSE2D(tag, th_type, element_type,              \
                                      n_element_row, n_element_col)            \
  void mat_transpose_##tag##2d(torch::Tensor x, torch::Tensor y) {             \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                     \
    const int M = x.size(0);                                                   \
    const int N = x.size(1);                                                   \
    dim3 block(WARP_SIZE_S, WARP_SIZE_S);                                      \
    dim3 grid((N + WARP_SIZE_S - 1) / (WARP_SIZE_S * n_element_col),           \
              (M + WARP_SIZE_S - 1) / (WARP_SIZE_S * n_element_row));          \
    mat_transpose_##tag##2d_kernel<<<grid, block>>>(                           \
                   reinterpret_cast<element_type *>(x.data_ptr()),             \
                   reinterpret_cast<element_type *>(y.data_ptr()), M, N);      \
  }


// 1d index
TORCH_BINDING_MAT_TRANSPOSE(f32_col2row, torch::kFloat32, float, 1)
TORCH_BINDING_MAT_TRANSPOSE(f32_row2col, torch::kFloat32, float, 1)
// 1d vec
TORCH_BINDING_MAT_TRANSPOSE(f32x4_col2row, torch::kFloat32, float, 4)
TORCH_BINDING_MAT_TRANSPOSE(f32x4_row2col, torch::kFloat32, float, 4)
// 2D index
TORCH_BINDING_MAT_TRANSPOSE2D(f32_col2row, torch::kFloat32, float, 1, 1)
TORCH_BINDING_MAT_TRANSPOSE2D(f32_row2col, torch::kFloat32, float, 1, 1)
// 2D vec
TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_col2row, torch::kFloat32, float, 1, 4)

// 2D shared memory 

TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_shared_col2row, torch::kFloat32, float, 1, 4)
TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_shared_bcf_col2row, torch::kFloat32, float, 1, 4)


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 1d index
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_col2row)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_row2col)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_col2row)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_row2col)

  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_col2row2d) 
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_row2col2d)

  // 2d index vec
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_col2row2d)

  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_shared_col2row2d)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_shared_bcf_col2row2d)

}
