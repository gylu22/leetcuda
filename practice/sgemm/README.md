# SGEMM

sgemm_naive_f32_kernel： 
* 最基础版本，无任何加速方法，单个线程负责读取A矩阵一行和B矩阵一列，计算输出矩阵一个元素的结果；

sgemm_sliced_k_f32_kernel： 
* 使用shared memory 版本，分块加载，Tile大小和block大小相等；
* global memory -> shared memory,bank conflict 分析(block size 32x32): 
tileA和tileB在写入shared memory的时候, 对于一个warp 来说，ty 固定， tx：0-31, 刚好均匀分布在32路bank中
* 读shared memroy -> bank conflict 分析(block size 32x32)：
在计算的读取shared memory的时候，循环tile_size=32的情况下，一个warp，第一次循环， ty 固定，0-31号线程都访问bank0 0号元素， 触发广播，无冲突。
读取tileB: ty 固定， tx: 0-31 号线程,j均匀分布到32路bank中，无冲突。

sgemm_t_8x8_sliced_k_f32x4：
* shared memory + thread coarsing + 向量化读取
```cpp
    // 分析读取shared memory 时的bank conflict 
    // block 大小是 16 x 16 ， 
    // 对于第一个 Warp 0-15号， ty=0，tx：0-15；16-31号线程 ty=1,tx:0-15
    // tileA来说 128 x 8
    //============================ tileA[i][ty * TM + j]==========================  
    // Address = (ty * 8 + j) * 8 + i 
    // 对于前16号线程 ty=0 address = 8*j + i bank = (8*j + i) % 32 触发广播
    // 对于16-31号线程 ty = 1 address = 64+ 8*j +i  bank = (64+ 8*j +i) % 32 = (8*j +i) % 32  触发广播，但是和前16号线程处于同一个bank，此时触发2-way bank conflict 
    // 此时采用 padding  128 x 9 
    // ty =0 bank = (9*j + i) % 32
    // ty =1 bank = (72 + 9*j + i) % 32 
    // 此时不在同一个 bank

    // tileB来说 8 x 128
    //============================ tileB[i][tx * TN + j] ========================== 
    // Address = 128 * i + tx * 8 + j = 8 * tx (简化 i=0,j=0)
    // 此时 tx 0-15 
    // tx = 0,tx = 4 ,tx =8 tx = 12, 都在bank 0 造成 4-way bank conflict
    // 此时padding 无法解决bank冲突问题
    // bank = 8 * tx % 32 仍然是四路bank 冲突
```
segmm_t_8x8_sliced_k_bcf_f32x4:
* shared memory _ thread coarsing + 向量化读取 + a 转置读取
```c++
template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8>
__global__ void sgemm_t_8x8_sliced_k_bcf_f32x4_kernel(float *a, float *b, float *c,
                                                  int M, int N, int K) {

    float __shared__ tileA[BK][BM], tileB[BK][BN];
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
      float4 x_vec = FLOAT4(a_ptr[ta_y * K + ta_x + k]);

      tileA[ta_x][ta_y] = x_vec.x;
      tileA[ta_x+1][ta_y] = x_vec.y;
      tileA[ta_x+2][ta_y] = x_vec.z;
      tileA[ta_x+3][ta_y] = x_vec.w;

      FLOAT4(tileB[tb_y][tb_x]) = FLOAT4(b_ptr[(k + tb_y) * N + tb_x]);
      
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
          tileA_BK[j] = tileA[i][ty * TM + j];
        // 分析此时 tileA的 bank conflict
        // block 16 x 16 
        // ty = 0 address = (i * 128) + ty * 8 + j = 0 bank =0 
        // ty = 1 address = 8 bank = 0 广播
        // 完全没有bank conflict  但是写入的时候有bank conflict
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