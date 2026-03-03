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