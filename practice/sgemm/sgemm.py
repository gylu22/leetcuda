import time
from functools import partial
from typing import Optional
import os 
import numpy as np

# 设置架构列表，确保在 import torch 之前
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
# 注意：请确保 sgemm.cu 文件路径正确
lib = load(
    name="sgemm_lib",
    sources=[
        "practice/sgemm/sgemm.cu",
    ],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
    verbose=True, # 编译时显示详细信息，方便调试
)

MAX_TFLOPS = -1

def check_accuracy(custom_out: torch.Tensor, reference_out: torch.Tensor, tag: str, atol: float = 1e-4, rtol: float = 1e-5):
    """
    比较自定义 Kernel 输出与 PyTorch 参考输出的准确性。
    """
    # 确保类型一致（防止半精度转换带来的问题，虽然这里主要是 FP32）
    if custom_out.dtype != reference_out.dtype:
        reference_out = reference_out.to(custom_out.dtype)

    # 计算绝对误差和相对误差
    abs_diff = torch.abs(custom_out - reference_out)
    # 防止除以零，当参考值为0时，只看绝对误差
    rel_diff = abs_diff / (torch.abs(reference_out) + 1e-8)
    
    max_abs_err = torch.max(abs_diff).item()
    max_rel_err = torch.max(rel_diff).item()
    
    # 获取最大误差的位置
    max_err_idx = torch.argmax(abs_diff.flatten())
    max_err_pos = (max_err_idx // custom_out.shape[1], max_err_idx % custom_out.shape[1])
    
    # 判断是否通过
    # 条件：绝对误差 < atol 且 相对误差 < rtol
    passed = torch.all((abs_diff <= atol) | (rel_diff <= rtol)).item()
    
    status = "PASS" if passed else "FAIL"
    color_code = "\033[92m" if passed else "\033[91m" # Green for PASS, Red for FAIL
    reset_code = "\033[0m"
    
    return passed, max_abs_err, max_rel_err, max_err_pos, f"{color_code}{status}{reset_code}"

def run_benchmark(
    perf_func: callable,
    a: torch.Tensor,
    b: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    stages: int = -1,
    swizzle: bool = False,
    swizzle_stride: int = 1,
    warmup: int = 2,
    iters: int = 20,
    show_all: bool = False,
    check_acc: bool = True, # 新增参数控制是否检查精度
):
    global MAX_TFLOPS

    M = a.size(0)
    K = a.size(1)
    N = b.size(1)

    if a.size(0) > 1024 or a.size(1) >= 1024 or b.size(1) > 1024:
        iters = 10

    if swizzle:
        swizzle_stride = int((int(N / 8) // 256) * 256)
        swizzle_stride = swizzle_stride if swizzle_stride >= 256 else 1
        swizzle = swizzle if swizzle_stride >= 256 else False
    else:
        swizzle_stride = 1

    if stages:
        assert swizzle_stride is not None

    # --- 1. 准备参考结果 (PyTorch Native) ---
    reference_out = None
    if check_acc:
        # 使用 torch.matmul 计算标准结果
        reference_out = torch.matmul(a, b)
        # 如果 perf_func 需要预分配的 out 张量，我们需要确保它的形状和参考结果一致
        if out is None:
            out = torch.empty_like(reference_out)
        else:
            # 确保 out 的形状正确
            if out.shape != reference_out.shape:
                out = torch.empty_like(reference_out)

    # 初始化 out (如果需要)
    if out is not None:
        out.fill_(0)

    # --- 2. Warmup ---
    if out is not None:
        for i in range(warmup):
            if stages > 1:
                perf_func(a, b, out, stages, swizzle, swizzle_stride)
            else:
                perf_func(a, b, out)
    else:
        for i in range(warmup):
            _ = perf_func(a, b)

    torch.cuda.synchronize()
    
    # --- 3. Benchmark Timing ---
    start = time.time()
    if out is not None:
        for i in range(iters):
            if stages > 1:
                perf_func(a, b, out, stages, swizzle, swizzle_stride)
            else:
                perf_func(a, b, out)
    else:
        for i in range(iters):
            out = perf_func(a, b)
    torch.cuda.synchronize()
    end = time.time()
    
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    
    # --- 4. Accuracy Check ---
    acc_info = ""
    if check_acc and out is not None and reference_out is not None:
        passed, max_abs, max_rel, err_pos, status_str = check_accuracy(out, reference_out, tag)
        
        # 格式化误差信息
        acc_info = f" | Err: {max_abs:.2e} (rel {max_rel:.2e}) [{status_str}]"
        
        # if not passed:
        #     print(f"\n⚠️  WARNING: Accuracy check failed for {tag}!")
        #     print(f"   Max Abs Error: {max_abs} at index {err_pos}")
        #     print(f"   Custom Value : {out[err_pos[0], err_pos[1]].item()}")
        #     print(f"   Reference Val: {reference_out[err_pos[0], err_pos[1]].item()}")
        #     # 即使失败也继续运行，以便观察性能，但在生产环境中可能需要抛出异常
        #     # raise AssertionError(f"Accuracy check failed for {tag}")

    # --- 5. Performance Stats & Printing ---
    # 获取前几个值用于展示 (从 custom out 获取)
    out_val = out.flatten()[:3].detach().cpu().numpy().tolist()
    out_val = [round(v, 6) for v in out_val] # 减少小数位以便显示
    out_val_str = ", ".join([f"{v:<10}" for v in out_val])

    TFLOPS = (2 * M * N * K) * 1e-9 / (mean_time)
    mean_time_str = f"{mean_time:<8.4f}"
    swizzle_display = "NOOP" if swizzle_stride == 1 else swizzle_stride

    # 计算提升
    improve_str = ""
    if TFLOPS > MAX_TFLOPS:
        if MAX_TFLOPS > 0:
            improve = ((TFLOPS - MAX_TFLOPS) / MAX_TFLOPS) * 100
            improve_str = f"(+{improve:.2f}%)"
        else:
            improve_str = "(NEW)"
        MAX_TFLOPS = TFLOPS

    # 构建最终打印行
    # 格式: Tag : [values], time: X.XX ms, swizzle: Y, TFLOPS: Z.ZZ (+W.WW%) | Err: ... [PASS]
    print(
        f"{tag:>25}: [{out_val_str}], time:{mean_time_str}ms, "
        f"swizzle: {str(swizzle_display):<4}, TFLOPS: {TFLOPS:<6.2f}{improve_str}{acc_info}"
    )

    if show_all:
        print("Custom Output:\n", out)
        print("Reference Output:\n", reference_out)
        
    return out, mean_time

# --- Main Execution ---
if __name__ == "__main__":
    Ms = [4096, 8192] # 为了演示快速，减少了测试规模，你可以改回 [4096, 8192, 16384]
    Ns = [4096, 8192]
    Ks = [2048, 4096]
    
    MAX_M, MAX_N, MAX_K = 16384, 16384, 8192
    
    # pre allocate for fast profiling.
    # 使用固定的 seed 保证每次运行随机数一致，方便复现误差
    torch.manual_seed(42)
    A = torch.randn((MAX_M, MAX_K), dtype=torch.float).cuda()
    B = torch.randn((MAX_K, MAX_N), dtype=torch.float).cuda()
    C = torch.randn((MAX_M, MAX_N), dtype=torch.float).cuda()
    torch.cuda.synchronize()

    MNKs = [(M, N, K) for M in Ms for N in Ns for K in Ks]
    
    for M, N, K in MNKs:
        MAX_TFLOPS = -1
        print("-" * 140)
        print(" " * 60 + f"M={M}, N={N}, K={K}")
        a = A[:M, :K].contiguous()
        b = B[:K, :N].contiguous()
        c = C[:M, :N].contiguous()
        torch.cuda.synchronize()

        # CUDA Cores FP32
        # 确保函数存在，如果 lib 中没有这个符号会报错
        
        run_benchmark(lib.sgemm_naive_f32, a, b, "f32(naive)", c, check_acc=True)
        run_benchmark(lib.sgemm_sliced_k_f32, a, b, "f32(clice)", c, check_acc=True)
       