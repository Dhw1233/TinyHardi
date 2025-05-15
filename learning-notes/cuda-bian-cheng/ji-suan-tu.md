---
description: 本文内容大部分摘自不同的博客以及官方instruction
---

# 计算图

### launch kernel

在了解计算图之前，先认识一下内核启动的机制

GPU中的Kernel Launch（内核启动）是CUDA编程中的核心机制，用于将计算任务分发到GPU并行执行。

一般是通过cuda的<<<>>>的三尖号启动，其中的参数为所用的线程Grid，block中包含的线程数目，以下是一个简单的demo

```cpp
#include "solve.h"
#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        C[idx] = A[idx] + B[idx];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}

```

这里一个block里面有256个thread，一个Grid有N / 256 个block，使用改语法进行launch

