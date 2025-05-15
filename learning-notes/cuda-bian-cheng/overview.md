---
description: 这一部分来看一下GPU的架构，了解cuda编程的抽象概念
---

# Overview

分层CPU什么是CUDA编程？

> The NVIDIA® CUDA® Toolkit provides a development environment for creating high performance GPU-accelerated applications. With the CUDA Toolkit, you can develop, optimize, and deploy your applications on GPU-accelerated embedded systems, desktop workstations, enterprise data centers, cloud-based platforms and HPC supercomputers. The toolkit includes GPU-accelerated libraries, debugging and optimization tools, a C/C++ compiler, and a runtime library to deploy your application.

这是CUDA toolkit官方对于cuda编程的解释，CUDA是一个专门对GPU-based 加速器进行编程的工具语言，cuda toolkit则包含了对该语言进行编程所需要的各种调试和优化工具，以及编译器和相关的库

### CUDA 编程模型

<figure><img src="../../.gitbook/assets/image.png" alt=""><figcaption><p> CPU arch vs GPU arch</p></figcaption></figure>

CPU 的架构把更多的transistor用来做数据流的控制和data cache（L1，L2 ，L3）而GPU则更多的用于运算

<figure><img src="../../.gitbook/assets/image (1).png" alt=""><figcaption><p>分层架构</p></figcaption></figure>

在CUDA编程的设计中，有3个核心的抽象成分，分别是线程的分层设计，共享内存以及Barrier Sychronization，这三个抽象为cuda编程的scalability和data prallel提供了保证，在GPU的架构中，一个SM（Stream Manager）可以对应任意一个block的线程，因此在实际编程过程中，只要system runner知道有几个SM可以用，就可以对应的把任务拓展到上面去。
