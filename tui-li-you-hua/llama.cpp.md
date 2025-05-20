---
description: 非常著名的跨平台大语言模型部署框架
---

# 😅 llama.cpp

{% embed url="https://zhuanlan.zhihu.com/p/1893801096918585567" %}

这篇文章详细讲解了llama.cpp是如何通过ggml这个模型来构建的计算图，首先通过ggml tensor建立原始计算图的列表，然后将这个列表通过DFS加入到计算图模型中，通过对leaf进行分类可以将计算图划分到不同的后端上面，首先对leaf做一遍dfs把所有leaf节点的hash存储起来，然后根据leaf的weight后端确定之后node的后端，这样就可以在计算中形成天然的划分。

当需要对tensor进行offloading或者loading操作时，计算图会在另外一个后端上部署相似的内存结构，同时记录原始tensor的hash，最后可以look up。

接下来就是llama.cpp的pipeline，schedule 以及quantization和operator等特性，其同步和异步也异常复杂，需要看源码学习。
