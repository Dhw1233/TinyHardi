---
description: 投机解码，推优化的重要技术
---

# spec decoding

<figure><img src="../.gitbook/assets/image (2).png" alt=""><figcaption><p>算法流程</p></figcaption></figure>

所谓投机解码，就是让一个能与相同结构的大模型生成相同输出分布的自回归小模型代替decoding过程，让这个小模型生成k个token，同时让大模型并行的对这k个token产生的logit进行验证，这样，可以把一次大模型decode进行batch，进行多个token的验证，提高计算访存比。使用一次大模型decode + N 次小模型decode代替 k 个小模型生成的token，从而提升系统的throughput。本质上是把计算时间转化成验证时间。
