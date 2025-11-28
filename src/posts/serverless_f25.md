---
title: Serverless AI
date: 2025-11-27
description: Highlights of a UVA F25 course.
references:
    - 'Kwon et al. ["Efficient Memory Management for Large Language Model Serving with PagedAttention"](https://dl.acm.org/doi/pdf/10.1145/3600006.3613165). 2023.'
    - 'Yu et al. ["Orca: A Distributed Serving System for Transformer-Based Generative Models"](https://www.usenix.org/system/files/osdi22-yu.pdf). 2022.'
    - 'Zhong et al. ["DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving"](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf). 2024.'
    - 'Agrawal et al. ["Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve"](https://www.usenix.org/system/files/osdi24-agrawal.pdf). 2024.'
    - 'Qin et al. ["Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot"](https://www.usenix.org/system/files/fast25-qin.pdf). 2025.'
    - 'Zhang et al. ["BlitzScale: Fast and Live Large Model Autoscaling with O(1) Host Caching"](https://www.usenix.org/system/files/osdi25-zhang-dingyan.pdf). 2025.'
    - 'Pu et al. ["Shuffling, Fast and Slow: Scalable Analytics on Serverless Infrastructure"](https://www.usenix.org/system/files/nsdi19-pu.pdf). 2019.'

---

Shoutout to [Professor Yue Cheng](https://tddg.github.io/) for a great Fall semester!

This course was structured similarly to most graduate courses at UVA—a mix of paper readings, presentations, and a course project. I was most interested in the LLM serving component, but also enjoyed discussions around cloud computing, FaaS, and computing hardware. I've highlighted some of my favorite serving readings from the semester below.

### PagedAttention [1]

From Stoica's lab at Berkeley, this work introduces [vLLM](https://github.com/vllm-project/vllm), an open-sourced inference engine. The authors find that previous KV cache memory management systems lead to internal (reserved, unused memory) and external (free, non-contiguous memory) fragmentation. Inspired by traditional OS, they propose PagedAttention, which allocates fixed-size KV blocks similar to pages. This naturally extends to support [prefix caching](https://docs.vllm.ai/en/v0.9.2/features/automatic_prefix_caching.html).

### Orca [2]

From SNU, this work introduces iteration-level scheduling and selective batching. As opposed to previous methods, which waits for all requests in a batch to complete, [continuous batching](https://www.anyscale.com/blog/continuous-batching-llm-inference) schedules new requests as soon as any previous request completes, improving serving latency and throughput.

### DistServe [3]

From the Hao AI lab at UCSD, this work introduces prefill-decode (PD) disaggregation. The authors identify that prefill and decode workloads have very different characteristics; decode steps should run with larger batch sizes due to their memory-bound nature, while prefill steps are compute-bound. They show that placing prefill and decode instances on separate GPUs maximizes per-GPU goodput.

### Sarathi-Serve [4]

From Microsoft, this work introduces [chunked-prefills](https://docs.vllm.ai/en/v0.4.2/models/performance.html), which splits prefill requests into smaller chunks. Given the compute-bound nature of prefill, this allows for prefills to be jointly scheduled with decodes without a significant latency penalty.

### MoonCake [5]

From Moonshot AI, this work details the serving platform for [Moonshot's Kimi](https://www.kimi.com/). The authors propose a multi-level [KV cache pool](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/feature_guide/KV_Cache_Pool_Guide.html) for PD disaggregated architectures, which utilizes [Remote Direct Memory Access (RDMA)](https://developer.nvidia.com/gpudirect) for fast transfers.

### BlitzScale [6]

From SJTU, this work proposes faster GPU autoscaling. Instead of loading models from slow network storage, the authors leverage RDMA for the case of deploying new instances of an existing, live model. This works well with online RL workloads, where checkpoints are updated and deployed [every 90 minutes](https://cursor.com/blog/tab-rl). Lequn Chen's [blog on inter-node RL weight transfer](https://le.qun.ch/en/blog/2025/09/07/rl-weight-transfer/) is also an exceptional resource.

### Bonus: Shuffling, Fast and Slow [7]

Not a serving work, but interesting nonetheless. Also from Stoica's lab, this work introduces a hybrid sorting method which combines cheap, slow (S3) with fast, expensive (Redis) storage to trade off cost and performance.
