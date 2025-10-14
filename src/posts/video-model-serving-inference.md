---
title: Serving DiTs/RFTs at Scale
date: 2025-10-13
description: Designing the next generation of video model serving systems.
references:
    - 'Monas & Jang. ["https://www.1x.tech/discover/1x-world-model"](https://www.1x.tech/discover/1x-world-model). 2024.'
    - 'Noam Shazeer. ["Fast Transformer Decoding: One Write-Head is All You Need"](https://arxiv.org/abs/1911.02150). 2019.'
    - 'Sun et al. ["You Only Cache Once: Decoder-Decoder Architectures for Language Models"](https://arxiv.org/abs/2405.05254). 2024.'
    - 'Pope et al. ["Efficiently Scaling Transformer Inference"](https://arxiv.org/pdf/2211.05102). 2022.'
    - 'Leviathan et al. ["Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192). 2023.'
    - 'Qin et al. ["Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving"](https://arxiv.org/abs/2407.00079). 2025.'
    - 'Jouppi et al. ["In-Datacenter Performance Analysis of a Tensor Processing Unit"](https://arxiv.org/abs/1704.04760). 2017.'
    - 'Salimans & Ho. ["Progressive Distillation for Fast Sampling of Diffusion Models"](https://arxiv.org/abs/2202.00512). 2022.'
    - 'Ho & Salimans. ["Classifier-Free Diffusion Guidance"](https://arxiv.org/abs/2207.12598). 2022.'
    - 'Meng et al. ["On Distillation of Guided Diffusion Models"](https://arxiv.org/abs/2210.03142). 2023.'
    - 'Peebles & Xie. ["Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748). 2023.'
    - 'Esser et al. ["Scaling Rectified Flow Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2403.03206). 2024.'
    - 'Liu et al. ["Timestep Embedding Tells: It''s Time to Cache for Video Diffusion Model"](https://arxiv.org/abs/2411.19108). 2025.'
    - 'Xi et al. ["Sparse VideoGen: Accelerating Video Diffusion Transformers with Spatial-Temporal Sparsity"](https://arxiv.org/abs/2502.01776). 2025.'
    - 'Xia et al. ["Training-free and Adaptive Sparse Attention for Efficient Long Video Generation"](https://arxiv.org/abs/2502.21079). 2025.'
    - 'Ho et al. ["Cascaded Diffusion Models for High Fidelity Image Generation"](https://arxiv.org/abs/2106.15282). 2021.'
    - 'Ho et al. ["Imagen Video: High Definition Video Generation with Diffusion Models"](https://arxiv.org/abs/2210.02303). 2022.'
    - 'Zhang et al. ["SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration"](https://arxiv.org/abs/2410.02367). 2025.'
    - 'Rachel Xin. ["DiT-Serve and DeepCoder: Enabling Video and Code Generation at Scale"](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-46.html). 2025.'
    - 'Zhong et al. ["DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving"](https://arxiv.org/abs/2401.09670). 2024.'
    - 'Ma et al. ["dInfer: An Efficient Inference Framework for Diffusion Language Models"](https://arxiv.org/abs/2510.08666). 2025.'
    - 'HuggingFace Team. ["Stable Video Diffusion"](https://huggingface.co/docs/diffusers/en/using-diffusers/svd).'
---


**This post assumes the reader has basic familiarity with diffusion models and LLM serving.**


## Introduction

I stumbled across [this tweet](https://x.com/TheInclusionAI/status/1977653483353559105) recently.

Everyone is thinking about language model inference and serving [insert source here], but what about video models? The [release of Sora 2](https://openai.com/index/sora-2/) marks a paradigm shift, redefining content creation as an industry and throwing generative video models into the limelight.

**Video models are the future - and will come to dominate over the next decade**, aside from in generative media ("AI slop").

The clear use case is in world models - action-conditioned video generation models [1]. These models learn the complexities of the real world and can envision several futures based on different possible actions. The use of these models will allow us to scale up robotics simulation and evaluation, paving the way towards general purpose intelligence robots.

![1X world model](../assets/images/world_model)
*Robotics world model. source: [1]*

The [Sora app hit 1 million downloads in less than 5 days](https://x.com/billpeeb/status/1976099194407616641), even faster than ChatGPT did. What happens at 10 million? 100 million? **Infrastructure for serving video models at scale is in its infancy, and approaches will continue to evolve as we bridge the gap between current and future demand.**

Massive effort and resources have been poured into efforts to serve autoregressive transformer-based models at scale. There has been work on model research ([2], [3]), research-engineering for serving ([4], [5]), engineering platforms for deployment ([6]), and custom hardware ([7]). Inference engines like SGLang have been co-designed specifically for next-token prediction, squeezing out performance through techniques like continuous batching, KV caching, prefix (radix) caching, PD disaggregation, and lower-level kernel optimizations via libraries like FlashInfer.

However, at the time of writing, there is a limited amount of publicly available resources on how to think about video model serving. In this post, we'll look at existing works to understand where we are today, and consider what the future of serving video models at scale might look like.


## Model Optimizations
### Step Distillation
In early 2022, Tim Salimans and Jonathan Ho proposed the idea of progressive distillation [8]. The algorithm starts with a teacher model with a large number of sampling steps (the paper's experiments use 1024/8192). Iteratively, the model is distilled into a "faster" student model with half the number of required sampling steps, eventually reaching as few as 4 steps. Each time, the student model learns to predict the output of two steps of the teacher model.

![Progressive distillation example](../assets/images/progressive_distillation.png)
*Figure 1. Progressive distillation example, 4-step sampler -> 1-step sampler. source: [8]*

The idea is fairly intuitive, however, readers are still encouraged to check out the paper to understand the proposed alternative parameterizations ($x$, $\epsilon$, and $v$ prediction) and underlying math.

### CFG Distillation
Classifier-free guidance (CFG) is a technique proposed to increase sample quality at the cost of diversity [9]. By specifying a guidance weight ($w$), we can interpolate ($w \lt 1$) between the conditional and unconditional distributions - and extrapolate ($w \gt 1$) to push generations further towards the conditioning signal.

![CFG example](../assets/images/cfg_different_weights.png)
*Figure 2. CFG with different weights on SDXL.*

During training, conditioning (e.g. text prompt) is randomly dropped with some probability (typically 10% of the time). This allows the model to learn both conditional (prompt-conditioned) and unconditional ("null prompt") behavior in the same network. At inference, we evaluate the model twice per diffusion step: once with the prompt and once with the null prompt. This provides both the conditional and unconditional predictions, which are then combined using $\hat{\epsilon} = \epsilon_{\text{uncond}} + w \cdot (\epsilon_{\text{cond}} - \epsilon_{\text{uncond}})$ (reparameterization of Eq. 6 from the paper) where $\epsilon_{\text{cond}}$ and $\epsilon_{\text{uncond}}$ are the noise predictions for the conditional and unconditional inputs, respectively.

![CFG math example](../assets/images/cfg_math.png)
*Figure 3. CFG-guided logits. The guided logits are pushed away from the unconditional.*

Running the model twice per step is computationally expensive, so practical implementations often run it once per step with double the effective batch size to compute both conditional and unconditional outputs in parallel. However, this still increases memory usage (larger activations) and latency (typically already compute-bound). Ideally, we would like a model that directly predicts the guided output.

Meng et al. (2023) proposed and validated this idea, demonstrating its compability with progressive distillation [10]. The key tradeoff is that the distilled model is tied to a specific guidance weight $w$ used during training, losing flexibility at inference time. In practice, this is acceptable since we typically use a single fixed guidance weight (usually in the range of 5-8).

### Timestep Caching

### Attention Sparsity

### Refiner


### Activation Quantization


## Serving Optimizations
This section is more experimental than previous sections. The discussion will be based off DiT-Serve and conventional LLM serving ideas.


### (Continuous) Batching

### Model Compilation and CUDA Graphs

### VAE Chunking
