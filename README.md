# 🧠 From Quantized Model to Production: The Complete Inference & Deployment Guide

> **Audience**: AI research scientists who develop quantization/optimization algorithms (AWQ, SmoothQuant, GPTQ, etc.) and want to understand what happens *after* the model is optimized — all the way to hardware execution.

## 📋 Table of Contents

| # | Document | What You'll Learn |
|---|----------|-------------------|
| 1 | [Grand Overview](./01_grand_overview.md) | The full stack from PyTorch model → hardware execution, the "big picture" |
| 2 | [Glossary & Concept Map](./02_glossary_and_concept_map.md) | Every term defined + visual relationship maps between concepts |
| 3 | [Inference Engines Deep Dive](./03_inference_engines.md) | vLLM, SGLang, llama.cpp, TensorRT-LLM — compared in depth |
| 4 | [Runtimes & Backends](./04_runtimes_and_backends.md) | ONNX Runtime, CUDA, ROCm, OpenVINO, ExecuTorch — what are they? |
| 5 | [Hardware Landscape](./05_hardware_landscape.md) | NVIDIA GPUs, AMD GPUs/CPUs, Intel, Google TPUs, NPUs, custom ASICs |
| 6 | [Quantization to Deployment Bridge](./06_quantization_to_deployment.md) | How YOUR quantized model actually gets deployed end-to-end |
| 7 | [Company Landscape](./07_company_landscape.md) | What NVIDIA, Renesas, Synopsys, Qualcomm, etc. are building |
| 8 | [Model Formats & Serialization](./08_model_formats.md) | GGUF, ONNX, SafeTensors, TorchScript — why so many formats? |
| 9 | [Serving Infrastructure & Middleware](./09_serving_and_middleware.md) | Triton Inference Server, KServe, API gateways, load balancers |
| 10 | [Edge & Embedded Deployment](./10_edge_and_embedded.md) | TFLite, ExecuTorch, microcontrollers, Renesas RA/RZ MCUs |
| 11 | [Compiler Stack](./11_compiler_stack.md) | TVM, XLA, Triton (compiler), MLIR — the crucial missing piece |
| 12 | [Practical Code Guide](./12_practical_code_guide.md) | End-to-end code: PyTorch → quantize → export → deploy → serve |

## 🎯 The Core Question This Guide Answers

```
You've developed an AWQ/SmoothQuant/GPTQ quantized model in PyTorch.
It works. Accuracy is great. Now what?

    ┌─────────────────────────────────────────────────────┐
    │                YOUR QUANTIZED MODEL                  │
    │            (PyTorch nn.Module, FP16/INT4/INT8)       │
    └──────────────────────┬──────────────────────────────┘
                           │
                    What happens here?
                           │
                           ▼
    ┌─────────────────────────────────────────────────────┐
    │              ??? MAGIC BLACK BOX ???                  │
    │                                                      │
    │   Export? → Format? → Runtime? → Backend? →          │
    │   Inference Engine? → Serving? → Middleware? →       │
    │   Hardware?                                          │
    └──────────────────────┬──────────────────────────────┘
                           │
                           ▼
    ┌─────────────────────────────────────────────────────┐
    │           PRODUCTION: Tokens coming out!              │
    │     (Running on NVIDIA H100 / AMD MI300 / Edge)      │
    └─────────────────────────────────────────────────────┘
```

**This guide demystifies that "magic black box."**

## 🧩 Essential Topics You Might Be Missing

As a quantization researcher, here are concepts that are **critical** but often overlooked:

| Topic | Why It Matters to You |
|-------|----------------------|
| **Compiler Stack (TVM/XLA/Triton/MLIR)** | Your quantized ops need *kernel-level* support — compilers generate those kernels |
| **Model Serialization Formats** | Your INT4 weights need a container format that *preserves quantization metadata* |
| **Graph Optimization** | Runtimes fuse/optimize ops — some fusions *break* quantization boundaries |
| **Kernel Libraries (cuBLAS, CUTLASS, cuDNN)** | Your W4A16 matmul only runs fast if there's a *kernel* implementing it on hardware |
| **Execution Providers** | ONNX Runtime's abstraction for different hardware — your quant format must be supported |
| **Serving vs. Inference** | Inference = running the model. Serving = batching, routing, scaling, API |
| **Middleware** | The layer between your model and the network — load balancing, auth, monitoring |
| **Disaggregated Serving** | Prefill vs. decode on different hardware — quantization interacts differently |
| **Edge Compilers** | Deploying to Renesas/Qualcomm/Synopsys NPUs requires custom compilation pipelines |
| **Speculative Decoding** | Interacts with quantization — draft models can be differently quantized |

## 📚 Prerequisites

Before diving in, you should be comfortable with:
- **PyTorch basics** — `nn.Module`, tensors, `forward()`, GPU training
- **What quantization is** — reducing weight/activation precision (FP16 → INT8/INT4)
- **Basic GPU concepts** — CUDA cores, memory bandwidth, HBM vs SRAM
- **LLM fundamentals** — transformers, attention, autoregressive generation

If you need a refresher on quantization itself (not deployment), see the original papers for [AWQ](https://arxiv.org/abs/2306.00978), [GPTQ](https://arxiv.org/abs/2210.17323), and [SmoothQuant](https://arxiv.org/abs/2211.10438).

## 🏗️ How to Read This Guide

1. **Start with [01_grand_overview.md](./01_grand_overview.md)** — get the full picture
2. **Read [02_glossary_and_concept_map.md](./02_glossary_and_concept_map.md)** — anchor every term
3. **Then dive into whichever topic interests you most**
4. **Use [12_practical_code_guide.md](./12_practical_code_guide.md)** for hands-on examples

### ⚡ Quick Start Path (3-document fast track)

If you're short on time and just want to understand the end-to-end flow:

1. [01 — Grand Overview](./01_grand_overview.md) — the full stack in one picture
2. [06 — Quantization to Deployment Bridge](./06_quantization_to_deployment.md) — how *your* research connects to production
3. [12 — Practical Code Guide](./12_practical_code_guide.md) — copy-paste code for every deployment path

## 📝 Changelog

| Date | Changes |
|------|---------|
| March 2026 | Initial release — all 12 documents |

> **Note**: Each document includes a "Last updated" date. Given how fast this field moves, always check dates and verify version-specific claims against current documentation.

---
*Last updated: March 2026*
