[← Back to Table of Contents](./README.md)

# 01 — The Grand Overview: From Quantized Model to Hardware Execution

> **The single most important document in this guide.** Read this first.

*Last updated: March 2026*

## 🎯 The Full Stack — One Picture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYER 7: APPLICATION / API                       │
│  OpenAI-compatible API, Chat UI, RAG pipeline, Agent framework      │
├─────────────────────────────────────────────────────────────────────┤
│                    LAYER 6: MIDDLEWARE                               │
│  Load balancer, API gateway, auth, rate limiting, monitoring        │
│  (Envoy, Nginx, Kong, custom)                                       │
├─────────────────────────────────────────────────────────────────────┤
│                    LAYER 5: SERVING INFRASTRUCTURE                  │
│  NVIDIA Triton Inference Server, KServe, BentoML, Ray Serve        │
│  (Model management, scaling, multi-model, A/B testing)              │
├─────────────────────────────────────────────────────────────────────┤
│                    LAYER 4: INFERENCE ENGINE                        │
│  vLLM, SGLang, TensorRT-LLM, llama.cpp, MLC-LLM                   │
│  (Batching, KV-cache, PagedAttention, scheduling, speculative dec.) │
├─────────────────────────────────────────────────────────────────────┤
│                    LAYER 3: RUNTIME / EXECUTION PROVIDER            │
│  ONNX Runtime, libtorch, ExecuTorch, ggml                          │
│  (Graph optimization, op dispatch, memory management)               │
├─────────────────────────────────────────────────────────────────────┤
│                    LAYER 2: COMPILER / KERNEL LIBRARIES              │
│  TVM, XLA, Triton (compiler), MLIR, torch.compile/Inductor         │
│  cuBLAS, CUTLASS, cuDNN, MIOpen, oneDNN, Metal Performance Shaders │
│  (Kernel generation, graph lowering, hardware-specific optimization)│
├─────────────────────────────────────────────────────────────────────┤
│                    LAYER 1: HARDWARE BACKEND / DRIVER               │
│  CUDA (NVIDIA), ROCm/HIP (AMD), SYCL/oneAPI (Intel), Metal (Apple) │
│  Vulkan, OpenCL, proprietary NPU SDKs                               │
├─────────────────────────────────────────────────────────────────────┤
│                    LAYER 0: HARDWARE                                │
│  NVIDIA GPUs (H100/B200/GB200), AMD GPUs (MI300X/MI355X)           │
│  Intel CPUs/GPUs (Xeon/Gaudi), Google TPUs, Apple Silicon           │
│  Qualcomm NPUs, Renesas AI Accelerators, Synopsys ARC NPUs        │
│  AWS Trainium/Inferentia, Groq LPUs, Cerebras WSE                   │
└─────────────────────────────────────────────────────────────────────┘
```

## 🔄 The Journey of YOUR Quantized Model

Here is exactly what happens after you finish developing your AWQ/SmoothQuant model:

### Step 1: You Have a Quantized PyTorch Model
```python
# Your research output — a quantized model
import torch
model = MyLLM.from_pretrained("meta-llama/Llama-3-8B")
quantized_model = apply_awq(model, calibration_data)
# quantized_model is a torch.nn.Module with INT4 weights
# It works in PyTorch. Now what?
```

### Step 2: Export / Serialize to a Deployment Format

Your PyTorch model must be converted into a format that deployment tools understand:

```
PyTorch nn.Module
       │
       ├──► GGUF format ──────────► llama.cpp
       ├──► ONNX format ──────────► ONNX Runtime
       ├──► SafeTensors ───────────► vLLM / SGLang / HuggingFace
       ├──► TensorRT Engine ──────► TensorRT-LLM
       ├──► torch.export() ───────► ExecuTorch (edge)
       ├──► TFLite flatbuffer ────► TensorFlow Lite (edge)
       └──► Proprietary format ───► Vendor-specific NPU SDK
```

**Key insight**: The format must preserve your quantization metadata (scales, zero-points, group size, etc.)

### Step 3: The Inference Engine Takes Over

The inference engine handles the *LLM-specific* optimizations:

```
┌─────────────── Inference Engine Responsibilities ───────────────┐
│                                                                  │
│  ● Continuous Batching    — serve many users simultaneously      │
│  ● KV-Cache Management   — PagedAttention (like virtual memory)  │
│  ● Scheduling             — which requests to process when       │
│  ● Speculative Decoding   — predict multiple tokens at once      │
│  ● Tensor/Expert Parallel — split model across GPUs              │
│  ● Prefix Caching         — reuse KV-cache for shared prompts    │
│  ● Quantized Execution    — dispatch to INT4/INT8/FP8 kernels    │
│  ● Structured Output      — JSON/grammar-constrained generation  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Step 4: Runtime Dispatches to Hardware-Specific Code

```
Inference Engine says: "Execute this INT4 matmul"
         │
         ▼
┌─── Runtime Layer ───┐
│                      │
│  "Which hardware?"   │──► NVIDIA GPU → cuBLAS / CUTLASS INT4 kernel
│                      │──► AMD GPU    → MIOpen / hipBLAS kernel
│                      │──► Intel CPU  → oneDNN / AMX INT8 kernel
│                      │──► Apple M-series → Metal + Accelerate
│                      │──► Google TPU → XLA compiled kernel
│                      │──► Edge NPU  → Vendor SDK kernel
│                      │
└──────────────────────┘
```

### Step 5: Hardware Executes

The actual silicon runs the computation. Your INT4 weights are loaded into
tensor cores (NVIDIA), matrix engines (AMD), AMX blocks (Intel), or NPU
compute elements, and the matmuls happen.

### Step 6: Serving Infrastructure Routes the Response

```
Hardware outputs logits
    │
    ▼
Inference engine does sampling (temperature, top-k, top-p)
    │
    ▼
Serving layer streams tokens back via SSE/WebSocket
    │
    ▼
Middleware handles auth, rate limiting, logging
    │
    ▼
Client receives: "The answer is 42"
```

## 📊 Where Each Provider Fits in the Stack

| Provider | Layer(s) | What It Does |
|----------|----------|-------------|
| **vLLM** | 4 (Engine) | High-throughput LLM serving with PagedAttention |
| **SGLang** | 4 (Engine) | Fast LLM serving with RadixAttention, structured output |
| **llama.cpp** | 3+4 (Runtime+Engine) | CPU-first LLM inference in C/C++, uses ggml runtime |
| **TensorRT-LLM** | 2+3+4 (Compiler+Runtime+Engine) | NVIDIA's optimized LLM inference stack |
| **ONNX Runtime** | 3 (Runtime) | Cross-platform model execution with execution providers |
| **Triton Inference Server** | 5 (Serving) | Multi-model serving, load balancing, model management |
| **CUDA / cuBLAS** | 1+2 (Driver+Kernel) | NVIDIA's compute platform and linear algebra library |
| **ROCm / HIP** | 1+2 (Driver+Kernel) | AMD's GPU compute platform |
| **OpenVINO** | 2+3 (Compiler+Runtime) | Intel's inference optimization toolkit |
| **ExecuTorch** | 3 (Runtime) | Meta's on-device inference runtime |
| **MLC-LLM** | 2+3+4 (Compiler+Runtime+Engine) | Universal deployment via TVM compiler |
| **XLA** | 2 (Compiler) | Google's compiler for TPUs and GPUs |
| **torch.compile** | 2 (Compiler) | PyTorch's JIT compiler using TorchInductor |

## 🔗 The Relationship Graph

```
                        ┌──────────────┐
                        │  Your Model  │
                        │  (Quantized) │
                        └──────┬───────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
      ┌──────────────┐ ┌────────────┐ ┌──────────────┐
      │  SafeTensors │ │    GGUF    │ │     ONNX     │
      │   (Format)   │ │  (Format)  │ │   (Format)   │
      └──────┬───────┘ └─────┬──────┘ └──────┬───────┘
             │               │               │
     ┌───────┴────┐          │         ┌─────┴──────┐
     │            │          │         │            │
     ▼            ▼          ▼         ▼            ▼
 ┌───────┐  ┌────────┐  ┌────────┐ ┌──────┐  ┌─────────┐
 │ vLLM  │  │SGLang  │  │llama   │ │ONNX  │  │OpenVINO │
 │       │  │        │  │.cpp    │ │RT    │  │         │
 └───┬───┘  └───┬────┘  └───┬────┘ └──┬───┘  └────┬────┘
     │          │            │         │           │
     ▼          ▼            ▼         ▼           ▼
 ┌────────────────────────────────────────────────────────┐
 │              Hardware Backends                          │
 │  CUDA │ ROCm │ Metal │ Vulkan │ SYCL │ CPU │ TPU     │
 └────────────────────────────────────────────────────────┘
     │          │         │         │        │       │
     ▼          ▼         ▼         ▼        ▼       ▼
 ┌────────┐┌───────┐┌────────┐┌──────┐┌───────┐┌──────┐
 │NVIDIA  ││ AMD   ││ Apple  ││ Any  ││ Intel ││Google│
 │GPU     ││ GPU   ││Silicon ││ GPU  ││CPU/GPU││ TPU  │
 └────────┘└───────┘└────────┘└──────┘└───────┘└──────┘
```

## ⚡ Key Insight: Why This Is Confusing

The confusion comes from the fact that **boundaries between layers are blurry**:

1. **llama.cpp** is *both* a runtime (ggml) *and* an inference engine
2. **TensorRT-LLM** spans compiler, runtime, and engine layers
3. **vLLM** is an engine but directly calls CUDA kernels (bypassing a separate runtime)
4. **Some "runtimes" include serving** (ONNX Runtime Server)
5. **Some "serving frameworks" include inference** (Triton + TensorRT-LLM)

**The terms are not used consistently across the industry.** This guide gives you a mental model that works regardless.

## 📈 Current Landscape Trends (March 2026)

| Trend | Details |
|-------|---------|
| **PyTorch-native inference** | TensorRT-LLM rebuilt on PyTorch; vLLM uses torch.compile |
| **FP4/FP8 quantization** | NVIDIA Blackwell native FP4 support; industry moving beyond INT4/INT8 |
| **Disaggregated serving** | Separating prefill and decode onto different hardware |
| **Speculative decoding everywhere** | EAGLE, Medusa, MTP — all major engines support it |
| **Edge AI explosion** | ExecuTorch, llama.cpp on phones, NPU-specific compilers |
| **MoE model dominance** | Expert parallelism becoming a first-class concern |
| **Open-source convergence** | vLLM + SGLang approaching feature parity; TensorRT-LLM open-sourced |
| **Multi-hardware support** | Every engine expanding beyond NVIDIA-only |

---

**Next**: [02 — Glossary & Concept Map →](./02_glossary_and_concept_map.md)
