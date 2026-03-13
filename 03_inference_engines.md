[← Back to Table of Contents](./README.md)

# 03 — Inference Engines Deep Dive

> The most important layer for a quantization researcher: where your quantized model actually runs.

*Last updated: March 2026*

## 🎯 What Is an Inference Engine?

An inference engine is the **factory manager** for LLM inference. It doesn't just run the model — it orchestrates *everything*:

```
┌────────────────────────── Inference Engine ──────────────────────────┐
│                                                                      │
│  Request Queue ──► Scheduler ──► Batch Assembly ──► Model Execution  │
│       │                │              │                    │         │
│       │                │              │                    ▼         │
│       │                │              │           ┌──────────────┐   │
│       │                │              │           │ KV-Cache     │   │
│       │                │              │           │ Manager      │   │
│       │                │              │           │ (Paged       │   │
│       │                │              │           │  Attention)  │   │
│       │                │              │           └──────────────┘   │
│       │                │              │                    │         │
│       │                │              │                    ▼         │
│       │                │              │           ┌──────────────┐   │
│       │                │              │           │ Sampling     │   │
│       │                │              │           │ (temp, top-k │   │
│       │                │              │           │  top-p)      │   │
│       │                │              │           └──────────────┘   │
│       │                │              │                    │         │
│       ◄────────────────┴──────────────┴────────────────────┘         │
│                     Continuous Batching Loop                         │
└──────────────────────────────────────────────────────────────────────┘
```

## 📊 The Big Comparison Table

| Feature | **vLLM** | **SGLang** | **TensorRT-LLM** | **llama.cpp** | **MLC-LLM** |
|---------|----------|-----------|-------------------|--------------|-------------|
| **Language** | Python | Python | Python + C++ | C/C++ | Python + C++ |
| **Primary Target** | Cloud GPU serving | Cloud GPU serving | NVIDIA GPU serving | Local/edge/CPU | Universal deployment |
| **GitHub Stars** (Mar 2026) | ~50k | ~24k | ~13k | ~96k | ~20k |
| **Backed By** | UC Berkeley / vLLM Org | LMSYS / UC Berkeley | NVIDIA | ggml.ai community | CMU / OctoAI |
| **License** | Apache 2.0 | Apache 2.0 | Apache 2.0 | MIT | Apache 2.0 |
| | | | | | |
| **NVIDIA GPU** | ✅ Excellent | ✅ Excellent | ✅ Best-in-class | ✅ Good (CUDA) | ✅ Good |
| **AMD GPU** | ✅ ROCm | ✅ ROCm | ❌ | ✅ HIP | ✅ ROCm/Vulkan |
| **Intel CPU/GPU** | ✅ CPU + XPU | ✅ CPU | ❌ | ✅ (SYCL) | ✅ |
| **Apple Silicon** | ❌ | ❌ | ❌ | ✅ Metal (excellent) | ✅ Metal |
| **Google TPU** | ✅ (via TPU project) | ✅ (JAX backend) | ❌ | ❌ | ❌ |
| **ARM CPU** | ✅ | ✅ | ❌ | ✅ (NEON) | ✅ |
| **Vulkan** | ❌ | ❌ | ❌ | ✅ | ✅ |
| | | | | | |
| **PagedAttention** | ✅ (inventor) | ✅ | ✅ | ❌ | ❌ |
| **RadixAttention** | ❌ | ✅ (inventor) | ❌ | ❌ | ❌ |
| **Continuous Batching** | ✅ | ✅ | ✅ (inflight batching) | ✅ (server mode) | ✅ |
| **Speculative Decoding** | ✅ EAGLE/MTP/Medusa | ✅ EAGLE/MTP | ✅ EAGLE/MTP/Draft | ⚠️ Basic (draft model) | ✅ |
| **Disaggregated Prefill** | ✅ | ✅ | ✅ | ❌ | ❌ |
| **Structured Output** | ✅ | ✅ (excellent) | ✅ | ✅ (grammars) | ✅ |
| **Multi-LoRA** | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Expert Parallelism** | ✅ | ✅ | ✅ | ❌ | ❌ |
| **Tensor Parallelism** | ✅ | ✅ | ✅ | ❌ (RPC) | ✅ |
| **Pipeline Parallelism** | ✅ | ✅ | ✅ | ❌ | ❌ |
| **Data Parallelism** | ✅ | ✅ | ✅ | ❌ | ❌ |
| | | | | | |
| **INT4 (AWQ)** | ✅ | ✅ | ✅ | ✅ (GGUF Q4) | ✅ |
| **INT4 (GPTQ)** | ✅ | ✅ | ✅ | ❌ | ✅ |
| **INT8 (SmoothQuant)** | ✅ | ✅ | ✅ | ✅ (Q8_0) | ✅ |
| **FP8** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **FP4** | ✅ | ✅ | ✅ (native Blackwell) | ✅ (MXFP4) | ❌ |
| **GGUF Quants** | ✅ | ❌ | ❌ | ✅ (native) | ❌ |
| **BitsAndBytes** | ✅ | ✅ | ❌ | ❌ | ❌ |
| **Quantized KV-Cache** | ✅ | ✅ | ✅ | ✅ | ✅ |
| | | | | | |
| **OpenAI-compat API** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Model Format** | SafeTensors/HF | SafeTensors/HF | SafeTensors/HF | GGUF | MLC compiled |
| **Max Model Count** | 700+ architectures | 700+ architectures | 100+ architectures | 200+ GGUF models | 100+ |

## 🔬 Deep Dive: Each Engine

### Understanding Prefill vs Decode (Critical for Quantization)

All inference engines manage two distinct phases — and quantization affects each differently:

```
┌─────── PREFILL ──────────────────────── DECODE ──────────────────┐
│                                          │                       │
│  Process entire prompt at once           │  Generate 1 token at  │
│  (parallel matrix multiplications)       │  a time (sequential)  │
│                                          │                       │
│  COMPUTE-BOUND                           │  MEMORY-BANDWIDTH     │
│  Bottleneck: Tensor Core throughput      │  BOUND                │
│                                          │  Bottleneck: HBM read │
│                                          │  speed (read all      │
│                                          │  weights per token)   │
│                                          │                       │
│  Quantization helps via:                 │  Quantization helps   │
│  • More ops/sec at lower precision       │  via:                 │
│    (FP8 = 2× FP16 on tensor cores)     │  • Fewer bytes to       │
│  • But only if compute is bottleneck     │    read per token     │
│                                          │  • INT4 = 4× less     │
│                                          │    bandwidth than FP16│
│                                          │                       │
│  Metric: TTFT (Time to First Token)     │  Metric: TPOT (Time    │
│                                          │  Per Output Token)    │
│                                          │                       │
│  Disaggregated serving runs prefill      │  ...and decode on     │
│  on compute-optimized GPUs...            │  bandwidth-optimized  │
│                                          │  GPUs separately      │
└──────────────────────────────────────────┴────────────────────────┘
```

> **Key insight**: W4A16 (4-bit weights, 16-bit activations) is so popular because it directly targets the **decode bottleneck** — fewer bytes to read from HBM per token — without the difficulty of quantizing activations. See [02 — Glossary](./02_glossary_and_concept_map.md#-key-concepts-for-quantization-researchers) for more on attention variants (MHA/GQA/MQA) that also affect memory.

### Attention Variants Impact on Engines

Modern LLMs use different attention patterns that affect KV-cache size:

| Attention Type | KV Heads | Used By | KV-Cache Impact | Engine Support |
|----------------|----------|---------|-----------------|----------------|
| **MHA** (Multi-Head) | Same as Q heads | GPT-2, older models | Largest KV-cache | All engines |
| **GQA** (Grouped-Query) | Fewer than Q heads | Llama 3, Mistral, Gemma | 4-8× smaller KV-cache | All engines |
| **MQA** (Multi-Query) | 1 | PaLM, Falcon, StarCoder | Smallest KV-cache | All engines |

All major engines (vLLM, SGLang, TRT-LLM, llama.cpp) support all three. GQA is now dominant — it reduces KV-cache memory, meaning **more room for batching** and **less need for aggressive KV-cache quantization**.

### vLLM — The Industry Standard

```
┌─────────────────────── vLLM Architecture ──────────────────────┐
│                                                                │
│  OpenAI-Compatible API Server                                  │
│       │                                                        │
│       ▼                                                        │
│  AsyncLLMEngine                                                │
│       │                                                        │
│       ├──► Scheduler (continuous batching)                     │
│       │       │                                                │
│       │       ▼                                                │
│       ├──► KV Cache Manager (PagedAttention)                   │
│       │       │                                                │
│       │       ▼                                                │
│       ├──► Model Executor                                      │
│       │       │                                                │
│       │       ├──► PyTorch model (w/ custom attention backends)│
│       │       ├──► torch.compile (optional, via Inductor)      │
│       │       └──► Custom CUDA kernels (FlashAttention, etc.)  │
│       │                                                        │
│       └──► Sampler (temperature, top-k, top-p, penalties)      │
│                                                                │
│  Hardware: CUDA (primary), ROCm, CPU, XPU, TPU                 │
└─────────────────────────────────────────────────────────────────┘
```

**Why quantization researchers should care about vLLM:**
- Directly supports AWQ, GPTQ, SmoothQuant, FP8, FP4, BitsAndBytes
- Uses Marlin/Machete/CUTLASS kernels for quantized matmuls
- Adding a new quantization format means implementing a `QuantizationConfig` and corresponding linear layer
- Most likely deployment target for your research

**Quick Start:**
```python
from vllm import LLM, SamplingParams

# Load AWQ quantized model
llm = LLM(model="TheBloke/Llama-2-7B-AWQ", quantization="awq")

# Generate
outputs = llm.generate(["Hello, world!"], SamplingParams(temperature=0.7))
```

### SGLang — The Performance Challenger

```
┌─────────────────────── SGLang Architecture ────────────────────┐
│                                                                │
│  OpenAI-Compatible API                                         │
│       │                                                        │
│       ▼                                                        │
│  TokenizerManager ──► DetokenizerManager                       │
│       │                                                        │
│       ▼                                                        │
│  Scheduler (zero-overhead CPU scheduler)                       │
│       │                                                        │
│       ├──► RadixAttention (prefix tree-based KV-cache sharing) │
│       │                                                        │
│       ├──► TpModelWorker                                       │
│       │       │                                                │
│       │       └──► PyTorch model + FlashInfer attention        │
│       │                                                        │
│       └──► Constrained Decoding (grammar/regex/JSON)           │
│                                                                │
│  Hardware: CUDA (primary), ROCm, CPU, TPU (JAX)                │
└─────────────────────────────────────────────────────────────────┘
```

**Why quantization researchers should care about SGLang:**
- RadixAttention enables better prefix caching — quantization interacts with cache efficiency
- Zero-overhead scheduler → latency-sensitive to quantization kernel speed
- Strong structured output support — constrained decoding on quantized models
- Fastest-growing community (400k+ GPUs deployed)

### TensorRT-LLM — NVIDIA's Full-Stack Solution

```
┌──────────────────── TensorRT-LLM Architecture ─────────────────┐
│                                                                │
│  Python LLM API / C++ Runtime                                  │
│       │                                                        │
│       ▼                                                        │
│  PyTorch-native Model Definition                               │
│       │                                                        │
│       ├──► torch.compile + TensorRT backend                    │
│       │       │                                                │
│       │       └──► NVIDIA custom kernels                       │
│       │           ├── FP8/FP4 GEMM (Blackwell native)          │
│       │           ├── FlashAttention (fused)                   │
│       │           ├── CUTLASS kernels                          │
│       │           └── Custom MoE kernels                       │
│       │                                                        │
│       ├──► Inflight Batching + Paged KV-Cache                  │
│       ├──► Multi-GPU (TP/PP/EP)                                │
│       ├──► Speculative Decoding (EAGLE/MTP/Draft)              │
│       └──► Quantization (FP8, FP4, INT4 AWQ, INT8 SQ)          │
│                                                                │
│  Integration: NVIDIA Triton Server, NVIDIA Dynamo              │
│  Hardware: NVIDIA GPUs ONLY (H100/A100/B200/GB200)             │
└──────────────────────────────────────────────────────────────────┘
```

**Why quantization researchers should care about TensorRT-LLM:**
- Best FP4/FP8 performance (native Blackwell support)
- Fully open-source since March 2025
- NVIDIA provides pre-quantized models on HuggingFace (nvidia/DeepSeek-R1-FP4)
- Tight integration with NVIDIA hardware features (tensor cores, NVLink)

### llama.cpp — The Universal Local Engine

```
┌────────────────────── llama.cpp Architecture ──────────────────┐
│                                                                │
│  CLI tools (llama-cli, llama-server)                           │
│       │                                                        │
│       ▼                                                        │
│  llama.h / llama.cpp (C/C++ core)                              │
│       │                                                        │
│       ▼                                                        │
│  ggml (tensor computation library)                             │
│       │                                                        │
│       ├──► CPU: AVX/AVX2/AVX512/AMX/NEON/SVE/RVV               │
│       ├──► NVIDIA GPU: CUDA (custom kernels)                   │
│       ├──► AMD GPU: HIP/ROCm                                   │
│       ├──► Apple Silicon: Metal                                │
│       ├──► Intel GPU: SYCL                                     │
│       ├──► Vulkan (any GPU)                                    │
│       ├──► CANN (Huawei Ascend)                                │
│       ├──► OpenCL (Qualcomm Adreno)                            │
│       └──► Many more backends                                  │
│                                                                │
│  Model Format: GGUF (with custom quantization types)           │
│  Quantization: Q2_K, Q3_K, Q4_0, Q4_K, Q5_K, Q6_K, Q8_0,       │
│                IQ1-IQ4 (importance-based), F16, BF16           │
└─────────────────────────────────────────────────────────────────┘
```

**Why quantization researchers should care about llama.cpp:**
- **GGUF quantization types are different from your research** — they use custom block quantization
- Largest hardware support of any engine (17+ backends)
- 96k GitHub stars — massive adoption for local/edge inference
- To deploy your AWQ model here, you'd need to convert to GGUF format
- The `convert_hf_to_gguf.py` script is the bridge from HuggingFace to llama.cpp

## 📈 Popularity & Adoption Trends (March 2026)

> **Note**: HuggingFace TGI (Text Generation Inference) is also widely used for quick deployment from HuggingFace Hub. It's covered in [09 — Serving & Middleware](./09_serving_and_middleware.md) as it sits between an inference engine and a serving solution.

```
GitHub Stars Over Time (approximate):

llama.cpp  ████████████████████████████████████████████████ 96k (platform king)
vLLM       ██████████████████████████                      50k (cloud serving standard)
SGLang     ████████████                                    24k (fastest growing)
MLC-LLM    ██████████                                      20k (cross-platform)
TRT-LLM    ██████                                          13k (NVIDIA ecosystem)
```

## 🔧 Which Engine for Your Quantized Model?

> **See also**: [06 — Quantization to Deployment Bridge](./06_quantization_to_deployment.md) for the full end-to-end pipeline from your research to production, and [12 — Practical Code Guide](./12_practical_code_guide.md) for working code examples.

```
                  Start Here
                      │
                      ▼
           ┌─────────────────────┐
           │ What hardware are   │
           │ you targeting?      │
           └──────────┬──────────┘
                      │
        ┌─────────────┼──────────────┐
        │             │              │
   NVIDIA only    Multi-vendor    Edge/Local
        │             │              │
        ▼             ▼              ▼
  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ Need max │  │ Need max │  │ CPU or   │
  │ perf?    │  │ hardware │  │ phone?   │
  └────┬─────┘  │ coverage?│  └────┬─────┘
       │        └────┬─────┘       │
   ┌───┴───┐   ┌────┴─────┐  ┌────┴─────┐
   │  Yes  │   │   Yes    │  │   Yes    │
   │       │   │          │  │          │
   ▼       ▼   ▼          ▼  ▼          │
TRT-LLM  vLLM vLLM    llama.cpp  llama.cpp
         or   or       or         or
         SGLang SGLang  MLC-LLM   ExecuTorch
```

## 🔗 How Quantization Formats Map to Engines

| Quantization Method | vLLM | SGLang | TRT-LLM | llama.cpp | ONNX RT |
|---------------------|------|--------|---------|-----------|---------|
| AWQ (W4A16) | ✅ native | ✅ native | ✅ native | ⚠️ convert to GGUF | ⚠️ limited |
| GPTQ (W4A16) | ✅ native | ✅ native | ✅ native | ❌ | ⚠️ limited |
| SmoothQuant (W8A8) | ✅ native | ✅ native | ✅ native | ✅ Q8_0 | ✅ |
| FP8 (W8A8) | ✅ native | ✅ native | ✅ native | ✅ | ✅ |
| FP4 (W4A4) | ✅ | ✅ | ✅ Blackwell | ✅ MXFP4 | ❌ |
| GGUF Quants | ✅ loads GGUF | ❌ | ❌ | ✅ native | ❌ |
| BitsAndBytes NF4 | ✅ | ✅ | ❌ | ❌ | ❌ |
| Custom research format | Need custom kernel | Need custom kernel | Need custom kernel | Need GGUF type | Need EP |

---

**Next**: [04 — Runtimes & Backends →](./04_runtimes_and_backends.md)
