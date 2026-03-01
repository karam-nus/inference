[← Back to Table of Contents](./README.md)

# 02 — Glossary & Concept Map

> Every term you encounter in inference/deployment, precisely defined and linked to related concepts.

*Last updated: March 2026*

## 🗺️ Master Concept Map

```
┌─────────────────────────────── THE DEPLOYMENT UNIVERSE ────────────────────────────────┐
│                                                                                         │
│  ┌─────────────────┐   exports to   ┌──────────────────┐   loaded by   ┌─────────────┐│
│  │ TRAINING         │──────────────►│ MODEL FORMAT      │────────────►│ INFERENCE    ││
│  │ FRAMEWORK        │               │                   │              │ ENGINE       ││
│  │                  │               │ • ONNX             │              │              ││
│  │ • PyTorch        │               │ • GGUF             │              │ • vLLM       ││
│  │ • JAX            │               │ • SafeTensors      │              │ • SGLang     ││
│  │ • TensorFlow     │               │ • TorchScript      │              │ • TRT-LLM   ││
│  │                  │               │ • TensorRT Engine  │              │ • llama.cpp  ││
│  └─────────────────┘               └──────────────────┘              └──────┬────────┘│
│                                                                             │          │
│                                                                     uses    │          │
│                                                                             ▼          │
│  ┌─────────────────┐   generates    ┌──────────────────┐  dispatches ┌─────────────┐  │
│  │ COMPILER         │──────────────►│ KERNEL            │◄───────────│ RUNTIME      │  │
│  │                  │               │                   │            │              │  │
│  │ • TVM            │               │ • cuBLAS           │            │ • ONNX RT    │  │
│  │ • XLA            │               │ • CUTLASS          │            │ • libtorch   │  │
│  │ • Triton (lang)  │               │ • FlashAttention   │            │ • ggml       │  │
│  │ • MLIR           │               │ • custom kernels   │            │ • ExecuTorch │  │
│  │ • torch.compile  │               │                   │            │              │  │
│  └─────────────────┘               └────────┬─────────┘            └──────┬────────┘  │
│                                              │                            │            │
│                                       runs on│                    talks to│            │
│                                              ▼                            ▼            │
│                                    ┌──────────────────┐    ┌──────────────────┐        │
│                                    │ HARDWARE BACKEND  │    │ SERVING LAYER    │        │
│                                    │                   │    │                  │        │
│                                    │ • CUDA / cuDNN    │    │ • Triton Server  │        │
│                                    │ • ROCm / HIP      │    │ • KServe         │        │
│                                    │ • Metal            │    │ • BentoML        │        │
│                                    │ • SYCL / oneAPI    │    │ • Ray Serve      │        │
│                                    │ • Vulkan           │    │                  │        │
│                                    └────────┬─────────┘    └────────┬─────────┘        │
│                                              │                      │                  │
│                                       runs on│              fronted by                 │
│                                              ▼                      ▼                  │
│                                    ┌──────────────────┐    ┌──────────────────┐        │
│                                    │ HARDWARE          │    │ MIDDLEWARE       │        │
│                                    │                   │    │                  │        │
│                                    │ • NVIDIA GPU      │    │ • Load Balancer  │        │
│                                    │ • AMD GPU/CPU     │    │ • API Gateway    │        │
│                                    │ • Intel CPU/GPU   │    │ • Auth / TLS     │        │
│                                    │ • Google TPU      │    │ • Rate Limiter   │        │
│                                    │ • Apple Silicon   │    │ • Monitoring     │        │
│                                    │ • Edge NPU/MCU   │    │                  │        │
│                                    └──────────────────┘    └──────────────────┘        │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📖 Glossary (Alphabetical)

### A

| Term | Definition | Related To | Analogy |
|------|-----------|------------|---------|
| **Accelerator** | Specialized hardware designed for specific computations (GPU, TPU, NPU) | Hardware | A sports car for matrix math |
| **AMX** | Advanced Matrix Extensions — Intel's CPU instruction set for matrix ops | Intel, CPU inference | Like tensor cores, but on CPU |
| **ASIC** | Application-Specific Integrated Circuit — custom chip for one task | NPU, TPU, Groq LPU | A factory built for one product |
| **AWQ** | Activation-aware Weight Quantization — your quantization algorithm | Quantization, INT4 | — |
| **AITER** | AMD Inference Tiled Engine Routines — AMD's optimized kernels for LLM inference on ROCm | AMD, ROCm, Kernels | AMD's answer to NVIDIA's custom CUDA kernels |

### B

| Term | Definition | Related To | Analogy |
|------|-----------|------------|---------|
| **Backend** | The hardware-specific implementation that executes computations | Runtime, Hardware | The engine inside a car |
| **Batching** | Processing multiple requests together for efficiency | Inference Engine | Washing multiple dishes at once |
| **BentoML** | Open-source serving framework for ML models | Serving | — |

### C

| Term | Definition | Related To | Analogy |
|------|-----------|------------|---------|
| **Compiler (ML)** | Transforms model graphs into optimized machine code for specific hardware | TVM, XLA, MLIR | A translator from Python to assembly |
| **Continuous Batching** | Dynamically adding/removing requests from a batch as they complete | Inference Engine | A bus that picks up/drops off passengers at every stop |
| **CUDA** | NVIDIA's parallel computing platform and programming model | NVIDIA GPU, cuBLAS | NVIDIA's "operating system" for GPUs |
| **cuBLAS** | NVIDIA's GPU-accelerated linear algebra library | CUDA, Kernels | Pre-optimized matrix math functions |
| **CUTLASS** | NVIDIA's open-source CUDA C++ template library for matrix operations | CUDA, Kernels | Building blocks for custom GPU matrix operations |
| **cuDNN** | NVIDIA's deep learning primitives library | CUDA, Kernels | Pre-optimized deep learning ops |

### D

| Term | Definition | Related To | Analogy |
|------|-----------|------------|---------|
| **Disaggregated Serving** | Separating prefill (prompt processing) and decode (token generation) on different hardware | Serving, vLLM, TRT-LLM | Different factory lines for assembly vs. packaging |
| **Driver** | Software that enables OS-to-hardware communication | Hardware Backend | The steering wheel connecting you to the engine |
| **Decode Phase** | Autoregressive token generation — one token at a time, memory-bandwidth-bound | Prefill, Inference Engine | Writing one word at a time in an answer |

### E

| Term | Definition | Related To | Analogy |
|------|-----------|------------|---------|
| **Execution Provider (EP)** | ONNX Runtime's abstraction for different hardware backends | ONNX Runtime | A plugin system for hardware |
| **ExecuTorch** | Meta's on-device/edge inference runtime for PyTorch models | Edge, Runtime | PyTorch for your phone |
| **Expert Parallelism (EP)** | Distributing MoE experts across GPUs | MoE, Parallelism | Each GPU handles different specialists |

### F

| Term | Definition | Related To | Analogy |
|------|-----------|------------|---------|
| **FlashAttention** | Memory-efficient attention algorithm that's IO-aware | Kernel, Attention | A librarian who reads books without copying them first |
| **FP4 / FP8** | 4-bit / 8-bit floating point formats | Quantization, Blackwell | Smaller numbers, less memory, faster math |
| **FlashInfer** | Library of GPU kernels for LLM serving, especially PagedAttention variants | Kernel, SGLang, vLLM | Specialized attention kernels for serving |

### G

| Term | Definition | Related To | Analogy |
|------|-----------|------------|---------|
| **GGUF** | GPT-Generated Unified Format — llama.cpp's model format | llama.cpp, Model Format | A self-contained model file with metadata |
| **ggml** | Tensor library underlying llama.cpp | Runtime, llama.cpp | NumPy for C, designed for inference |
| **Graph Optimization** | Transforming a computation graph (fusing ops, eliminating redundancy) | Compiler, Runtime | Simplifying a recipe by combining steps |
| **GQA** | Grouped-Query Attention — shares KV heads across query heads (used in Llama 3, Mistral). Reduces KV-cache size vs MHA | Attention, KV-Cache | Multiple readers sharing fewer notebooks |
| **MHA** | Multi-Head Attention — each query head has its own KV head (original transformer). Largest KV-cache | Attention, KV-Cache | Every reader has their own notebook |
| **MQA** | Multi-Query Attention — all query heads share a single KV head. Smallest KV-cache, may reduce quality | Attention, KV-Cache | All readers share one notebook |

### H

| Term | Definition | Related To | Analogy |
|------|-----------|------------|---------|
| **HIP** | Heterogeneous-Compute Interface for Portability — AMD's CUDA-like API | ROCm, AMD | AMD's answer to CUDA |
| **Hardware Backend** | The software layer (driver + libraries) that interfaces with hardware | CUDA, ROCm, Metal | The translator between software and silicon |

### I

| Term | Definition | Related To | Analogy |
|------|-----------|------------|---------|
| **Inference** | Running a trained model to produce predictions/outputs | Forward pass | Using a trained chef to cook meals |
| **Inference Engine** | Software that optimizes and manages LLM inference (batching, KV-cache, scheduling) | vLLM, SGLang, TRT-LLM | The factory manager optimizing production |
| **Importance Matrix (imatrix)** | Per-weight importance scores derived from calibration data, used by llama.cpp to allocate more bits to important weights during quantization | GGUF, Calibration | Giving better tools to your best workers |
| **ITL** | Inter-Token Latency — time between consecutive output tokens during generation | Serving, TPOT | How fast words appear one after another |

### K

| Term | Definition | Related To | Analogy |
|------|-----------|------------|---------|
| **Kernel** | A function compiled to run on a specific hardware (GPU kernel, CPU kernel) | CUDA, Compiler | A single specialized machine on the factory floor |
| **KServe** | Kubernetes-native model serving framework | Serving, k8s | Kubernetes for ML models |
| **KV-Cache** | Cached key/value tensors from previous tokens to avoid recomputation | Attention, Memory | Remembering previous conversation so you don't repeat yourself |

### L

| Term | Definition | Related To | Analogy |
|------|-----------|------------|---------|
| **llama.cpp** | LLM inference in C/C++ — CPU-first, multi-backend | Inference Engine + Runtime | The Swiss Army knife of local LLM inference |
| **LoRA** | Low-Rank Adaptation — efficient fine-tuning method | Serving (multi-LoRA) | Changing the car's paint without rebuilding the engine |

### M

| Term | Definition | Related To | Analogy |
|------|-----------|------------|---------|
| **Metal** | Apple's GPU programming framework | Apple Silicon | CUDA for Apple |
| **Middleware** | Software between the application and the serving infrastructure | API Gateway, Load Balancer | The receptionist between clients and the chef |
| **MLIR** | Multi-Level Intermediate Representation — compiler infrastructure | Compiler, TVM | A universal language for compiler passes |
| **MoE** | Mixture of Experts — model architecture where only some parameters are active | Expert Parallelism | A hospital with specialist doctors, not all consulted per patient |
| **Marlin** | High-performance INT4×FP16 GEMM kernel for NVIDIA GPUs, used in vLLM/SGLang for AWQ/GPTQ models | Kernel, vLLM, W4A16 | A speed-optimized assembly line for quantized math |
| **ModelOpt** | NVIDIA's quantization and optimization toolkit (supports FP8, FP4, INT8, INT4, sparsity) | NVIDIA, Quantization | NVIDIA's official quantization toolbox |

### N

| Term | Definition | Related To | Analogy |
|------|-----------|------------|---------|
| **NPU** | Neural Processing Unit — dedicated AI accelerator | Edge, ASIC | A GPU but only for neural networks |
| **NCCL** | NVIDIA Collective Communications Library — GPU-to-GPU communication | Multi-GPU, Parallelism | The postal service between GPUs |
| **NNCF** | Neural Network Compression Framework — Intel's toolkit for INT8/INT4 quantization and pruning | Intel, OpenVINO, PTQ | Intel's compression toolbox |

### O

| Term | Definition | Related To | Analogy |
|------|-----------|------------|---------|
| **ONNX** | Open Neural Network Exchange — standard model format | Model Format, Interop | PDF for ML models |
| **ONNX Runtime** | Cross-platform inference runtime with pluggable execution providers | Runtime | A universal model player |
| **OpenVINO** | Intel's inference optimization toolkit | Intel, Compiler+Runtime | Intel's TensorRT equivalent |

### P

| Term | Definition | Related To | Analogy |
|------|-----------|------------|---------|
| **PagedAttention** | vLLM's memory management for KV-cache, inspired by OS virtual memory | KV-Cache, vLLM | Virtual memory but for attention |
| **Pipeline Parallelism** | Splitting model layers across GPUs | Multi-GPU, Parallelism | An assembly line across multiple stations |
| **Prefill** | Processing the entire prompt at once (compute-bound) | Inference, Disaggregated | Reading a whole question before answering |
| **PTQ** | Post-Training Quantization — quantize a model *after* training is complete. Cheaper than QAT but may lose quality at low bit-widths. Examples: AWQ, GPTQ, SmoothQuant | Quantization, Calibration | Shrinking clothes after they're already made |

### Q

| Term | Definition | Related To | Analogy |
|------|-----------|------------|---------|
| **QAT** | Quantization-Aware Training — simulate quantization *during* training so model learns to be robust to low precision. Better quality than PTQ but much more expensive (needs full training run) | Quantization, PTQ | Designing clothes to be small from the start |
| **Quark** | AMD's quantization toolkit for ROCm/MI-series GPUs (supports FP8, INT8, INT4) | AMD, ROCm, Quantization | AMD's official quantization toolbox |

### R

| Term | Definition | Related To | Analogy |
|------|-----------|------------|---------|
| **ROCm** | Radeon Open Compute — AMD's GPU compute platform | AMD, HIP | AMD's CUDA |
| **Runtime** | Software that loads a model and executes its operations | ONNX RT, ggml, libtorch | The engine that runs the model |

### S

| Term | Definition | Related To | Analogy |
|------|-----------|------------|---------|
| **SafeTensors** | A safe, fast model serialization format by Hugging Face | Model Format, HF | ZIP file for model weights, with safety |
| **Serving** | Making a model available as a service (API endpoints, scaling, routing) | Triton, KServe | Running a restaurant, not just cooking |
| **Speculative Decoding** | Using a small draft model to predict multiple tokens, verified by the main model | Inference Engine | A junior writer drafts, senior editor approves |
| **SSE** | Server-Sent Events — HTTP-based protocol for streaming tokens from server to client (used by most LLM APIs) | Serving, API | A live news ticker that pushes updates |
| **SYCL** | Open standard for heterogeneous computing (used by Intel) | Intel, oneAPI | A cross-platform CUDA |

### T

| Term | Definition | Related To | Analogy |
|------|-----------|------------|---------|
| **Tensor Core** | NVIDIA's specialized matrix multiply units on GPUs | NVIDIA GPU, Matmul | Specialized math circuits |
| **TensorRT** | NVIDIA's high-performance inference optimizer and runtime | NVIDIA, Compiler+Runtime | NVIDIA's optimizing compiler for neural nets |
| **TensorRT-LLM** | NVIDIA's LLM-specific inference library built on TensorRT + PyTorch | Inference Engine | TensorRT but specifically for LLMs |
| **Triton (compiler)** | OpenAI's language for writing GPU kernels in Python | Compiler, Kernels | Writing GPU code in Python instead of CUDA C++ |
| **Triton Inference Server** | NVIDIA's model serving platform (NOT the same as Triton compiler!) | Serving | ⚠️ Easily confused with Triton compiler |
| **torch.compile** | PyTorch 2.0's JIT compiler, uses TorchInductor as backend | Compiler, PyTorch | Making PyTorch faster without changing code |
| **TVM** | Apache TVM — open-source ML compiler framework | Compiler | A universal compiler for any hardware |
| **TGI** | Text Generation Inference — HuggingFace's production LLM serving solution (Rust + Python) | Serving, HuggingFace | Quick-deploy serving from HuggingFace Hub |
| **TTFT** | Time to First Token — latency from request submission to first generated token. Dominated by prefill time | Serving, Latency | How long before the waiter starts bringing food |
| **TPOT** | Time Per Output Token — average time to generate each subsequent token after the first. Dominated by decode time | Serving, Latency | How fast subsequent plates arrive |
| **TPU** | Tensor Processing Unit — Google's custom AI accelerator | Google, Hardware | Google's proprietary GPU alternative |

### V

| Term | Definition | Related To | Analogy |
|------|-----------|------------|---------|
| **vLLM** | High-throughput LLM serving engine with PagedAttention | Inference Engine | The Toyota of LLM serving — reliable and popular |
| **Vulkan** | Cross-platform graphics/compute API | GPU, Backend | A universal GPU language |

### W

| Term | Definition | Related To | Analogy |
|------|-----------|------------|---------|
| **WxAy notation** | Shorthand for weight and activation precision. W4A16 = 4-bit weights, 16-bit activations. W8A8 = 8-bit both. W4A4 = 4-bit both. The "W" part is usually easier to quantize than the "A" part | Quantization | Shirt size (W) × shoe size (A) — independent choices |

### X

| Term | Definition | Related To | Analogy |
|------|-----------|------------|---------|
| **XLA** | Accelerated Linear Algebra — Google's ML compiler | TPU, Compiler | Google's compiler for TPUs (and GPUs) |

---

## 🔑 Key Concepts for Quantization Researchers

### Prefill vs Decode: Why It Matters for Quantization

```
┌──────────── THE TWO PHASES OF LLM INFERENCE ─────────────┐
│                                                            │
│  PREFILL (Prompt Processing)                              │
│  ├── Processes ALL input tokens in parallel               │
│  ├── COMPUTE-BOUND (large matrix multiplications)         │
│  ├── Bottleneck: Tensor Core throughput                   │
│  ├── Quantization impact: More ops/sec with lower prec   │
│  │   → FP8/INT8 compute 2× faster than FP16 on tensor   │
│  │     cores, but this only helps if compute is the       │
│  │     bottleneck (it is during prefill)                  │
│  └── Metric: Time to First Token (TTFT)                  │
│                                                            │
│  DECODE (Token Generation)                                │
│  ├── Generates ONE token at a time (autoregressive)       │
│  ├── MEMORY-BANDWIDTH-BOUND (read all weights per token) │
│  ├── Bottleneck: HBM → SM data transfer speed            │
│  ├── Quantization impact: Read fewer bytes per token      │
│  │   → INT4 reads 4× less data than FP16 per token       │
│  │   → This is the PRIMARY reason W4A16 speeds up LLMs!  │
│  └── Metric: Time Per Output Token (TPOT)                │
│                                                            │
│  KEY INSIGHT:                                             │
│  Weight quantization (W4/W8) helps DECODE most            │
│  (fewer bytes to read per token).                         │
│  Activation quantization (A8/A4) helps PREFILL most       │
│  (more compute ops per second).                           │
│  This is why W4A16 is so popular — it targets the         │
│  decode bottleneck without the difficulty of quantizing   │
│  activations.                                             │
└────────────────────────────────────────────────────────────┘
```

### Attention Variants: MHA vs GQA vs MQA

```
These determine KV-cache size, which affects memory and quantization decisions.

┌─── Multi-Head Attention (MHA) ───┐  Original transformer (GPT-2, etc.)
│                                    │
│  Q heads: H     K heads: H        │  KV-cache per token:
│  ┌─┐┌─┐┌─┐┌─┐  ┌─┐┌─┐┌─┐┌─┐    │  2 × H × d × num_layers × precision
│  │0││1││2││3│  │0││1││2││3│    │  (largest KV-cache)
│  └─┘└─┘└─┘└─┘  └─┘└─┘└─┘└─┘    │
│   ↓  ↓  ↓  ↓    ↓  ↓  ↓  ↓      │
│   1:1 mapping (each Q has own K)  │
└────────────────────────────────────┘

┌─── Grouped-Query Attention (GQA) ─┐  Llama 3, Mistral, Gemma
│                                     │
│  Q heads: 8     K heads: 2         │  KV-cache per token:
│  ┌─┐┌─┐┌─┐┌─┐  ┌───┐              │  2 × G × d × num_layers × precision
│  │0││1││2││3│  │ 0 │              │  (G groups, much smaller)
│  └─┘└─┘└─┘└─┘  └───┘              │
│  ┌─┐┌─┐┌─┐┌─┐  ┌───┐              │
│  │4││5││6││7│  │ 1 │              │
│  └─┘└─┘└─┘└─┘  └───┘              │
│   4 Q heads share 1 K head         │
└─────────────────────────────────────┘

┌─── Multi-Query Attention (MQA) ───┐  PaLM, Falcon, StarCoder
│                                     │
│  Q heads: 8     K head: 1          │  KV-cache per token:
│  ┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐ ┌──┐  │  2 × 1 × d × num_layers × precision
│  │0││1││2││3││4││5││6││7│ │0 │  │  (smallest KV-cache)
│  └─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘ └──┘  │
│   ALL Q heads share 1 K head       │
└─────────────────────────────────────┘

WHY THIS MATTERS FOR QUANTIZATION:
• GQA/MQA already reduce KV-cache → less pressure for KV-cache quantization
• KV-cache quantization (FP8, INT8) further reduces memory
• Smaller KV-cache = more batch capacity = higher throughput
• Most modern LLMs use GQA (Llama 3: 8 KV heads for 32 Q heads)
```

> **See also**: [09 — Serving & Middleware](./09_serving_and_middleware.md) for TTFT/TPOT metrics in production.

---

## ⚠️ Common Confusions

| Confusion | Clarification |
|-----------|--------------|
| **Triton (compiler) vs. Triton Inference Server** | Completely different projects! Triton compiler = OpenAI's GPU programming language. Triton Server = NVIDIA's model serving platform. |
| **Runtime vs. Inference Engine** | Runtime = executes individual ops. Engine = manages the full LLM lifecycle (batching, KV-cache, scheduling). Engines *use* runtimes. |
| **Backend vs. Hardware** | Backend = software that talks to hardware (CUDA, ROCm). Hardware = the physical chip (H100, MI300X). |
| **Serving vs. Inference** | Inference = running the model once. Serving = handling thousands of concurrent requests with load balancing, scaling, etc. |
| **Compiler vs. Runtime** | Compiler = transforms/optimizes the model graph *before* execution. Runtime = executes the model. Some tools (TVM, TRT) do both. |
| **ONNX (format) vs. ONNX Runtime** | ONNX = a file format for storing models. ONNX Runtime = software that *runs* ONNX models. |
| **Quantization (algorithm) vs. Quantized Execution** | You develop the algorithm. The inference engine needs *kernels* that support your specific format to actually run it fast. |
| **PTQ vs. QAT** | PTQ (Post-Training Quantization) quantizes *after* training — cheaper, used by AWQ/GPTQ/SmoothQuant. QAT (Quantization-Aware Training) simulates quantization *during* training — better quality but requires a full training run. Most deployment-focused research is PTQ. |
| **W4A16 vs. W4A4** | W4A16 = 4-bit weights, 16-bit activations (common, easy). W4A4 = 4-bit both (rare, hard — activations have outliers that are difficult to quantize). See [06 — Quantization to Deployment Bridge](./06_quantization_to_deployment.md). |

---

## 🔤 Acronym Quick Reference

| Acronym | Full Name |
|---------|-----------|
| AMX | Advanced Matrix Extensions |
| ASIC | Application-Specific Integrated Circuit |
| AWQ | Activation-aware Weight Quantization |
| CUDA | Compute Unified Device Architecture |
| EP | Execution Provider (ONNX Runtime context) / Expert Parallelism (MoE context) ⚠️ *context-dependent* |
| FP4/FP8/FP16 | 4/8/16-bit Floating Point |
| GQA | Grouped-Query Attention |
| GGUF | GPT-Generated Unified Format |
| GPTQ | GPT-Quantization |
| HIP | Heterogeneous-Compute Interface for Portability |
| ISA | Instruction Set Architecture |
| ITL | Inter-Token Latency |
| KV | Key-Value (as in KV-cache) |
| LPU | Language Processing Unit (Groq) |
| MHA | Multi-Head Attention |
| MLC | Machine Learning Compilation |
| MLIR | Multi-Level Intermediate Representation |
| MoE | Mixture of Experts |
| MQA | Multi-Query Attention |
| NCCL | NVIDIA Collective Communications Library |
| NNCF | Neural Network Compression Framework (Intel) |
| NPU | Neural Processing Unit |
| ONNX | Open Neural Network Exchange |
| PTQ | Post-Training Quantization |
| QAT | Quantization-Aware Training |
| ROCm | Radeon Open Compute |
| SYCL | (originally) System-wide Compute Language |
| SSE | Server-Sent Events |
| TFLite | TensorFlow Lite |
| TGI | Text Generation Inference (HuggingFace) |
| TPOT | Time Per Output Token |
| TPP | Tensor Processing Primitives |
| TPU | Tensor Processing Unit |
| TRT | TensorRT |
| TTFT | Time to First Token |
| TVM | Tensor Virtual Machine |
| WSE | Wafer-Scale Engine (Cerebras) |
| XLA | Accelerated Linear Algebra |

---

**Next**: [03 — Inference Engines Deep Dive →](./03_inference_engines.md)
