[← Back to Table of Contents](./README.md)

# 04 — Runtimes & Backends

> The layer that actually executes your model's operations on specific hardware.

*Last updated: March 2026*

## 🎯 What Is a Runtime vs. a Backend?

```
┌─────────────────── Terminology Clarification ──────────────────┐
│                                                                 │
│  RUNTIME = Software that loads a model graph and executes it    │
│            Examples: ONNX Runtime, libtorch, ggml, ExecuTorch   │
│                                                                 │
│  BACKEND = Hardware-specific code the runtime dispatches to     │
│            Examples: CUDA, ROCm/HIP, Metal, Vulkan, SYCL       │
│                                                                 │
│  EXECUTION PROVIDER (EP) = ONNX Runtime's term for "backend"   │
│            Examples: CUDAExecutionProvider, CPUExecutionProvider │
│                                                                 │
│  DRIVER = Low-level OS-to-hardware interface                    │
│            Examples: nvidia-driver, amdgpu, intel-gpu-tools     │
│                                                                 │
│  KERNEL LIBRARY = Pre-optimized math functions for hardware     │
│            Examples: cuBLAS, CUTLASS, MIOpen, oneDNN            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

   Your model → Runtime → Backend → Driver → Hardware
```

## 📊 Runtime Comparison Table

| Runtime | Creator | Primary Use | Languages | Hardware Support | Model Format |
|---------|---------|------------|-----------|-----------------|-------------|
| **ONNX Runtime** | Microsoft | Universal inference | C++, Python, C#, Java, JS | CUDA, ROCm, OpenVINO, TRT, DirectML, CoreML, QNN | ONNX |
| **libtorch** | Meta/PyTorch | PyTorch model execution | C++, Python | CUDA, ROCm, CPU, XPU, MPS | TorchScript, torch.export |
| **ggml** | ggerganov | Lightweight tensor ops | C | 17+ backends (see llama.cpp) | GGUF |
| **ExecuTorch** | Meta | On-device/edge inference | C++, Python | CPU, XNNPACK, CoreML, QNN, Vulkan, MPS, HTP | torch.export (.pte) |
| **TFLite Runtime** | Google | Mobile/edge inference | C++, Python, Java, Swift | CPU, GPU (OpenGL/CL), NNAPI, Edge TPU, Hexagon | FlatBuffer (.tflite) |
| **OpenVINO Runtime** | Intel | Intel hardware optimization | C++, Python | Intel CPU, Intel GPU, VPU, GNA | OpenVINO IR, ONNX |
| **TensorRT Runtime** | NVIDIA | NVIDIA GPU inference | C++, Python | CUDA only | TensorRT Engine (.plan) |

## 🔍 Deep Dive: Key Runtimes

### ONNX Runtime — The Universal Player

```
┌──────────────────── ONNX Runtime Architecture ─────────────────┐
│                                                                  │
│  ONNX Model (.onnx)                                             │
│       │                                                          │
│       ▼                                                          │
│  Graph Optimization Passes                                       │
│  ├── Constant folding                                            │
│  ├── Operator fusion (Conv+BN, MatMul+Add, etc.)                │
│  ├── Shape inference                                             │
│  └── Quantization-aware optimizations                            │
│       │                                                          │
│       ▼                                                          │
│  Graph Partitioning                                              │
│  "Which EP handles which subgraph?"                              │
│       │                                                          │
│       ├──►┌──────────────────────┐                              │
│       │   │ CUDAExecutionProvider│ → NVIDIA GPU subgraphs       │
│       │   └──────────────────────┘                              │
│       │                                                          │
│       ├──►┌──────────────────────┐                              │
│       │   │ TensorrtEP           │ → Optimized TensorRT engine  │
│       │   └──────────────────────┘                              │
│       │                                                          │
│       ├──►┌──────────────────────┐                              │
│       │   │ OpenVINOEP           │ → Intel optimized            │
│       │   └──────────────────────┘                              │
│       │                                                          │
│       └──►┌──────────────────────┐                              │
│           │ CPUExecutionProvider │ → Fallback for remaining ops │
│           └──────────────────────┘                              │
│                                                                  │
│  Key Insight: Different parts of your model can run on          │
│  different hardware simultaneously!                              │
└──────────────────────────────────────────────────────────────────┘
```

**Full list of ONNX Runtime Execution Providers:**

| EP | Hardware | Quantization Support |
|----|----------|---------------------|
| CUDA EP | NVIDIA GPU | FP16, INT8, FP8 |
| TensorRT EP | NVIDIA GPU (via TensorRT) | INT8, FP8, INT4 (limited) |
| ROCm EP | AMD GPU | FP16, INT8 |
| DirectML EP | Windows GPU (any) | FP16, INT8 |
| OpenVINO EP | Intel CPU/GPU/VPU | INT8, INT4 (NNCF) |
| CoreML EP | Apple (macOS/iOS) | FP16, INT8 |
| QNN EP | Qualcomm NPU/DSP | INT8, INT4 |
| NNAPI EP | Android NPU/DSP | INT8 |
| CPU EP | Any CPU | INT8, dynamic quantization |
| XNNPACK EP | Mobile CPU (ARM/x86) | FP16, INT8 |
| Vitis AI EP | Xilinx/AMD FPGA | INT8 |
| ACL EP | ARM CPU | FP16, INT8 |
| Azure EP | Azure cloud AI | Various |
| SNPE EP | Qualcomm (legacy) | INT8 |
| MIGraphX EP | AMD GPU | FP16, INT8 |
| CANN EP | Huawei Ascend NPU | FP16, INT8 |
| WebGPU EP | Browser GPU | FP16 |
| WebNN EP | Browser NN accelerator | Varies |

### ggml — The Lightweight Champion

```
┌────────────────────────── ggml Architecture ───────────────────┐
│                                                                 │
│  Design Philosophy: "Pure C, no dependencies, runs everywhere"  │
│                                                                 │
│  Core Abstractions:                                             │
│  ├── ggml_tensor: N-dimensional array with type info            │
│  ├── ggml_context: Memory pool for tensors                      │
│  ├── ggml_cgraph: Computation graph                             │
│  └── ggml_backend: Hardware dispatch                            │
│                                                                 │
│  Quantization Types (unique to ggml):                           │
│  ├── Q4_0: 4-bit uniform quantization (32 weights per block)   │
│  ├── Q4_1: 4-bit with min value offset                         │
│  ├── Q4_K_M: 4-bit with K-quant super-blocks (recommended)    │
│  ├── Q5_K_M: 5-bit K-quant (quality-optimized)                │
│  ├── Q6_K: 6-bit K-quant                                       │
│  ├── Q8_0: 8-bit symmetric                                     │
│  ├── IQ1_S: 1.5-bit importance quantization                   │
│  ├── IQ2_XS: 2.3-bit importance quantization                  │
│  ├── IQ3_S: 3.4-bit importance quantization                   │
│  └── IQ4_XS: 4.25-bit importance quantization                 │
│                                                                 │
│  Backend Dispatch (17+ backends):                               │
│  ├── CPU: x86 (AVX/AVX2/AVX512/AMX), ARM (NEON/SVE), RISC-V  │
│  ├── CUDA: custom kernels for NVIDIA GPUs                       │
│  ├── Metal: Apple GPU compute shaders                           │
│  ├── Vulkan: cross-platform GPU compute                         │
│  ├── SYCL: Intel GPUs                                           │
│  ├── HIP: AMD GPUs                                              │
│  ├── CANN: Huawei Ascend NPUs                                  │
│  ├── OpenCL: Qualcomm Adreno GPUs                              │
│  └── ZenDNN: AMD CPU optimization                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key for quantization researchers**: ggml's quantization types are a *separate research line* from AWQ/GPTQ/SmoothQuant. They use block-based quantization with different block sizes and encoding schemes.

### Hardware Backend Software Stack

```
┌────────────────────── NVIDIA CUDA Stack ───────────────────────┐
│                                                                 │
│  Application (vLLM, TRT-LLM, etc.)                             │
│       │                                                         │
│       ▼                                                         │
│  CUDA Runtime API (cudart)                                      │
│       │                                                         │
│       ├──► cuBLAS (dense linear algebra: GEMM, etc.)           │
│       ├──► CUTLASS (template library for custom GEMMs)          │
│       ├──► cuDNN (convolutions, normalization, activation)      │
│       ├──► NCCL (multi-GPU communication)                       │
│       ├──► FlashAttention (memory-efficient attention)          │
│       ├──► FlashInfer (PagedAttention kernels)                 │
│       └──► Custom Triton kernels (written in Triton language)  │
│       │                                                         │
│       ▼                                                         │
│  CUDA Driver API                                                │
│       │                                                         │
│       ▼                                                         │
│  NVIDIA GPU Driver                                              │
│       │                                                         │
│       ▼                                                         │
│  NVIDIA GPU Hardware (Tensor Cores, CUDA Cores, HBM)           │
└─────────────────────────────────────────────────────────────────┘

┌────────────────────── AMD ROCm Stack ──────────────────────────┐
│                                                                 │
│  Application (vLLM, SGLang, llama.cpp)                          │
│       │                                                         │
│       ▼                                                         │
│  HIP Runtime (source-compatible with CUDA)                      │
│       │                                                         │
│       ├──► hipBLAS / rocBLAS (linear algebra)                  │
│       ├──► MIOpen (deep learning primitives)                    │
│       ├──► RCCL (multi-GPU communication)                       │
│       ├──► Composable Kernel (AMD's CUTLASS equivalent)        │
│       ├──► AITER (AMD Inference Tiled Engine Routines)         │
│       └──► FlashAttention (ROCm port)                          │
│       │                                                         │
│       ▼                                                         │
│  ROCm Runtime + AMD GPU Driver                                  │
│       │                                                         │
│       ▼                                                         │
│  AMD GPU Hardware (Matrix Cores, Compute Units, HBM)            │
└─────────────────────────────────────────────────────────────────┘

┌────────────────────── Intel oneAPI Stack ──────────────────────┐
│                                                                 │
│  Application                                                    │
│       │                                                         │
│       ▼                                                         │
│  SYCL Runtime (or OpenVINO Runtime)                             │
│       │                                                         │
│       ├──► oneDNN / oneMKL (math libraries)                    │
│       ├──► AMX instructions (CPU matrix acceleration)          │
│       ├──► VNNI instructions (vector neural network)           │
│       └──► XMX (Xe Matrix Extensions, for Intel GPUs)          │
│       │                                                         │
│       ▼                                                         │
│  Intel GPU Driver / CPU                                         │
│       │                                                         │
│       ▼                                                         │
│  Intel Xeon CPU (AMX, AVX-512) or Intel Arc/Gaudi GPU          │
└─────────────────────────────────────────────────────────────────┘
```

## 🔗 How Your Quantized Matmul Actually Executes

When your AWQ INT4 model does a forward pass, here's what happens at the kernel level:

```
Python: output = quantized_linear(input)
    │
    ▼ (PyTorch dispatch)
C++ : at::matmul(input_fp16, dequant(weight_int4))
    │
    ▼ (vLLM custom op dispatch)
Custom kernel selection based on:
    ├── weight dtype (INT4)
    ├── input dtype (FP16)
    ├── group size (128)
    ├── GPU architecture (sm_90 for H100)
    └── matrix dimensions (M, N, K)
    │
    ▼
Selected kernel: Marlin INT4xFP16 GEMM
    │
    ▼ (CUDA launch)
GPU executes:
    1. Load INT4 weights from HBM (compressed, 2 weights per byte)
    2. Load FP16 scales from HBM
    3. Dequantize INT4 → FP16 on-the-fly in shared memory
    4. Compute FP16 GEMM using Tensor Cores
    5. Write FP16 output to HBM
```

## 📊 Kernel Libraries for Quantized Inference

> **Deep dive**: For detailed coverage of CUTLASS, Composable Kernel, and how compilers generate these kernels, see [11 — Compiler Stack](./11_compiler_stack.md).

| Kernel Library | Owner | Hardware | Quantization Support | Used By |
|---------------|-------|----------|---------------------|---------|
| **Marlin** | IST Austria + community | NVIDIA GPU | W4A16 (GPTQ/AWQ), W8A16 | vLLM, SGLang |
| **Machete** | NVIDIA | NVIDIA GPU | W4A16, W8A8, W4A8 | vLLM |
| **CUTLASS** | NVIDIA | NVIDIA GPU | INT8, FP8, FP4, mixed-precision | TRT-LLM, vLLM |
| **FlashAttention** | Tri Dao | NVIDIA GPU, AMD GPU | FP16, BF16, FP8 | vLLM, SGLang, TRT-LLM |
| **FlashInfer** | FlashInfer team | NVIDIA GPU | FP16, BF16, FP8, INT4 | SGLang, vLLM |
| **oneDNN** | Intel | Intel CPU/GPU | INT8, BF16, FP16 | OpenVINO, ONNX RT |
| **MIOpen** | AMD | AMD GPU | FP16, BF16, INT8 | ROCm stack |
| **XNNPACK** | Google | Mobile CPU | INT8, FP16 | TFLite, ExecuTorch |
| **Composable Kernel** | AMD | AMD GPU | Mixed-precision, INT8, FP8 | vLLM (ROCm), SGLang |
| **AITER** | AMD | AMD GPU | FP8, INT8, mixed | SGLang |

---

**Next**: [05 — Hardware Landscape →](./05_hardware_landscape.md)
