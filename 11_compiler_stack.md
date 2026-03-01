[← Back to Table of Contents](./README.md)

*Last updated: March 2026*

# 11 — Compiler Stack

> The hidden layer between your model and hardware — and why it matters for quantization.

## 🤔 Why Does a Compiler Exist for ML?

```
The Problem:
  You write:  y = torch.nn.Linear(4096, 4096)(x)   # High-level Python
  GPU sees:   SASS instructions, register allocs,    # Low-level machine code
              shared memory loads, warp scheduling

Who translates? THE COMPILER STACK.

┌───────── WITHOUT ML COMPILERS ─────────────────────┐
│                                                      │
│  PyTorch → Hand-written CUDA kernels for every op   │
│                                                      │
│  Problems:                                           │
│  • 100s of ops × 10s of hardware targets = 1000s    │
│    of hand-written kernels                           │
│  • No cross-op optimization (fusion)                 │
│  • Every new op needs new kernel for every HW       │
│  • Quantized ops multiply the problem               │
└──────────────────────────────────────────────────────┘

┌───────── WITH ML COMPILERS ────────────────────────┐
│                                                      │
│  PyTorch → IR (graph) → Compiler → Optimized code  │
│                                                      │
│  Benefits:                                           │
│  • Automatic op fusion                               │
│  • Hardware-specific code generation                 │
│  • New hardware = new backend (not rewrite all ops)  │
│  • Quantized ops can be lowered automatically        │
└──────────────────────────────────────────────────────┘
```

## 📊 The Compiler Landscape

```
┌──────────── ML COMPILER STACK (Layers) ──────────────────────┐
│                                                                │
│  HIGH-LEVEL FRAMEWORKS                                        │
│  ┌────────────────────────────────────────────────────────┐   │
│  │ PyTorch, JAX, TensorFlow, ONNX                         │   │
│  └──────────────────────────┬─────────────────────────────┘   │
│                              │                                 │
│  GRAPH CAPTURE / TRACING     │                                 │
│  ┌──────────────────────────┴──────────────────────────┐     │
│  │ torch.compile (Dynamo), tf.function, jax.jit         │     │
│  │ torch.export, fx.symbolic_trace                      │     │
│  └──────────────────────────┬──────────────────────────┘     │
│                              │                                 │
│  HIGH-LEVEL IR               │                                 │
│  ┌──────────────────────────┴──────────────────────────┐     │
│  │ torch.fx Graph, HLO (JAX/XLA), TF Graph, ONNX Graph│     │
│  └──────────────────────────┬──────────────────────────┘     │
│                              │                                 │
│  GRAPH OPTIMIZATIONS         │                                 │
│  ┌──────────────────────────┴──────────────────────────┐     │
│  │ Operator fusion, constant folding, dead code elim,   │     │
│  │ layout optimization, common subexpression elimination│     │
│  └──────────────────────────┬──────────────────────────┘     │
│                              │                                 │
│  MID-LEVEL IR                │                                 │
│  ┌──────────────────────────┴──────────────────────────┐     │
│  │ Triton IR, TVM TIR, Linalg (MLIR), StableHLO       │     │
│  └──────────────────────────┬──────────────────────────┘     │
│                              │                                 │
│  LOW-LEVEL IR / CODE GEN     │                                 │
│  ┌──────────────────────────┴──────────────────────────┐     │
│  │ LLVM IR, PTX (NVIDIA), GCN ISA (AMD), SPIR-V       │     │
│  └──────────────────────────┬──────────────────────────┘     │
│                              │                                 │
│  MACHINE CODE                │                                 │
│  ┌──────────────────────────┴──────────────────────────┐     │
│  │ SASS (NVIDIA), binary (CPU), AMDGPU binary          │     │
│  └─────────────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────────────┘
```

## 🔍 Deep Dive: Each Compiler

### torch.compile + Inductor (PyTorch)

```
The most important compiler for your work as a PyTorch researcher.

┌──────── torch.compile Architecture ───────────────────┐
│                                                         │
│  @torch.compile                                        │
│  def my_model(x):                                      │
│      return model(x)                                    │
│                                                         │
│  Step 1: TorchDynamo (Graph Capture)                   │
│  ├── Intercepts Python bytecode                        │
│  ├── Traces PyTorch operations                         │
│  ├── Handles control flow via graph breaks             │
│  └── Outputs: torch.fx.GraphModule                     │
│                                                         │
│  Step 2: AOTAutograd (Optional: backward graph)        │
│  ├── Traces backward pass                              │
│  └── Decomposes ops to ATen primitives                 │
│                                                         │
│  Step 3: Inductor Backend (Default)                    │
│  ┌────────────────────────────────────────┐            │
│  │  Graph Optimization Passes:            │            │
│  │  • Operator fusion (matmul + bias +    │            │
│  │    activation → single kernel)         │            │
│  │  • Memory planning                     │            │
│  │  • Layout optimization                 │            │
│  │                                         │            │
│  │  Code Generation:                       │            │
│  │  ├── GPU: Generates Triton kernels     │            │
│  │  ├── CPU: Generates C++/OpenMP         │            │
│  │  └── Compiles to binary                │            │
│  └────────────────────────────────────────┘            │
│                                                         │
│  Result: JIT-compiled, fused, optimized execution      │
└─────────────────────────────────────────────────────────┘
```

**Why this matters for quantization**:
```python
# torch.compile can fuse quantized operations!
# Example: AWQ-style dequantize + matmul fusion

# Without compile: 2 separate kernel launches
#   1. dequantize INT4 → FP16 (memory bound)
#   2. FP16 matmul (compute bound)

# With compile + Inductor:
#   1. Single fused kernel: read INT4, dequant in registers, matmul
#   → Eliminates intermediate FP16 tensor in memory!

@torch.compile(mode="max-autotune")
def quantized_linear(x, qweight, scales, zeros):
    # Inductor can fuse this into a single kernel
    weight = (qweight - zeros) * scales  # Dequantize
    return x @ weight.T                   # Matmul
```

### Triton (the Compiler, not the Server)

```
⚠️ NOT the same as NVIDIA Triton Inference Server!

Triton (OpenAI) = A Python-like language for writing GPU kernels

┌──────── Triton Compiler Pipeline ─────────────────────┐
│                                                         │
│  @triton.jit                                           │
│  def matmul_kernel(a_ptr, b_ptr, c_ptr, ...):         │
│      # Python-like syntax, block-level programming     │
│      offs = tl.arange(0, BLOCK_SIZE)                   │
│      a = tl.load(a_ptr + offs)                         │
│      b = tl.load(b_ptr + offs)                         │
│      c = tl.dot(a, b)                                  │
│      tl.store(c_ptr + offs, c)                         │
│                                                         │
│       │                                                 │
│       ▼  Triton Frontend                               │
│  Triton IR (MLIR-based)                                │
│       │                                                 │
│       ▼  Optimization Passes                           │
│  ├── Coalesce memory accesses                          │
│  ├── Software pipelining                               │
│  ├── Shared memory allocation                          │
│  └── Warp-level optimization                           │
│       │                                                 │
│       ▼  Backend                                        │
│  ├── NVIDIA: Triton IR → PTX → SASS (cubin)           │
│  ├── AMD: Triton IR → AMDGPU IR → ISA                 │
│  └── Intel: Triton IR → SPIR-V / Gen ISA              │
│                                                         │
│  Key advantage: Write ONCE, run on NVIDIA/AMD/Intel    │
│  Much easier than raw CUDA, almost as fast             │
└─────────────────────────────────────────────────────────┘
```

**Triton for quantized kernels** (critical for your work):
```python
import triton
import triton.language as tl

@triton.jit
def awq_dequant_matmul_kernel(
    # AWQ INT4 matmul: fused dequantize + GEMM
    x_ptr, qweight_ptr, scales_ptr, zeros_ptr, output_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        # Load INT4 weights (packed in int32, 8 weights per int32)
        qw = tl.load(qweight_ptr + ...)  # INT32 packed
        
        # Dequantize in registers (NO memory write!)
        scales = tl.load(scales_ptr + ...)
        zeros = tl.load(zeros_ptr + ...)
        
        # Unpack INT4 from INT32
        w0 = (qw >> 0) & 0xF
        w1 = (qw >> 4) & 0xF
        # ... unpack all 8
        
        # Dequantize: w_fp16 = (w_int4 - zero) * scale
        w_dequant = (w0 - zeros) * scales
        
        # Load activations
        x = tl.load(x_ptr + ...)
        
        # Accumulate matmul
        acc += tl.dot(x, w_dequant)
    
    # Store result
    tl.store(output_ptr + ..., acc)

# This is essentially what AutoAWQ / vLLM kernels do!
# Triton makes it readable; CUDA equivalent is 500+ lines
```

### XLA (Accelerated Linear Algebra)

```
Google's compiler, primary backend for JAX and TensorFlow

┌──────── XLA Architecture ─────────────────────────────┐
│                                                         │
│  JAX (jax.jit) / TensorFlow (tf.function)              │
│       │                                                 │
│       ▼                                                 │
│  StableHLO (Stable High-Level Operations)              │
│  ├── Hardware-independent IR                            │
│  ├── ~100 operations (dot, conv, reduce, ...)          │
│  └── Supports dynamic shapes                           │
│       │                                                 │
│       ▼                                                 │
│  XLA HLO (XLA's internal IR)                           │
│  ├── Graph optimization passes:                        │
│  │   ├── Algebraic simplification                      │
│  │   ├── Op fusion (element-wise chains)               │
│  │   ├── Buffer assignment (memory planning)           │
│  │   └── Layout assignment (row-major, tiled, etc.)    │
│  │                                                      │
│  │  Quantization in XLA:                                │
│  │  ├── AQT (Accurate Quantized Training) for JAX     │
│  │  ├── INT8 matmul via custom calls                   │
│  │  └── Per-channel quantization supported             │
│  │                                                      │
│       ▼  Target-specific backend                       │
│  ├── GPU: CUDA/cuDNN/cuBLAS calls                      │
│  ├── TPU: TPU-specific instructions (MXU ops)          │
│  ├── CPU: LLVM code generation                         │
│  └── Plugin API for custom accelerators                │
│                                                         │
│  Key: XLA is the ONLY way to target Google TPUs        │
│  JAX + XLA = the "functional compilation" approach     │
└─────────────────────────────────────────────────────────┘
```

### Apache TVM

```
The most hardware-agnostic ML compiler

┌──────── TVM Architecture ─────────────────────────────┐
│                                                         │
│  Input: PyTorch, TF, ONNX, MXNet model                │
│       │                                                 │
│       ▼  Frontend Import                               │
│  Relay IR (high-level, functional graph IR)             │
│  ├── Type-checked, functional                          │
│  ├── Hardware-independent                              │
│  └── Graph-level optimizations (fusion, folding)       │
│       │                                                 │
│       ▼  Lower to TIR                                  │
│  TIR (Tensor IR — loop-level IR)                       │
│  ├── Loop nests for each operator                      │
│  ├── Scheduling primitives:                            │
│  │   ├── split, reorder, vectorize                     │
│  │   ├── compute_at, cache_read/write                  │
│  │   └── parallel, unroll, bind (to GPU threads)       │
│  └── Auto-scheduling: MetaSchedule / AutoTVM          │
│       │                                                 │
│       ▼  Code Generation                               │
│  ├── CUDA (NVIDIA GPUs)                                │
│  ├── OpenCL (mobile GPUs, FPGAs)                       │
│  ├── Metal (Apple)                                      │
│  ├── Vulkan (cross-platform GPU)                       │
│  ├── LLVM (CPUs)                                       │
│  ├── C (microcontrollers — µTVM)                       │
│  ├── Hexagon (Qualcomm)                                │
│  ├── CMSIS-NN (ARM Cortex-M)                           │
│  ├── Ethos-U (ARM NPU)                                 │
│  ├── DRP-AI (Renesas)    ← Your company!               │
│  └── WebGPU / WASM (browser)                           │
│                                                         │
│  Key TVM features:                                      │
│  • Auto-tuning: Search for best schedule per operator  │
│  • µTVM: Compile for bare-metal microcontrollers       │
│  • Quantization: Built-in quantization passes          │
│  • MLC-LLM: LLM runtime built on TVM                  │
└─────────────────────────────────────────────────────────┘
```

**TVM quantization flow**:
```python
# TVM's built-in quantization
import tvm
from tvm import relay

# Import model
mod, params = relay.frontend.from_pytorch(scripted_model, input_shapes)

# Quantize (TVM's own quantization)
with relay.quantize.qconfig(
    calibrate_mode="kl_divergence",
    weight_scale="max",
    global_scale=8.0,
    skip_conv_layers=[0],
):
    quantized_mod = relay.quantize.quantize(mod, params)

# Compile for target
target = tvm.target.Target("cuda")  # or "llvm", "opencl", etc.
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(quantized_mod, target=target, params=params)

# Export
lib.export_library("model.so")
```

### MLIR (Multi-Level Intermediate Representation)

```
MLIR = Framework for building compilers (not a compiler itself!)

Created by Chris Lattner (also created LLVM, Swift, Clang)

┌──────── MLIR Architecture ────────────────────────────┐
│                                                         │
│  MLIR is a COMPILER INFRASTRUCTURE — a set of          │
│  reusable components for building ML compilers.        │
│                                                         │
│  Key concept: DIALECTS                                  │
│  Each dialect defines ops at a different abstraction:   │
│                                                         │
│  ┌─── High-Level Dialects ────┐                        │
│  │ • StableHLO (XLA/JAX ops)  │                        │
│  │ • TOSA (portable ML ops)   │ ← Quantization-aware! │
│  │ • Linalg (linear algebra)  │                        │
│  └──────────────┬─────────────┘                        │
│                  │ Progressive lowering                  │
│  ┌──────────────▼─────────────┐                        │
│  │ • Affine (loop analysis)   │                        │
│  │ • SCF (structured control  │                        │
│  │        flow)               │                        │
│  │ • Vector (SIMD ops)        │                        │
│  └──────────────┬─────────────┘                        │
│                  │                                       │
│  ┌──────────────▼─────────────┐                        │
│  │ • GPU dialect              │                        │
│  │ • LLVM dialect             │ → LLVM IR → machine    │
│  │ • SPIR-V dialect           │   code                 │
│  └────────────────────────────┘                        │
│                                                         │
│  Who uses MLIR?                                         │
│  • XLA (StableHLO → MLIR → TPU/GPU code)              │
│  • Triton (Triton → MLIR → PTX)                       │
│  • IREE (Google's edge compiler via MLIR)              │
│  • ONNX-MLIR (ONNX → MLIR → native code)              │
│  • torch-mlir (PyTorch FX → MLIR)                      │
│  • Many hardware vendors (custom backends)             │
│                                                         │
│  TOSA Dialect (Tensor Operator Set Architecture):       │
│  • Standardized set of ~80 tensor ops                  │
│  • Includes quantized variants (rescale, clamp)        │
│  • Designed for portability across NPUs                │
│  • ARM, Qualcomm, many vendors support TOSA            │
│  → Future of portable quantized model deployment!      │
└─────────────────────────────────────────────────────────┘
```

### CUTLASS & Composable Kernel

> 📎 *See also [Doc 04 — Runtimes & Backends](./04_runtimes_and_backends.md) for how CUTLASS fits into the kernel library comparison table alongside Triton, cuBLAS, and others.*

```
These are NOT compilers, but TEMPLATE LIBRARIES for writing
high-performance kernels. They sit between compiler and hand-tuned code.

┌──────── CUTLASS (NVIDIA) ─────────────────────────────┐
│                                                         │
│  What: C++ template library for GEMM/Conv on NVIDIA GPUs│
│                                                         │
│  Hierarchy:                                              │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Level 0: Tensor Core MMA (hardware instruction)   │  │
│  │ Level 1: Warp-level tile (groups of MMA)          │  │
│  │ Level 2: Threadblock tile (shared memory)         │  │
│  │ Level 3: Device-level (grid launch)               │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
│  Quantization support:                                  │
│  ✅ INT8 × INT8 → INT32 GEMM                          │
│  ✅ FP8 (E4M3/E5M2) GEMM (Hopper+)                   │
│  ✅ INT4 × FP16 mixed-precision                        │
│  ✅ FP4 (Blackwell) — CUTLASS 3.7+                    │
│  ✅ Grouped GEMM (for MoE)                             │
│  ✅ Sparse GEMM (structured sparsity)                  │
│                                                         │
│  Why it matters to you:                                 │
│  AWQ/GPTQ kernels in vLLM use CUTLASS or Marlin       │
│  (Marlin = INT4 GEMM kernel derived from CUTLASS)      │
└─────────────────────────────────────────────────────────┘

┌──────── Composable Kernel (AMD) ──────────────────────┐
│                                                         │
│  What: AMD's equivalent of CUTLASS for ROCm            │
│  Used by: MIOpen, AITER (AMD Inference Triton kernels) │
│  Quantization: INT8 GEMM, FP8 GEMM (MI300X+)         │
│  Notable: AITER provides FP8/INT8 optimized kernels   │
│           specifically for LLM inference on AMD GPUs   │
└─────────────────────────────────────────────────────────┘
```

## 📊 Compiler Comparison Table

| Compiler | Creator | Input | Output Targets | Quantization | LLM Support | Open Source |
|----------|---------|-------|----------------|-------------|-------------|-------------|
| **torch.compile** | Meta | PyTorch | NVIDIA/AMD/Intel GPU, CPU | Via torch.ao | ✅ (vLLM uses) | ✅ |
| **Triton** | OpenAI | Triton DSL | NVIDIA, AMD, Intel GPU | Manual (you write it) | ✅ (custom kernels) | ✅ |
| **XLA** | Google | JAX/TF | TPU, GPU, CPU | AQT, INT8 | ✅ | ✅ |
| **TVM** | Apache | Any framework | 20+ targets inc. MCUs | Built-in passes | ✅ (MLC-LLM) | ✅ |
| **TensorRT** | NVIDIA | ONNX/TRT API | NVIDIA GPU only | FP8, INT8, INT4 | ✅ (TRT-LLM) | ✅ (fully open-sourced March 2025) |
| **OpenVINO** | Intel | ONNX/TF/PT | Intel CPU/GPU/VPU | NNCF INT8/INT4 | ✅ | ✅ |
| **IREE** | Google | MLIR | CPU, GPU, Vulkan, VMVX | TOSA quant | ⚠️ Growing | ✅ |
| **ExecuTorch** | Meta | torch.export | Mobile (CPU, NPU, GPU) | PT2E INT8/INT4 | ✅ (Llama) | ✅ |
| **Vela** | ARM | TFLite | Ethos-U NPU | INT8 only | ❌ | ✅ |

## 🔗 How Compilers Interact with Quantization

```
┌──────── Quantization × Compiler Interaction ──────────┐
│                                                         │
│  APPROACH 1: Pre-quantized weights + custom kernels    │
│  (AWQ, GPTQ approach — what you do)                   │
│                                                         │
│  1. Quantize weights offline (Python, calibration)     │
│  2. Store quantized weights (SafeTensors/GGUF)         │
│  3. At runtime, call hand-written or Triton kernel     │
│     that does dequant+matmul fused                     │
│  4. Compiler DOESN'T do the quantization               │
│     Compiler just compiles the fused kernel            │
│                                                         │
│  APPROACH 2: Compiler-driven quantization              │
│  (TensorRT, TVM, OpenVINO approach)                    │
│                                                         │
│  1. Give FP16/FP32 model to compiler                   │
│  2. Provide calibration data                           │
│  3. Compiler inserts quantization nodes (QDQ)          │
│  4. Compiler fuses quant + compute + dequant           │
│  5. Compiler generates optimized kernel                │
│  → Less control, but automatic optimization            │
│                                                         │
│  APPROACH 3: Hybrid (emerging — torch.compile + quant) │
│                                                         │
│  1. Quantize weights with your algorithm (Python)      │
│  2. Express quantized ops as torch.ao ops              │
│  3. torch.compile + Inductor fuses and optimizes       │
│  → Best of both: your algorithm + compiler fusion      │
│  → This is where the ecosystem is heading!             │
└─────────────────────────────────────────────────────────┘
```

## 🔮 Emerging Trends (2025-2026)

| Trend | Description | Impact on Quantization |
|-------|-------------|----------------------|
| **torch.compile everywhere** | vLLM, SGLang adopting torch.compile for kernel fusion | Your quantized ops can benefit from automatic fusion |
| **MLIR standardization** | TOSA dialect for portable quantized ops | Write quant op once, deploy everywhere |
| **Triton on AMD/Intel** | Triton kernels run on non-NVIDIA GPUs | Quantized Triton kernels become portable |
| **PT2E quantization** | PyTorch 2 Export-based quantization | Unified quantization + compilation pipeline |
| **Custom ops in Inductor** | Register custom quant kernels in torch.compile | Seamless integration of your research |
| **GGML evolution** | ggml adding more quant types, CPU SIMD opt | Pushing limits of CPU quantized inference |

---

**Next**: [12 — Practical Code Guide →](./12_practical_code_guide.md)
