[← Back to Table of Contents](./README.md)

*Last updated: March 2026*

# 10 — Edge & Embedded Deployment

> Taking AI from the data center to phones, cameras, cars, and microcontrollers.

## 🌍 The Edge Spectrum

```
┌──────────────── EDGE DEPLOYMENT SPECTRUM ────────────────────┐
│                                                              │
│  ◄─── MORE CAPABLE ─────────────────── MORE CONSTRAINED ───► │
│                                                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐ ┌─────┐  │
│  │  Desktop  │ │  Phone   │ │  Smart   │ │  IoT    │ │ MCU │ │
│  │  / AI PC  │ │  / Tab   │ │  Camera  │ │ Gateway │ │     │ │
│  ├──────────┤ ├──────────┤ ├──────────┤ ├─────────┤ ├─────┤  │
│  │M4 Ultra  │ │Snapdrgn │ │ MediaTek │ │ Renesas │ │ARM  │   │
│  │RTX 4090  │ │8 Elite  │ │ Dimensity│ │ RZ/V2H │ │M0/M4│    │
│  │Arc GPU   │ │Exynos   │ │ Ambarella│ │ NXP     │ │     │   │
│  ├──────────┤ ├──────────┤ ├──────────┤ ├─────────┤ ├─────┤  │
│  │32-192GB  │ │8-16 GB  │ │1-4 GB   │ │256MB-2GB│ │64KB-│    │
│  │RAM       │ │RAM      │ │RAM      │ │RAM      │ │2MB  │    │
│  ├──────────┤ ├──────────┤ ├──────────┤ ├─────────┤ ├─────┤  │
│  │LLM 70B  │ │LLM 7B   │ │YOLO,    │ │Small CNN│ │Tiny │     │
│  │Diffusion│ │LLM 3B   │ │MobileNet│ │Anomaly  │ │KWS  │     │
│  │Any model │ │ViT      │ │ResNet   │ │detection│ │Wake │    │
│  ├──────────┤ ├──────────┤ ├──────────┤ ├─────────┤ ├─────┤  │
│  │llama.cpp │ │ExecuTrch│ │TFLite   │ │TFLite   │ │TFLM │    │
│  │ONNX RT  │ │CoreML   │ │SNPE     │ │OpenVINO │ │CMSIS│     │
│  │vLLM     │ │MLC-LLM  │ │DRP-AI   │ │DRP-AI   │ │-NN  │     │
│  └──────────┘ └──────────┘ └──────────┘ └─────────┘ └─────┘  │
│                                                              │
│  Power:  100-500W    5-15W     1-5W      0.1-2W    0.01W     │
│  TOPS:   100-1000    10-50     2-20      0.5-10    0.01-1    │
└────────────────────────────────────────────────────────────────┘
```

## 📱 Mobile & Desktop Edge Platforms

### ExecuTorch (Meta / PyTorch) — The New Standard

| Aspect | Details |
|--------|---------|
| **What** | PyTorch's official on-device inference framework (successor to PyTorch Mobile) |
| **Status** | Production-ready (2024+), actively developed |
| **Input** | torch.export → ExportedProgram → `.pte` file |
| **Backends** | XNNPACK (CPU), CoreML (Apple), QNN (Qualcomm), Vulkan (GPU), MPS (Metal) |
| **Quantization** | PT2E quantization (PyTorch 2 Export-based), INT8, INT4 |
| **Model Types** | LLMs (Llama 3), diffusion, vision, audio |

```
┌──────── ExecuTorch Pipeline ──────────────────┐
│                                               │
│  PyTorch nn.Module                            │
│       │                                       │
│       ▼  torch.export.export()                │
│  ExportedProgram (ATen IR graph)              │
│       │                                       │
│       ▼  Quantize (PT2E flow)                 │
│  Quantized ExportedProgram                    │
│       │                                       │
│       ▼  to_edge() + to_backend()             │
│  Edge Dialect (optimized for target)          │
│       │                                       │
│       ├──→ XNNPACK delegate (CPU, any platform) │
│       ├──→ CoreML delegate (Apple devices)    │
│       ├──→ QNN delegate (Qualcomm Hexagon)    │
│       └──→ Vulkan delegate (Android GPU)      │
│       │                                       │
│       ▼  to_executorch()                      │
│  .pte file (portable, self-contained)         │
│       │                                       │
│       ▼  ExecuTorch Runtime (C++)             │
│  Runs on device (iOS/Android/Embedded)        │
└─────────────────────────────────────────────────┘
```

```python
# ExecuTorch quantization + export example
import torch
from torch.export import export
from executorch.exir import to_edge
from torch.ao.quantization.quantize_pt2e import (
    prepare_pt2e, convert_pt2e
)
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer, get_symmetric_quantization_config
)

# 1. Export model
model = MyModel()
example_inputs = (torch.randn(1, 3, 224, 224),)
exported = export(model, example_inputs)

# 2. Quantize (PT2E flow)
quantizer = XNNPACKQuantizer().set_global(
    get_symmetric_quantization_config()  # INT8 symmetric
)
prepared = prepare_pt2e(exported, quantizer)
prepared(*example_inputs)  # Calibration
quantized = convert_pt2e(prepared)

# 3. Lower to edge + delegate
edge = to_edge(quantized)
edge = edge.to_backend(XnnpackPartitioner())

# 4. Export to .pte
et_program = edge.to_executorch()
with open("model.pte", "wb") as f:
    f.write(et_program.buffer)
```

### Core ML (Apple)

| Aspect | Details |
|--------|---------|
| **What** | Apple's ML framework for all Apple devices |
| **Hardware** | CPU, GPU (Metal), Neural Engine (ANE) |
| **Input** | `.mlpackage` or `.mlmodelc` (compiled) |
| **Quantization** | Palettization (2/4/6/8-bit lookup), linear quant, pruning |
| **LLM Support** | Via `MLModel` with stateful KV-cache (iOS 18+) |
| **Tools** | `coremltools` (Python) for conversion |

```python
# Converting PyTorch → Core ML
import coremltools as ct

# For vision models
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=(1, 3, 224, 224))],
    compute_units=ct.ComputeUnit.ALL,  # CPU + GPU + ANE
    minimum_deployment_target=ct.target.iOS17
)

# For LLMs (stateful, iOS 18+)
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=(1, ct.RangeDim(1, 2048)))],
    states=[
        ct.StateType(
            wrapped_type=ct.TensorType(shape=(1, 32, 2048, 128)),
            name="kv_cache"
        )
    ],
    compute_precision=ct.precision.FLOAT16,
)

# Quantize
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights
)
config = OptimizationConfig(
    global_config=OpLinearQuantizerConfig(mode="linear_symmetric", weight_threshold=512)
)
mlmodel_quantized = linear_quantize_weights(mlmodel, config)
mlmodel_quantized.save("model_int8.mlpackage")
```

### Qualcomm QNN SDK (Mobile NPU)

```
┌──────── Qualcomm AI Stack ────────────────────┐
│                                               │
│  PyTorch / TF / ONNX model                    │
│       │                                       │
│       ▼                                       │
│  Qualcomm AI Hub (cloud compilation service)  │
│  or QNN SDK (local)                           │
│       │                                       │
│       ▼ Compile for target SoC                │
│  ┌─────────────────────────────────────┐      │
│  │ Snapdragon 8 Gen 3                   │     │
│  │ ┌─────────┐ ┌────────┐ ┌────────┐  │       │
│  │ │ Hexagon  │ │ Adreno │ │  Kryo  │  │      │
│  │ │ NPU      │ │ GPU    │ │  CPU   │  │      │
│  │ │ 73 TOPS  │ │        │ │        │  │      │
│  │ │ INT4/8   │ │ FP16   │ │ INT8   │  │      │
│  │ └─────────┘ └────────┘ └────────┘  │       │
│  └─────────────────────────────────────┘      │
│                                               │
│  Quantization:                                │
│  • Native INT4 on Hexagon (ideal for LLMs)    │
│  • INT8, INT16 per-channel                    │
│  • Mixed-precision across NPU/GPU/CPU         │
│                                               │
│  LLM capability (2024+):                      │
│  • Llama 3 8B runs on Snapdragon X Elite      │
│  • Phi-3 mini runs on Snapdragon 8 Gen 3      │
│  • ~30 tokens/sec on-device                   │
└─────────────────────────────────────────────────┘
```

### llama.cpp — Universal Edge LLM Runtime

```
Why llama.cpp is the king of edge LLM deployment:

✅ Pure C/C++ — no Python, no framework dependencies
✅ 17+ backends — Metal, CUDA, Vulkan, SYCL, CPU...
✅ Aggressive quantization — Q2 to Q8, importance matrix
✅ 96,000+ GitHub stars — massive community
✅ Ollama, LM Studio, Jan built on top of it

Performance examples (Apple M4 Max, 128GB):
┌──────────────────────────────────────────────────┐
│ Model              │ Quant   │ Size  │ tok/s     │
│────────────────────│─────────│───────│────────── │
│ Llama 3 8B         │ Q4_K_M  │ 4.9GB │ ~55 t/s   │
│ Llama 3 70B        │ Q4_K_M  │ 40GB  │ ~12 t/s   │
│ Mistral 7B         │ Q5_K_M  │ 5.1GB │ ~50 t/s   │
│ Phi-3 mini 3.8B    │ Q4_0    │ 2.2GB │ ~80 t/s   │
│ DeepSeek-R1 671B   │ Q2_K    │ 200GB │ ~2 t/s    │
└──────────────────────────────────────────────────┘
```

### MLC-LLM — Compiler-Based Edge Deployment

| Aspect | Details |
|--------|---------|
| **What** | TVM-based universal LLM deployment (compile for any backend) |
| **Approach** | Compile model → optimized native code for target |
| **Backends** | CUDA, Metal, Vulkan, OpenCL, WebGPU, WASM |
| **Key Advantage** | WebGPU support (run LLMs in browser!) |
| **Quantization** | q4f16_1 (INT4 weights, FP16 activations), q4f32_1 |

## 🔩 Embedded & IoT Deployment

### TensorFlow Lite Micro (TFLM)

```
Purpose: Run tiny ML models on microcontrollers (no OS required!)

┌──────── TFLM Architecture ────────────────────┐
│                                               │
│  TFLite Model (.tflite, INT8 quantized)       │
│       │                                       │
│       ▼                                       │
│  TFLM Interpreter (C++, ~20KB binary)         │
│  ├── Flatbuffer parser (model loading)        │
│  ├── Memory planner (static arena allocation) │
│  ├── Op resolver (registered kernels only)    │
│  └── Kernel implementations:                  │
│      ├── Reference (portable, slow)           │
│      ├── CMSIS-NN (ARM Cortex-M optimized)    │
│      └── Custom (vendor-specific)             │
│       │                                       │
│       ▼                                       │
│  Runs on:                                     │
│  • ARM Cortex-M0/M4/M7/M33/M55                │
│  • ESP32, Arduino                             │
│  • Renesas RA family                          │
│  • Any 32-bit MCU with 64KB+ Flash            │
│                                               │
│  Model size limits: ~100KB - 2MB typically    │
│  Inference time: 10ms - 1000ms per inference  │
└─────────────────────────────────────────────────┘

Typical TFLM models:
  • Keyword spotting ("Hey Siri"): ~20KB, INT8
  • Anomaly detection: ~50KB, INT8
  • Person detection: ~300KB, INT8
  • Gesture recognition: ~100KB, INT8
  ⚠️ NOT for LLMs — far too small
```

### Renesas DRP-AI (Detailed)

```
┌──────── Renesas AI Deployment Stack ──────────┐
│                                               │
│  Target Hardware:                             │
│  ┌─────────────────────────────────────────┐  │
│  │ RZ/V2H (flagship AI MPU)                │  │
│  │ ├── ARM Cortex-A55 (quad core)          │  │
│  │ ├── ARM Cortex-M33 (real-time)          │  │
│  │ ├── Mali-G31 GPU                        │  │
│  │ ├── DRP-AI3: 20 TOPS (INT8)            │   │
│  │ ├── ISP (Image Signal Processor)        │  │
│  │ └── 4GB LPDDR4x                         │  │
│  └─────────────────────────────────────────┘  │
│                                               │
│  DRP-AI = Dynamically Reconfigurable Processor│
│  • Programmable AI accelerator                │
│  • Reconfigures its datapath per-layer        │
│  • INT8 multiply-accumulate                   │
│  • Power efficient: ~3 TOPS/W                 │
│                                               │
│  Software Flow:                               │
│  1. Train in PyTorch/TF → export ONNX         │
│  2. DRP-AI Translator (ONNX → DRP-AI binary)  │
│     OR                                        │
│     DRP-AI TVM (Apache TVM backend for DRP-AI)│
│  3. Compile for target RZ/V chip              │
│  4. Deploy with Renesas e-AI SDK              │
│                                               │
│  Supported Operations:                        │
│  ✅ Conv2D, DepthwiseConv, FC, BatchNorm       │
│  ✅ ReLU, Sigmoid, Softmax                     │
│  ✅ MaxPool, AvgPool, GlobalAvgPool            │
│  ⚠️ Limited attention support                 │
│  ❌ No transformer blocks (no LLMs)            │
│                                               │
│  Use Cases:                                   │
│  • Factory inspection cameras                 │
│  • Automotive ADAS (object detection)         │
│  • Smart home cameras (person detection)      │
│  • Industrial anomaly detection               │
│  • Robotics (vision + control)                │
└─────────────────────────────────────────────────┘
```

### NXP i.MX RT (Edge AI)

| Aspect | Details |
|--------|---------|
| **Hardware** | i.MX 8M Plus (2.3 TOPS NPU), i.MX 93 (Ethos-U65 NPU) |
| **Software** | eIQ ML toolkit, TFLite, ONNX RT, Glow compiler |
| **Quantization** | INT8 (Ethos-U optimized), INT16 |
| **Use Cases** | Smart home, industrial HMI, voice assistants |

### ARM Ethos NPU IP

```
ARM Ethos = licensable NPU IP blocks (like ARM CPU cores)

┌──────── ARM Ethos Family ─────────────────────┐
│                                               │
│  Ethos-U55: 32-512 MAC/cycle                  │
│  ├── For Cortex-M class MCUs                  │
│  ├── INT8/INT16, <1 TOPS                      │
│  ├── ~0.1mm² area                             │
│  └── Used in: STM32N6, NXP                    │
│                                               │
│  Ethos-U65: 256-512 MAC/cycle                 │
│  ├── For Cortex-M/A class (more capable)      │
│  ├── INT8/INT16, 1-4 TOPS                     │
│  └── Used in: NXP i.MX 93, Samsung Exynos     │
│                                               │
│  Ethos-U85: Next-gen (2024+)                  │
│  ├── 2× perf of U65                           │
│  ├── INT8, INT4 support                       │
│  └── Transformer-optimized (attention support)│
│                                               │
│  Software: Vela compiler (TFLite → Ethos)     │
│  Input: TFLite INT8 quantized model           │
│  Output: Optimized TFLite with Ethos ops      │
└─────────────────────────────────────────────────┘
```

## 📊 Edge Quantization Comparison

| Platform | INT8 | INT4 | FP16 | Best Quant Method | Model Size Limit |
|----------|------|------|------|-------------------|-----------------|
| **Apple M4** | ✅ | ✅ (llama.cpp) | ✅ | GGUF Q4_K_M | 128GB unified |
| **Snapdragon 8 Gen3** | ✅ | ✅ (Hexagon) | ✅ (GPU) | QNN INT4 | ~16GB |
| **Renesas RZ/V2H** | ✅ | ❌ | ❌ | TFLite INT8 PTQ | ~50MB model |
| **ARM Ethos-U55** | ✅ | ❌ | ❌ | TFLite INT8 per-ch | ~2MB model |
| **ARM Ethos-U85** | ✅ | ✅ | ❌ | Vela INT8/INT4 | ~10MB model |
| **ESP32-S3** | ✅ (slow) | ❌ | ❌ | TFLM INT8 | ~500KB model |
| **Intel Movidius** | ✅ | ❌ | ✅ | OpenVINO INT8 | ~100MB model |
| **Google Edge TPU** | ✅ (only) | ❌ | ❌ | TFLite full INT8 | ~100MB model |

## 🔬 Relevance to Your Quantization Research

```
┌──────── How Your Research Connects to Edge ──────────┐
│                                                      │
│  YOUR ALGORITHMS        WHERE THEY DEPLOY            │
│  ──────────────         ──────────────────           │
│                                                      │
│  AWQ (W4A16)     ──→   Desktop/Server GPUs           │
│                         (CUDA tensor cores)          │
│                         NOT edge MCUs                │
│                                                      │
│  SmoothQuant     ──→   Server GPUs, some mobile NPUs │
│  (W8A8)                (universal support)           │
│                                                      │
│  Ultra-low-bit   ──→   Edge devices, mobile phones   │
│  (2-4 bit)              (biggest impact here!)       │
│                                                      │
│  KV-cache quant  ──→   LLM serving (server)          │
│                         Mobile LLM (limited memory)  │
│                                                      │
│  Mixed-precision ──→   Edge NPUs (different precisions │
│                         for different layers)        │
│                                                      │
│  GAP IN THE FIELD:                                   │
│  Most quant research targets NVIDIA GPUs.            │
│  Edge NPUs (Hexagon, Ethos, DRP-AI) have different   │
│  constraints:                                        │
│  • Per-channel scales (not per-group)                │
│  • No FP16 activations (INT8 only on many NPUs)      │
│  • Power budget matters more than latency            │
│  • Must fit in SRAM (no HBM!)                        │
│  → W4A8 or W8A8 research for edge NPUs is valuable   │
└────────────────────────────────────────────────────────┘
```

## 🧭 Edge Deployment Decision Tree

```
What are you deploying?

├─ LLM (> 1B params)?
│   ├─ Desktop/Laptop → llama.cpp (GGUF) or MLC-LLM
│   ├─ Phone (high-end) → ExecuTorch or MLC-LLM (INT4)
│   ├─ Phone (mid-range) → Smaller model (Phi-3 mini) + INT4
│   └─ MCU → ❌ Not feasible (need >500MB RAM minimum)
│
├─ Vision model (classification/detection)?
│   ├─ Phone → ExecuTorch or TFLite (INT8)
│   ├─ Smart camera → TFLite or vendor SDK (INT8)
│   ├─ Automotive → Vendor SDK (Renesas, NXP, TI)
│   └─ MCU → TFLM + CMSIS-NN (INT8, tiny models only)
│
├─ Audio/Speech model?
│   ├─ Phone → Core ML / ExecuTorch (INT8)
│   ├─ Smart speaker → TFLite (INT8)
│   └─ MCU → TFLM (keyword spotting only)
│
└─ Anomaly detection / time-series?
    ├─ Gateway → TFLite or OpenVINO (INT8)
    └─ MCU → TFLM (tiny autoencoder)
```

---

**Next**: [11 — Compiler Stack →](./11_compiler_stack.md)
