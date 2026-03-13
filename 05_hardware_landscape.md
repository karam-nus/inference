[← Back to Table of Contents](./README.md)

# 05 — Hardware Landscape

> Every piece of silicon that can run your quantized model, and how they differ.

*Last updated: March 2026*

## 📊 Master Hardware Comparison

> **Units note**: "TF" = TeraFLOPS (floating-point operations/sec), used for FP16/FP8 compute. "TOPS" = Tera Operations/sec, used for integer (INT8) compute and NPUs. These are **not directly comparable** — TOPS typically measures INT8 throughput, while TF measures floating-point throughput.

| Vendor | Product Line | Type | Target | Peak TF (FP16) | Peak TF (INT8) | Memory | Power | Quantization Native |
|--------|-------------|------|--------|----------------|----------------|--------|-------|-------------------|
| **NVIDIA** | H100 SXM | Data center GPU | Cloud | 990 TF | 1979 TF | 80GB HBM3 | 700W | FP8, INT8, INT4 |
| **NVIDIA** | B200 | Data center GPU | Cloud | 2250 TF | 4500 TF | 192GB HBM3e | 1000W | FP4, FP8, INT8, INT4 |
| **NVIDIA** | GB200 (Grace+Blackwell) | CPU+GPU Super Chip | Cloud | 2500 TF | 5000 TF | 192GB HBM3e (GPU) + 384GB LPDDR5X (Grace CPU) | 1200W | FP4, FP8, INT8 |
| **NVIDIA** | L40S | Workstation GPU | Inference | 366 TF | 733 TF | 48GB GDDR6 | 350W | FP8, INT8 |
| **NVIDIA** | Jetson Orin | Edge SoC | Edge | 100 TOPS | 200 TOPS | 32-64GB | 15-60W | INT8 |
| | | | | | | | | |
| **AMD** | MI300X | Data center GPU | Cloud | 1307 TF | 2614 TF | 192GB HBM3 | 750W | FP8, INT8 |
| **AMD** | MI355X | Data center GPU | Cloud | ~2500 TF | ~5000 TF | 288GB HBM3e | 800W | FP4, FP8, INT8 |
| **AMD** | Ryzen AI (XDNA NPU) | Laptop NPU | Edge | 50 TOPS | 50 TOPS | Shared | 15W | INT8, INT4 |
| **AMD** | EPYC (Zen 5) | Server CPU | Cloud/Edge | N/A | ~8 TOPS | DDR5 | 400W | INT8 (VNNI) |
| | | | | | | | | |
| **Intel** | Gaudi 3 | AI Accelerator | Cloud | 1835 TF | 3670 TF | 128GB HBM2e | 900W | FP8, INT8 |
| **Intel** | Xeon (Sapphire/Granite) | Server CPU | Cloud | N/A | ~20 TOPS (AMX) | DDR5 | 350W | INT8, BF16 (AMX) |
| **Intel** | Arc B-series GPU | Desktop GPU | Edge/Local | ~50 TF | ~100 TF | 12GB GDDR6 | 150W | INT8, INT4 (XMX) |
| | | | | | | | | |
| **Google** | TPU v5e | Cloud Accelerator | Cloud | 197 TF | 394 TF | 16GB HBM | 200W | INT8, BF16 |
| **Google** | TPU v6e (Trillium) | Cloud Accelerator | Cloud | 918 TF | 1836 TF | 32GB HBM | 300W | INT8, FP8, BF16 |
| | | | | | | | | |
| **Apple** | M4 Max | Laptop SoC | Local/Edge | 53.6 TF | N/A | 128GB unified | 45W | FP16 (ANE: INT8) |
| **Apple** | M4 Ultra | Desktop SoC | Local | 107 TF | N/A | 256GB unified | 75W | FP16 (ANE: INT8) |
| | | | | | | | | |
| **Qualcomm** | Cloud AI 100 Ultra | Cloud Accelerator | Cloud | 870 TOPS INT8 | 870 TOPS | 128GB DDR | 150W | INT8, INT4 |
| **Qualcomm** | Snapdragon 8 Elite / Gen 4 (Hexagon NPU) | Mobile NPU | Mobile | 75 TOPS | 75 TOPS | Shared | 5W | INT8, INT4 |
| | | | | | | | | |
| **AWS** | Trainium2 | Cloud Accelerator | Cloud | ~1500 TF | ~3000 TF | 96GB HBM | 600W | FP8, BF16 |
| **AWS** | Inferentia2 | Inference Accelerator | Cloud | 380 TF | 760 TF | 32GB HBM | 175W | INT8, BF16 |
| | | | | | | | | |
| **Groq** | LPU (GroqChip) | Inference ASIC | Cloud | N/A | 750 TOPS | 230MB SRAM | 300W | INT8 |
| **Cerebras** | WSE-3 | Wafer-scale Chip | Cloud | ~125 PF | N/A | 44GB SRAM | 23kW | FP16, BF16 |

## 🏗️ Hardware Architecture Concepts

### How Tensor Cores / Matrix Engines Work (Why Quantization Matters)

```
┌───────── NVIDIA Tensor Core (Blackwell) ─────────┐
│                                                  │
│  Supported Precisions:                           │
│  ┌─────────┬──────────┬─────────────┐            │
│  │ Input A │ Input B  │ Accumulator │            │
│  ├─────────┼──────────┼─────────────┤            │
│  │ FP64    │ FP64     │ FP64        │            │
│  │ TF32    │ TF32     │ FP32        │            │
│  │ BF16    │ BF16     │ FP32        │            │
│  │ FP16    │ FP16     │ FP16/FP32   │            │
│  │ FP8     │ FP8      │ FP16/FP32   │  ◄── 2x vs FP16
│  │ FP4     │ FP4      │ FP16/FP32   │  ◄── 4x vs FP16 (Blackwell!)
│  │ INT8    │ INT8     │ INT32       │  ◄── 2x vs FP16
│  │ INT4    │ INT4     │ INT32       │  ◄── 4x (limited)
│  └─────────┴──────────┴─────────────┘            │
│                                                  │
│  Matrix size per cycle: 16×16×16 (FP16)          │
│                         16×16×32 (FP8/INT8)      │
│                         16×16×64 (FP4/INT4)      │
│                                                  │
│  KEY INSIGHT: Lower precision = more ops/cycle   │
│  This is WHY quantization gives speedup!         │
└────────────────────────────────────────────────────┘
```

### Memory Hierarchy (Why Quantization Also Saves Memory Bandwidth)

```
┌─────────── GPU Memory Hierarchy ───────────┐
│                                            │
│  Registers        ~20 MB    10 TB/s         │ ◄── Fastest
│       │                                    │
│       ▼                                    │
│  Shared Memory     ~20 MB    10 TB/s       │
│  (L1 Cache)                                │
│       │                                    │
│       ▼                                    │
│  L2 Cache         ~50 MB    5 TB/s         │
│       │                                    │
│       ▼                                    │
│  HBM (DRAM)       80-192GB  2-3 TB/s       │ ◄── Bottleneck!
│                                            │
│  KEY INSIGHT: LLM decoding is MEMORY-BOUND │
│  INT4 weights use 4x less bandwidth than   │
│  FP16, giving ~4x speedup for decode!      │
└─────────────────────────────────────────────┘
```

## 🌐 Hardware → Software Stack Map

```
┌──────────────────────────────────────────────────────────────────────┐
│                    HARDWARE → SOFTWARE MAPPINGS                      │
│                                                                      │
│  ┌─────────────┐    ┌──────────────────────────────────────────┐     │
│  │ NVIDIA GPU   │───►│ CUDA → cuBLAS/CUTLASS/cuDNN             │     │
│  │ (H100/B200)  │    │ → TensorRT / TensorRT-LLM               │     │
│  │              │    │ → vLLM / SGLang / llama.cpp (CUDA)       │    │
│  └─────────────┘    └──────────────────────────────────────────┘     │
│                                                                      │
│  ┌─────────────┐    ┌──────────────────────────────────────────┐     │
│  │ AMD GPU      │───►│ ROCm/HIP → rocBLAS/MIOpen/AITER          │    │
│  │ (MI300X)     │    │ → vLLM / SGLang / llama.cpp (HIP)        │    │
│  └─────────────┘    └──────────────────────────────────────────┘     │
│                                                                      │
│  ┌─────────────┐    ┌──────────────────────────────────────────┐     │
│  │ Intel CPU    │───►│ oneAPI/SYCL → oneDNN/oneMKL (AMX/AVX)    │    │
│  │ (Xeon/Gaudi) │    │ → OpenVINO / ONNX RT / vLLM (CPU)        │    │
│  └─────────────┘    └──────────────────────────────────────────┘     │
│                                                                      │
│  ┌─────────────┐    ┌──────────────────────────────────────────┐     │
│  │ Google TPU   │───►│ XLA/JAX → TPU runtime                     │   │
│  │ (v5e/v6e)    │    │ → vLLM (TPU) / SGLang (JAX backend)      │    │
│  └─────────────┘    └──────────────────────────────────────────┘     │
│                                                                      │
│  ┌─────────────┐    ┌──────────────────────────────────────────┐     │
│  │ Apple Silicon│───►│ Metal → Metal Performance Shaders          │  │
│  │ (M4 Max)     │    │ → llama.cpp (Metal) / MLC-LLM              │  │
│  └─────────────┘    └──────────────────────────────────────────┘     │
│                                                                      │
│  ┌─────────────┐    ┌──────────────────────────────────────────┐     │
│  │ Qualcomm NPU │───►│ QNN SDK → Hexagon DSP/NPU                 │   │
│  │ (Hexagon)    │    │ → ExecuTorch (QNN EP) / ONNX RT (QNN EP)  │   │
│  └─────────────┘    └──────────────────────────────────────────┘     │
│                                                                      │
│  ┌─────────────┐    ┌──────────────────────────────────────────┐     │
│  │ Edge MCU     │───►│ Vendor SDK (Renesas e-AI, Synopsys ARC)   │   │
│  │ (ARM Cortex) │    │ → TFLite Micro / CMSIS-NN / custom        │   │
│  └─────────────┘    └──────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────┘
```

## 📈 Hardware Trends (March 2026)

| Trend | Impact on Quantization Research |
|-------|-------------------------------|
| **Native FP4** on Blackwell | Your W4A4 research now has HW support; FP4 outperforms INT4 |
| **Microscaling (MX) formats** | MXFP4, MXFP6, MXFP8 — new quantization targets with per-block scales |
| **Larger HBM** (192-288GB) | Less pressure to quantize for memory; more focus on speed |
| **AI PCs with NPUs** | Every new laptop has INT8/INT4 NPUs; quantized edge models are mainstream |
| **NVIDIA Grace Hopper/Blackwell CPU** | ARM-based server CPU with coherent GPU memory; new deployment paradigm |
| **Wafer-scale (Cerebras)** | No memory hierarchy bottleneck; quantization less about bandwidth |

## 🎯 Quantization Precision Support by Hardware Generation

| Hardware | FP32 | BF16 | FP16 | FP8 | FP4 | INT8 | INT4 | Binary |
|----------|------|------|------|-----|-----|------|------|--------|
| NVIDIA Ampere (A100) | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ⚠️ | ❌ |
| NVIDIA Hopper (H100) | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ⚠️ | ❌ |
| NVIDIA Blackwell (B200) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| AMD MI300X | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ⚠️ | ❌ |
| AMD MI355X | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Intel Gaudi 3 | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| Intel Xeon (AMX) | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| Google TPU v6e | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ |
| Apple M4 (GPU) | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Apple M4 (ANE) | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| Qualcomm Hexagon | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ |

⚠️ = Software emulation, not native hardware support

---

**Next**: [06 — Quantization to Deployment Bridge →](./06_quantization_to_deployment.md)
