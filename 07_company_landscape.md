[← Back to Table of Contents](./README.md)

*Last updated: March 2026*

# 07 — Company Landscape

> What NVIDIA, Renesas, Synopsys, and others are building — and how it connects to your work.

## 🏢 Company Classification

```
┌─────────────────────── COMPANY ECOSYSTEM MAP ──────────────────────┐
│                                                                     │
│  ┌──────────── SILICON MAKERS ────────────┐                        │
│  │                                         │                        │
│  │  GPU/Accelerator:  NVIDIA, AMD, Intel   │                        │
│  │  TPU/Custom ASIC:  Google, AWS, Groq    │                        │
│  │  Mobile SoC:       Qualcomm, Apple,     │                        │
│  │                    MediaTek, Samsung     │                        │
│  │  Edge/IoT MCU:     Renesas, NXP, STM,   │                        │
│  │                    Infineon, TI          │                        │
│  └─────────────────────────────────────────┘                        │
│                                                                     │
│  ┌──────────── IP / EDA PROVIDERS ────────┐                        │
│  │                                         │                        │
│  │  Synopsys:  NPU IP blocks (ARC NPU),   │                        │
│  │             EDA tools for chip design    │                        │
│  │  Cadence:   Tensilica NPU IP, EDA       │                        │
│  │  ARM:       Ethos NPU IP, CPU cores     │                        │
│  │  Imagination: PowerVR GPU/NPU IP        │                        │
│  │  CEVA:      SensPro NPU IP              │                        │
│  └─────────────────────────────────────────┘                        │
│                                                                     │
│  ┌──────────── SOFTWARE STACK BUILDERS ───┐                        │
│  │                                         │                        │
│  │  Meta:     PyTorch, ExecuTorch          │                        │
│  │  Google:   TensorFlow, JAX, XLA, TFLite │                        │
│  │  Microsoft: ONNX Runtime, DeepSpeed     │                        │
│  │  NVIDIA:   TensorRT-LLM, Triton Server  │                        │
│  │  HuggingFace: Transformers, Optimum     │                        │
│  │  Community: vLLM, SGLang, llama.cpp     │                        │
│  └─────────────────────────────────────────┘                        │
│                                                                     │
│  ┌──────────── CLOUD PROVIDERS ───────────┐                        │
│  │                                         │                        │
│  │  AWS:       Trainium, Inferentia, SageMaker                     │
│  │  Google:    TPU, Vertex AI                                       │
│  │  Azure:     NVIDIA GPUs, ONNX RT        │                        │
│  │  Oracle:    GPU Cloud, AI infra         │                        │
│  └─────────────────────────────────────────┘                        │
└─────────────────────────────────────────────────────────────────────┘
```

## 📊 Detailed Company Profiles

### NVIDIA — The Dominant Force

| Aspect | Details |
|--------|---------|
| **Role** | GPU hardware + full software stack (compiler → runtime → engine → serving) |
| **Hardware** | H100, B200, GB200, L40S, Jetson (edge), Grace (CPU) |
| **Software Stack** | CUDA → cuBLAS/CUTLASS/cuDNN → TensorRT → TensorRT-LLM → Triton Server → NVIDIA Dynamo |
| **Quantization** | Native FP8 (Hopper), FP4 (Blackwell), INT8, INT4; provides NVIDIA ModelOpt toolkit |
| **Key Products** | TensorRT-LLM (inference), Triton Inference Server (serving), NVIDIA Dynamo (orchestration) |
| **Strategy** | Vertically integrated: own the full stack from silicon to serving |
| **Relevance to You** | Primary deployment target; FP4/FP8 research aligns with Blackwell features |
| **Pre-quantized Models** | [nvidia/DeepSeek-R1-FP4](https://huggingface.co/nvidia/DeepSeek-R1-FP4), many on HuggingFace |

### AMD — The Challenger

| Aspect | Details |
|--------|---------|
| **Role** | GPU hardware + ROCm software ecosystem |
| **Hardware** | MI300X (current), MI355X (upcoming), EPYC CPUs, Ryzen AI (XDNA NPU) |
| **Software Stack** | ROCm/HIP → rocBLAS/MIOpen/AITER → Composable Kernel → vLLM/SGLang (ROCm) |
| **Quantization** | FP8 (MI300X), FP4 (MI355X); provides AMD Quark quantization toolkit |
| **Strategy** | ROCm + open ecosystem; heavy investment in vLLM/SGLang ROCm support |
| **Relevance to You** | Growing deployment target; AMD Quark is their quantization toolkit |
| **Key Partnerships** | SGLang (AMD co-develops), vLLM ROCm, Microsoft Azure |

### Intel — The CPU + Accelerator Play

| Aspect | Details |
|--------|---------|
| **Role** | CPU (Xeon), GPU (Arc/Gaudi), edge (Movidius), software (OpenVINO) |
| **Hardware** | Xeon (AMX for INT8/BF16), Gaudi 3 (AI accelerator), Arc GPUs, Movidius VPU |
| **Software Stack** | oneAPI/SYCL → oneDNN/oneMKL → OpenVINO → ONNX RT (Intel EP) |
| **Quantization** | NNCF toolkit (INT8/INT4), AMX hardware INT8, INC (Intel Neural Compressor) |
| **Strategy** | CPU inference leadership + Gaudi for cloud; OpenVINO for optimization |
| **Relevance to You** | If deploying to CPU or Intel GPUs; OpenVINO has good INT8 PTQ |

### Google — TPU + Cloud

| Aspect | Details |
|--------|---------|
| **Role** | Custom TPU hardware + cloud platform + ML frameworks (TF, JAX) |
| **Hardware** | TPU v5e, v6e (Trillium), Cloud GPUs |
| **Software Stack** | JAX/XLA → TPU runtime → Vertex AI |
| **Quantization** | BF16 native, INT8 quantization, AQT (Accurate Quantized Training) |
| **Strategy** | Cloud-first; TPU access via Google Cloud |
| **Relevance to You** | XLA compiler needs quantized op support for TPU deployment |

### Apple — On-Device AI

| Aspect | Details |
|--------|---------|
| **Role** | Consumer hardware (M-series) + on-device ML frameworks |
| **Hardware** | M4, M4 Pro, M4 Max, M4 Ultra (GPU + Neural Engine/ANE) |
| **Software Stack** | Metal → Metal Performance Shaders → Core ML → llama.cpp (Metal backend) |
| **Quantization** | FP16 GPU, INT8 on ANE, Core ML supports palettization/pruning |
| **Strategy** | All AI on-device, privacy-first; no cloud dependency |
| **Relevance to You** | llama.cpp Metal is the primary path for Apple Silicon deployment |

### Renesas — Edge & Automotive AI

> 📎 *See [Doc 10 — Edge & Embedded Deployment](./10_edge_and_embedded.md) for detailed Renesas deployment examples and hardware specs.*

| Aspect | Details |
|--------|---------|
| **Role** | Embedded MCU/MPU maker for automotive, IoT, industrial |
| **Hardware** | RA family (ARM Cortex-M), RZ family (ARM Cortex-A + GPU/DRP-AI), R-Car (automotive) |
| **AI Capabilities** | DRP-AI (Dynamically Reconfigurable Processor for AI) — small NPU in RZ/V series |
| **Software Stack** | e-AI Translator, DRP-AI TVM (TVM-based compiler), TFLite Micro compatible |
| **Quantization** | INT8 primarily, INT4 limited; uses TFLite quantization or custom |
| **Model Types** | Small vision models (MobileNet, YOLO), tiny NLP — NOT LLMs |
| **Strategy** | Bring AI to microcontrollers for edge inference (factory, automotive, camera) |
| **Relevance to You** | If your quantization targets extreme edge (2-bit, mixed-precision for MCUs) |

```
┌──────── Renesas AI Deployment Flow ────────┐
│                                             │
│  TensorFlow/PyTorch Model                   │
│       │                                     │
│       ▼                                     │
│  Quantize to INT8 (TFLite / custom)        │
│       │                                     │
│       ▼                                     │
│  e-AI Translator / DRP-AI TVM Compiler     │
│       │                                     │
│       ▼                                     │
│  Compiled binary for DRP-AI / ARM core     │
│       │                                     │
│       ▼                                     │
│  Runs on Renesas RZ/V MCU (1-10 TOPS)      │
│  (Camera AI, anomaly detection, voice)      │
└─────────────────────────────────────────────┘
```

### Synopsys — The EDA & IP Giant

> 📎 *See [Doc 10 — Edge & Embedded Deployment](./10_edge_and_embedded.md) for how IP blocks like Synopsys ARC NPU fit into edge deployment pipelines.*

| Aspect | Details |
|--------|---------|
| **Role** | EDA tools + IP blocks (including NPU IP) for chip designers |
| **Hardware IP** | ARC NPU IP (licensable NPU cores for custom SoCs), ARC Processor IP |
| **Software Stack** | MetaWare toolkit, Synopsys Neural Network SDK, ONNX support |
| **Quantization** | INT8/INT4 quantization via their SDK; targeted at their ARC NPU |
| **Model Types** | Small vision/audio models for custom SoCs |
| **Strategy** | Sell NPU IP to chip makers (Samsung, etc.) who embed it in their SoCs |
| **Relevance to You** | If your quantization IP could be embedded into custom SoCs via Synopsys IP |

```
┌──────── Synopsys NPU Deployment Flow ──────┐
│                                             │
│  Chip Designer (e.g., Samsung) buys         │
│  Synopsys ARC NPU IP block                 │
│       │                                     │
│       ▼                                     │
│  Integrates NPU into their custom SoC      │
│       │                                     │
│       ▼                                     │
│  App developer uses Synopsys NN SDK to:     │
│  1. Import ONNX/TFLite model               │
│  2. Quantize to INT8/INT4                   │
│  3. Compile for ARC NPU                    │
│  4. Deploy to end-user device               │
│                                             │
│  Use case: Smart camera, wearable,          │
│  hearing aid, automotive sensor             │
└─────────────────────────────────────────────┘
```

### Qualcomm — Mobile & Edge AI Leader

| Aspect | Details |
|--------|---------|
| **Role** | Mobile SoC (Snapdragon), cloud AI (Cloud AI 100), automotive |
| **Hardware** | Hexagon NPU (in Snapdragon), Adreno GPU, Kryo CPU |
| **Software Stack** | QNN (Qualcomm Neural Network) SDK, SNPE (legacy), AI Engine Direct |
| **Quantization** | INT8, INT4 (Hexagon native), mixed-precision |
| **Model Types** | On-device LLMs (up to 7B params), vision, speech |
| **Strategy** | Bring LLMs to phones and PCs; Snapdragon X for AI PCs |
| **Relevance to You** | Mobile/laptop quantization targets; INT4 on Hexagon is key |

### Other Notable Players

| Company | Role | Key Product | Quantization |
|---------|------|-------------|-------------|
| **Groq** | LPU ASIC maker | GroqChip, GroqCloud | INT8 (deterministic inference) |
| **Cerebras** | Wafer-scale compute | WSE-3 | FP16/BF16 (less need for quant) |
| **SambaNova** | Dataflow architecture | SN40L | BF16, INT8 |
| **Graphcore/SoftBank** | IPU architecture | Bow IPU | FP16, INT8 |
| **Tenstorrent** | RISC-V AI hardware | Wormhole, Grayskull | INT8, BF16 |
| **AWS** | Cloud + custom silicon | Trainium2, Inferentia2 | FP8, INT8, NeuronCore |
| **Huawei** | NPU + cloud | Ascend 910B, CANN | FP16, INT8 |
| **Moore Threads** | Chinese GPU | MTT S4000 | FP16, INT8 (MUSA platform) |

## 📈 Market Share & Industry Positioning (March 2026)

```
Data Center AI Accelerator Market Share:

NVIDIA    ████████████████████████████████████████████  ~78%
AMD       ████████                                      ~10%
Google    ████                                           ~4%
Intel     ███                                            ~3%
AWS       ██                                             ~2%
Others    ███                                            ~3%
                                                      (estimated)
```

```
Edge AI Chip Market (by design wins):

Qualcomm  ████████████████████████████████              ~35%
Apple     ██████████████████                            ~20%
MediaTek  ████████████████                              ~15%
Samsung   ██████████                                    ~10%
Intel     ██████                                         ~5%
Renesas   ████                                           ~4%
NXP       ████                                           ~4%
Others    ██████                                         ~7%
                                                      (estimated)
```

## 🔗 How This Maps to Your Work

| Your Research Focus | Most Relevant Companies | Why |
|--------------------|-----------------------|-----|
| W4A16 (AWQ/GPTQ) | NVIDIA, AMD | Tensor core support, kernel libraries |
| W8A8 (SmoothQuant) | Everyone | Universal support across hardware |
| FP8 | NVIDIA (Hopper+), AMD (MI300+) | Native HW FP8 tensor cores |
| FP4 | NVIDIA (Blackwell), AMD (MI355X) | Newest hardware, frontier research |
| Ultra-low-bit (2-bit) | llama.cpp community, Edge chip makers | Local/edge deployment |
| KV-cache quantization | vLLM, SGLang, TRT-LLM | Reduces memory for serving |
| Mixed-precision | NVIDIA (ModelOpt), AMD (Quark) | Hardware increasingly supports mixed formats |
| Structured pruning + quant | Edge NPU makers (Qualcomm, Synopsys, Renesas) | Structured sparsity on specialized hardware |

---

**Next**: [08 — Model Formats & Serialization →](./08_model_formats.md)
