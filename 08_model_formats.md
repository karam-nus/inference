[← Back to Table of Contents](./README.md)

*Last updated: March 2026*

# 08 — Model Formats & Serialization

> Why are there so many model formats? What does each one actually store, and which do you use when?

## 🤔 The Core Question

```
You have a quantized PyTorch model (nn.Module). Now what?

  model = MyQuantizedLLM()    # ← Lives in Python, in GPU memory
  
  HOW do you get it to:
  
  ┌─────────────────────────────────────────┐
  │  vLLM server?        → SafeTensors      │
  │  TensorRT-LLM?       → TRT Engine       │
  │  llama.cpp?           → GGUF             │
  │  ONNX Runtime?        → ONNX             │
  │  Mobile phone?        → TFLite / CoreML  │
  │  Web browser?         → ONNX (WASM)      │
  │  Renesas MCU?         → DRP-AI binary    │
  └─────────────────────────────────────────┘
  
  Answer: You SERIALIZE (export) it to a FORMAT that each runtime understands.
```

## 📊 Master Format Comparison Table

| Format | Extension | Creator | Graph? | Weights? | Metadata? | Cross-Platform? | Primary Use |
|--------|-----------|---------|--------|----------|-----------|-----------------|-------------|
| **SafeTensors** | `.safetensors` | HuggingFace | ❌ No | ✅ Yes | ✅ Limited | ✅ Yes | Weight storage (HF hub standard) |
| **GGUF** | `.gguf` | llama.cpp | ✅ Model config | ✅ Yes (quantized) | ✅ Rich | ✅ Yes | Local/edge LLM inference |
| **ONNX** | `.onnx` | Microsoft/Meta | ✅ Full graph | ✅ Yes | ✅ Yes | ✅ Yes | Cross-platform deployment |
| **TorchScript** | `.pt` | Meta/PyTorch | ✅ Full graph | ✅ Yes | ❌ No | ⚠️ PyTorch only | PyTorch-native export |
| **ExportedProgram** | `.pt2` | Meta/PyTorch | ✅ Full graph (ATen) | ✅ Yes | ✅ Yes | ⚠️ PyTorch 2.x | torch.export (new standard) |
| **TensorRT Engine** | `.engine` / `.plan` | NVIDIA | ✅ Optimized graph | ✅ Yes | ✅ Yes | ❌ GPU-specific | Max NVIDIA GPU perf |
| **TFLite** | `.tflite` | Google | ✅ Full graph | ✅ Yes (quantized) | ✅ Yes | ✅ Yes | Mobile/edge deployment |
| **Core ML** | `.mlpackage` | Apple | ✅ Full graph | ✅ Yes | ✅ Yes | ❌ Apple only | Apple device deployment |
| **OpenVINO IR** | `.xml` + `.bin` | Intel | ✅ Full graph | ✅ Yes | ✅ Yes | ⚠️ Intel-optimized | Intel CPU/GPU deployment |
| **Pickle/PyTorch** | `.pt` / `.bin` | PyTorch | ❌ No | ✅ Yes | ❌ No | ⚠️ Python only | Legacy weight storage |

## 🔍 Deep Dive: Each Format

### SafeTensors — The Modern Weight Standard

```
Purpose: Store model WEIGHTS safely and efficiently (no code, no graph)

┌──────────── .safetensors file structure ─────────────┐
│                                                      │
│  ┌─────────────────────────────┐                     │
│  │      Header (JSON)          │ ← tensor names,     │
│  │   {"model.layers.0.qweight":│    shapes, dtypes,  │
│  │    {"dtype":"int32",        │    byte offsets     │
│  │     "shape":[4096,512],     │                     │
│  │     "data_offsets":[0,N]}   │                     │
│  │   }                         │                     │
│  ├─────────────────────────────┤                     │
│  │                             │                     │
│  │   Raw tensor data           │ ← Memory-mapped,    │
│  │   (contiguous bytes)        │   zero-copy loading │
│  │                             │                     │
│  └─────────────────────────────┘                     │
└───────────────────────────────────────────────────────┘

Key Properties:
  ✅ Safe: No arbitrary code execution (unlike pickle)
  ✅ Fast: Memory-mapped, zero-copy loading
  ✅ Simple: JSON header + raw bytes
  ✅ Shardable: model-00001-of-00004.safetensors
  ❌ No graph: Just weights, model code needed separately
  ❌ No quantization encoding: Stores raw int4/int8 as uint8
```

**How vLLM/SGLang use SafeTensors**:
```python
# The engine knows the MODEL ARCHITECTURE from config.json
# SafeTensors only provides the WEIGHTS

# HuggingFace model repo structure:
# my-quantized-model/
# ├── config.json              ← Architecture (hidden_size, num_layers, etc.)
# ├── quantize_config.json     ← Quantization details (AWQ/GPTQ params)
# ├── model-00001.safetensors  ← Weight shard 1
# ├── model-00002.safetensors  ← Weight shard 2
# └── tokenizer.json           ← Tokenizer

# vLLM internally:
# 1. Reads config.json → knows architecture
# 2. Reads quantize_config.json → knows quant scheme
# 3. Memory-maps safetensors → loads weights
# 4. Reconstructs quantized model in GPU memory
```

### GGUF — The Local Inference King

```
Purpose: Self-contained LLM format for llama.cpp ecosystem

┌──────────── .gguf file structure ────────────────────┐
│                                                      │
│  ┌─────────────────────────────┐                     │
│  │      Magic Number           │ ← "GGUF" identifier │
│  ├─────────────────────────────┤                     │
│  │      Metadata (KV pairs)    │                     │
│  │  general.architecture: llama│                     │
│  │  general.name: "My Model"   │                     │
│  │  llama.context_length: 8192 │                     │
│  │  llama.embedding_length:4096│                     │
│  │  tokenizer.ggml.model: bpe  │ ← EVERYTHING in     │
│  │  tokenizer.ggml.tokens:[..]│    one file          │
│  │  quantization.type: Q4_K_M │                      │
│  ├─────────────────────────────┤                     │
│  │                             │                     │
│  │  Tensor Info Array          │ ← Name, shape,      │
│  │  (name, shape, type, off)   │    quant type per   │
│  │                             │    tensor           │
│  ├─────────────────────────────┤                     │
│  │                             │                     │
│  │  Tensor Data (quantized)    │ ← Mixed quant types │
│  │  Each tensor can have       │    possible (imatrix)│
│  │  different quantization     │                     │
│  │                             │                     │
│  └─────────────────────────────┘                     │
└───────────────────────────────────────────────────────┘
```

**GGUF Quantization Types** (most important for you):

| Type | Bits/Weight | Block Size | Method | Quality |
|------|-------------|-----------|--------|---------|
| `Q2_K` | ~2.6 | 256 | k-quant super blocks | Low (research) |
| `Q3_K_M` | ~3.4 | 256 | k-quant medium | Acceptable |
| `Q4_0` | 4.0 | 32 | Absmax per-block | Decent |
| `Q4_K_M` | ~4.5 | 256 | k-quant medium | Good (popular) |
| `Q5_K_M` | ~5.5 | 256 | k-quant medium | Very good |
| `Q6_K` | ~6.6 | 256 | k-quant | Near-FP16 |
| `Q8_0` | 8.0 | 32 | Per-block symmetric | Excellent |
| `F16` | 16.0 | — | IEEE FP16 | Lossless |
| `IQ4_XS` | ~4.2 | 256 | Importance-matrix | Good (newer) |
| `IQ2_XXS` | ~2.1 | 256 | Importance-matrix | Research-grade |

```
┌──────── How GGUF k-quant super blocks work ────────┐
│                                                    │
│  Super Block (256 weights):                        │
│  ┌─────────────────────────────────────────┐       │
│  │  scale_super (FP16) — shared scale      │       │
│  │  min_super (FP16) — shared min          │       │
│  │  ┌──────────┐ ┌──────────┐ ┌────────┐  │        │
│  │  │ Sub-block │ │ Sub-block │ │  ...   │  │      │
│  │  │ (32 wts)  │ │ (32 wts)  │ │        │  │      │
│  │  │ scale(6b) │ │ scale(6b) │ │        │  │      │
│  │  │ q[0..31]  │ │ q[0..31]  │ │        │  │      │
│  │  └──────────┘ └──────────┘ └────────┘  │        │
│  └─────────────────────────────────────────┘       │
│                                                    │
│  Dequant: w = scale_super × sub_scale × q + min    │
└──────────────────────────────────────────────────────┘
```

**Converting to GGUF**:
```bash
# From HuggingFace SafeTensors → GGUF
python convert_hf_to_gguf.py \
  ./my-awq-model/ \
  --outfile model.gguf \
  --outtype q4_k_m

# Quantize an existing GGUF
./llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M

# With importance matrix (better quality at low bits)
./llama-imatrix -m model-f16.gguf -f calibration.txt -o imatrix.dat
./llama-quantize --imatrix imatrix.dat model-f16.gguf model-iq4_xs.gguf IQ4_XS
```

### ONNX — The Universal Interchange Format

```
Purpose: Represent model GRAPH + WEIGHTS in a portable format

┌──────────── .onnx file (protobuf) ───────────────────┐
│                                                      │
│  ModelProto                                          │
│  ├── ir_version: 9                                   │
│  ├── opset_import: [{domain:"", version:18}]         │
│  ├── producer_name: "pytorch"                        │
│  │                                                   │
│  └── GraphProto                                      │
│      ├── name: "main_graph"                          │
│      ├── input: [TensorProto(name="input_ids",...)]  │
│      ├── output: [TensorProto(name="logits",...)]    │
│      │                                               │
│      ├── node[0]: NodeProto                          │
│      │   ├── op_type: "MatMulNBits"     ← Quant op   │
│      │   ├── inputs: ["hidden", "qweight", "scales"] │
│      │   └── attributes: {K:4096, N:11008, bits:4}   │
│      │                                               │
│      ├── node[1]: NodeProto                          │
│      │   ├── op_type: "Add"                          │
│      │   └── inputs: ["matmul_out", "bias"]          │
│      │                                               │
│      └── initializer: [                              │
│            TensorProto(name="qweight", data=[...]),  │
│            TensorProto(name="scales", data=[...])    │
│          ]                                           │
└────────────────────────────────────────────────────────┘

Key Properties:
  ✅ Full graph: Operations + data flow + weights
  ✅ Cross-platform: Any EP (CUDA, CPU, TRT, OpenVINO...)
  ✅ Quantization support: QDQ nodes, MatMulNBits op
  ✅ Graph optimization: ONNX Runtime applies passes
  ❌ Large files: Graph + weights = large protobuf
  ❌ Op coverage: Not all PyTorch ops have ONNX equivalents
  ❌ Dynamic shapes: Improved but can be tricky
```

**Exporting quantized models to ONNX**:
```python
# Option 1: via Optimum (recommended for HF models)
from optimum.onnxruntime import ORTModelForCausalLM
model = ORTModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-AWQ",
    export=True
)
model.save_pretrained("./llama2-7b-onnx/")

# Option 2: Direct torch.onnx.export
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=18,
    dynamic_axes={"input_ids": {0: "batch", 1: "seq_len"}},
    input_names=["input_ids"],
    output_names=["logits"]
)

# Option 3: Using torch.export + ONNX (PyTorch 2.x)
exported = torch.export.export(model, (dummy_input,))
onnx_program = torch.onnx.export(exported, "model.onnx")
```

### TensorRT Engine — Maximum NVIDIA Performance

```
Purpose: GPU-specific compiled/optimized inference binary

┌──────── TensorRT Compilation Pipeline ──────────┐
│                                                 │
│  PyTorch Model / ONNX / TRT-LLM Python          │
│       │                                         │
│       ▼                                         │
│  TensorRT Builder                               │
│  ├── Parse network graph                        │
│  ├── Layer fusion (Conv+BN+ReLU → 1 kernel)     │
│  ├── Precision calibration (FP32→FP16/INT8/FP8) │
│  ├── Kernel auto-tuning (try all implementations) │
│  ├── Memory optimization (layer-by-layer alloc) │
│  └── Build serialized engine                    │
│       │                                         │
│       ▼                                         │
│  .engine / .plan file                           │
│  ┌──────────────────────────────────┐           │
│  │  Serialized CUDA kernels         │           │
│  │  Memory allocation plan          │           │
│  │  Optimized for THIS specific GPU │  ← Not    │
│  │  (e.g., H100 SXM 80GB)         │    portable!│
│  │  Fused operations               │            │
│  │  Weights in optimized layout    │            │
│  └──────────────────────────────────┘           │
└────────────────────────────────────────────────────┘

⚠️ Key Limitation: Engine is tied to:
  - Specific GPU architecture (e.g., sm_90 for H100)
  - TensorRT version
  - Max batch size / sequence length (compiled in)
  → Must rebuild for different GPU or config!
```

### torch.export / ExportedProgram — The Future of PyTorch

```
Purpose: PyTorch 2.x's new export system (replacing TorchScript)

┌──────── torch.export Pipeline ──────────────────┐
│                                                 │
│  model = MyQuantizedLLM()                       │
│       │                                         │
│       ▼ torch.export.export(model, inputs)      │
│                                                 │
│  ExportedProgram                                │
│  ├── graph_module: torch.fx.GraphModule         │
│  │   └── Full ATen-level IR graph               │
│  │       (torch.ops.aten.mm, aten.add, etc.)    │
│  ├── range_constraints: dynamic shape specs     │
│  ├── state_dict: model weights                  │
│  └── module_call_graph: call hierarchy          │
│       │                                         │
│       ├──→ torch.compile (JIT, Inductor backend)│
│       ├──→ ExecuTorch (mobile/edge)             │
│       ├──→ ONNX export                          │
│       └──→ TensorRT (via Torch-TensorRT)        │
└────────────────────────────────────────────────────┘

Key Advantage: Single export, multiple backends
```

### TFLite — Mobile & Edge Standard

```
Purpose: Lightweight format for mobile/embedded inference

Structure:
  FlatBuffers schema (fast loading, no parsing)
  ├── Model metadata
  ├── Operator codes (built-in + custom)
  ├── Subgraphs (usually 1)
  │   ├── Tensors (with quantization parameters inline)
  │   │   └── {name, shape, type:INT8, quantization:{scale, zero_point}}
  │   ├── Operators
  │   │   └── {opcode, inputs, outputs, builtin_options}
  │   ├── Inputs / Outputs
  │   └── Buffers (weight data)
  └── Buffers (raw tensor data)

Quantization in TFLite:
  ✅ INT8 per-tensor and per-channel (gold standard)
  ✅ INT16×INT8 mixed precision
  ✅ Dynamic range quantization (weights-only INT8)
  ✅ Float16 quantization
  ❌ No INT4 natively (INT4 support experimental)
```

## 🔄 Format Conversion Map

```
┌──────────────── FORMAT CONVERSION PATHS ─────────────────┐
│                                                          │
│                    PyTorch nn.Module                     │
│                    ┌──────────┐                          │
│                    │ In Memory│                          │
│                    └────┬─────┘                          │
│          ┌──────────────┼────────────────┐               │
│          │              │                │               │
│          ▼              ▼                ▼               │
│    SafeTensors     torch.export      torch.onnx          │
│    (weights only)  (ExportedProg)    (.onnx file)        │
│      │               │    │              │               │
│      │               │    │         ┌────┼────┐          │
│      ▼               │    │         │    │    │          │
│   vLLM/SGLang       │    │         │    │    ▼           │
│   (load weights     │    │         │    │  ONNX RT       │
│    into engine's    │    │         │    │  (any EP)      │
│    own graph)       │    │         │    │                │
│                      │    │         │    ▼               │
│                      │    ▼         │  OpenVINO IR       │
│                      │  ExecuTorch │  (Intel)            │
│                      │  (.pte)     │                     │
│                      │             ▼                     │
│                      ▼           TensorRT                │
│                   Torch-TRT      Engine                  │
│                   (.engine)      (.engine)               │
│                                                          │
│   Separate path (not from PyTorch directly):             │
│                                                          │
│   HuggingFace SafeTensors ──→ convert_hf_to_gguf.py      │
│                                      │                   │
│                                      ▼                   │
│                                    GGUF                  │
│                                 (llama.cpp)              │
│                                                          │
│   TFLite ← TensorFlow model (separate ecosystem)         │
│   CoreML ← coremltools.convert(pytorch_model)            │
└────────────────────────────────────────────────────────────┘
```

## ⚡ Format Selection Decision Tree

```
What is your deployment target?

├─ NVIDIA GPU (cloud/server)?
│   ├─ Need max throughput? → TensorRT Engine (via TRT-LLM)
│   ├─ Need flexibility?   → SafeTensors + vLLM/SGLang
│   └─ Cross-GPU support?  → ONNX + TensorRT EP
│
├─ Multiple GPU vendors (NVIDIA + AMD + Intel)?
│   └─ ONNX + ONNX Runtime (multiple EPs)
│
├─ Local/Desktop (Mac/Windows/Linux)?
│   └─ GGUF + llama.cpp (or Ollama)
│
├─ Android/iOS mobile?
│   ├─ Android → TFLite or ExecuTorch
│   ├─ iOS     → Core ML or ExecuTorch
│   └─ Both    → ExecuTorch (PyTorch native)
│
├─ Web browser?
│   └─ ONNX + ONNX Runtime Web (WASM/WebGPU)
│
├─ Embedded MCU (Renesas, NXP, STM)?
│   └─ TFLite Micro or vendor-specific format
│
└─ Research/experimentation?
    └─ SafeTensors (easy load/save, HF ecosystem)
```

## 🔬 Quantization Information in Each Format

| Format | How Quant Info is Stored | Dequant at Runtime? |
|--------|------------------------|-------------------|
| **SafeTensors** | External `quantize_config.json`; weights stored as raw int32/uint8 packing | Engine unpacks using config |
| **GGUF** | Per-tensor `ggml_type` field (Q4_K_M, etc.); block structure embedded | llama.cpp dequantizes per block |
| **ONNX** | QDQ nodes (QuantizeLinear/DequantizeLinear) or MatMulNBits op | Runtime fuses QDQ or runs quant kernel |
| **TRT Engine** | Compiled into fused CUDA kernels; precision per-layer | Already compiled, no dequant needed |
| **TFLite** | Per-tensor `quantization` field (scale, zero_point) | Built into operator implementation |
| **ExportedProgram** | Quantized ATen ops in graph (quantized_decomposed::*) | Backend-specific handling |

## 📏 File Size Comparison (Llama 2 7B)

| Format | Precision | Approx. Size | Notes |
|--------|-----------|-------------|-------|
| SafeTensors | FP16 | ~13.5 GB | 2 shards |
| SafeTensors | AWQ INT4 | ~4.0 GB | + quantize_config.json |
| GGUF | Q4_K_M | ~4.1 GB | Single file, includes tokenizer |
| GGUF | Q2_K | ~2.7 GB | Lower quality |
| GGUF | Q8_0 | ~7.2 GB | Near-lossless |
| ONNX | FP16 | ~14 GB | Includes graph |
| ONNX | INT4 (MatMulNBits) | ~4.5 GB | Includes graph |
| TRT Engine | FP8 | ~8 GB | GPU-specific |
| TFLite | INT8 | ~7 GB | Mobile format *(estimated — TFLite is rarely used for 7B LLMs; included for comparison)* |

---

**Next**: [09 — Serving & Middleware →](./09_serving_and_middleware.md)
