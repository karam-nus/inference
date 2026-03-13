[← Back to Table of Contents](./README.md)

# 06 — Quantization to Deployment Bridge

> The most relevant document for YOU — how your quantization research connects to real-world deployment.

*Last updated: March 2026*

## 🎯 The Gap Between Research and Production

```
┌─────── Your Research World ─────────┐     ┌─────── Production World ─────────┐
│                                      │     │                                 │
│  • PyTorch nn.Module                 │     │  • Thousands of concurrent users│
│  • Calibration on small dataset      │     │  • <100ms latency target        │
│  • Measure perplexity, accuracy      │     │  • 24/7 availability            │
│  • Run on 1-8 GPUs                   │     │  • Cost per token matters       │
│  • Batch size 1-32                   │     │  • Multi-GPU, multi-node        │
│  • Python everywhere                 │     │  • C++/CUDA kernels             │
│                                      │     │  • Specific hardware targets    │
│                                      │ GAP │                                 │
│  You develop:                        │ ◄─► │  They need:                     │
│  • Quantization algorithm            │     │  • Optimized CUDA/GPU kernels   │
│  • Scale/zero-point computation      │     │  • Serialized model format      │
│  • Weight packing scheme             │     │  • Inference engine integration │
│  • Calibration strategy              │     │  • Hardware-specific dispatch   │
│                                      │     │  • Serving infrastructure       │
└──────────────────────────────────────┘     └───────────────────────────────────┘
```

## 🔄 The Complete Pipeline: Research → Production

### Phase 1: Algorithm Development (Your Domain)

```python
# Step 1: Develop your quantization algorithm
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-70B")

# Your algorithm: compute scales, quantize weights
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        # AWQ: find optimal per-channel scales using activation awareness
        scales = compute_awq_scales(module, calibration_data)
        # Quantize weights
        module.weight.data = quantize_w4a16(module.weight.data, scales)
        # Store quantization metadata
        module.scales = scales
        module.group_size = 128
        module.bits = 4
```

### Phase 2: Serialize to Deployment Format

```python
# Step 2a: Save as SafeTensors (for vLLM/SGLang)
from safetensors.torch import save_file

state_dict = {}
for name, param in model.named_parameters():
    state_dict[name] = param
for name, module in model.named_modules():
    if hasattr(module, 'scales'):
        state_dict[f"{name}.scales"] = module.scales
        state_dict[f"{name}.zeros"] = module.zeros

save_file(state_dict, "model.safetensors")

# Save quantization config
import json
quant_config = {
    "quant_method": "awq",
    "bits": 4,
    "group_size": 128,
    "zero_point": True,
    "version": "gemm"  # or "gemv" or "marlin"
}
with open("quantize_config.json", "w") as f:
    json.dump(quant_config, f)
```

```python
# Step 2b: Export as ONNX (for ONNX Runtime)
import torch.onnx

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=17,
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch", 1: "seq_len"}}
)
# Note: Custom quantized ops need ONNX custom op registration!
```

```bash
# Step 2c: Convert to GGUF (for llama.cpp)
python convert_hf_to_gguf.py \
    --model-dir ./my_quantized_model \
    --outtype q4_k_m \
    --outfile model-q4_k_m.gguf
# Note: This RE-QUANTIZES using ggml's scheme, not yours!
```

### Phase 3: Kernel Integration (The Critical Bridge)

```
┌─────── THIS IS THE HARDEST PART ──────────────────────────────────┐
│                                                                   │
│  Your quantized format: W4A16 with group_size=128, asymmetric     │
│                                                                   │
│  Needs a KERNEL that can:                                         │
│  1. Load packed INT4 weights (2 per byte)                         │
│  2. Load FP16 scales (one per group of 128 weights)               │
│  3. Load FP16 zero-points (one per group)                         │
│  4. Dequantize: w_fp16 = (w_int4 - zero) * scale                  │
│  5. Compute: output = input @ w_fp16.T                            │
│  6. All on Tensor Cores, maximizing throughput                    │
│                                                                   │
│  Available kernels for W4A16:                                     │
│  ├── Marlin kernel (vLLM) ─── very fast, NVIDIA GPU only          │
│  ├── Machete kernel (NVIDIA) ─ from CUTLASS, flexible             │
│  ├── ExLlama v2 kernel ────── popular, GPTQ-optimized             │
│  ├── AutoAWQ kernel ─────────  AWQ-specific, reference impl       │
│  └── CUTLASS mixed-precision ─ generic but configurable           │
│                                                                   │
│  If your format DOESN'T match any existing kernel:                │
│  → You need to WRITE or MODIFY a kernel                           │
│  → Or your quantized model will be SLOW (Python dequant+FP16 mm)  │
│                                                                   │
└────────────────────────────────────────────────────────────────────┘
```

### Phase 4: Inference Engine Integration

```python
# Step 4a: Run in vLLM (if your format is supported)
from vllm import LLM

llm = LLM(
    model="./my_awq_model",
    quantization="awq",           # Must match a registered quant config
    dtype="float16",
    tensor_parallel_size=4,
    max_model_len=8192,
    gpu_memory_utilization=0.9,
)

# If your format is NOT supported, you need to:
# 1. Register a new QuantizationConfig in vllm/model_executor/layers/quantization/
# 2. Implement the quantized Linear layer
# 3. Map to existing or new CUDA kernels
```

```python
# Step 4b: Run in SGLang
import sglang as sgl

runtime = sgl.Runtime(
    model_path="./my_awq_model",
    quantization="awq",
    tp_size=4,
)
```

```bash
# Step 4c: Run in llama.cpp (GGUF format required)
./llama-server \
    -m model-q4_k_m.gguf \
    --port 8080 \
    --ctx-size 4096 \
    --n-gpu-layers 35
```

### Phase 5: Serving at Scale

```python
# Step 5: Deploy behind Triton Inference Server
# triton_config.pbtxt
"""
name: "llama-awq"
backend: "vllm"
max_batch_size: 256
input [
  { name: "text_input" data_type: TYPE_STRING dims: [ -1 ] }
]
output [
  { name: "text_output" data_type: TYPE_STRING dims: [ -1 ] }
]
instance_group [
  { count: 1 kind: KIND_MODEL }
]
"""
```

> **Working code examples**: For complete, copy-paste-ready versions of each deployment path (AWQ→vLLM, GPTQ→SGLang, FP8→TRT-LLM, GGUF→llama.cpp, ONNX→ONNX Runtime), see [12 — Practical Code Guide](./12_practical_code_guide.md).

## 📊 Quantization Format Compatibility Matrix

| Your Format | SafeTensors | GGUF | ONNX | TRT Engine | ExecuTorch |
|-------------|-------------|------|------|------------|------------|
| W4A16 (AWQ, group=128) | ✅ | ⚠️ re-quant | ⚠️ custom op | ✅ | ⚠️ limited |
| W4A16 (GPTQ, group=128) | ✅ | ⚠️ re-quant | ⚠️ custom op | ✅ | ⚠️ limited |
| W8A8 (SmoothQuant) | ✅ | ⚠️ Q8_0 only | ✅ QDQ nodes | ✅ | ✅ |
| W8A8 (FP8 E4M3) | ✅ | ✅ F8_E4M3 | ✅ | ✅ native | ❌ |
| W4A4 (FP4 Blackwell) | ✅ | ✅ MXFP4 | ❌ | ✅ native | ❌ |
| Mixed-precision custom | ✅ (manual) | ❌ | ❌ custom | ❌ | ❌ |
| 2-bit / 1.5-bit extreme | ❌ | ✅ IQ2_XS | ❌ | ❌ | ❌ |

## 🔧 How to Get Your Research Deployed: Decision Tree

```
┌─────────────────────────────────────────────────┐
│  I developed a new quantization algorithm.      │
│  How do I get it deployed?                      │
└────────────────────┬────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          │ Does it match an    │
          │ existing format?     │
          └────────┬─────┬──────┘
                   │     │
              Yes  │     │  No
                   │     │
                   ▼     ▼
         ┌──────────┐ ┌──────────────────────────┐
         │ Lucky!   │ │ You need to:              │
         │ Just save│ │ 1. Define weight packing  │
         │ in that  │ │ 2. Write a CUDA kernel    │
         │ format & │ │    (or extend existing)   │
         │ deploy   │ │ 3. Register in engine     │
         └──────────┘ │ 4. Handle serialization   │
                      │                            │
                      │ Effort: 2-6 months         │
                      │                            │
                      │ Shortcut options:           │
                      │ • Contribute to vLLM        │
                      │ • Use Marlin template       │
                      │ • torch.compile custom ops │
                      │ • CUTLASS template          │
                      └────────────────────────────┘
```

## 🏗️ Adding a New Quantization Format to vLLM (Sketch)

```python
# 1. Create quantization config
# vllm/model_executor/layers/quantization/my_quant.py

from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

class MyQuantConfig(QuantizationConfig):
    def __init__(self, bits, group_size, ...):
        self.bits = bits
        self.group_size = group_size

    @classmethod
    def get_name(cls) -> str:
        return "my_quant"

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.float16, torch.bfloat16]

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase):
            return MyQuantLinearMethod(self)
        return None

# 2. Implement the linear method
class MyQuantLinearMethod:
    def create_weights(self, ...):
        # Define packed weight tensors
        ...

    def apply(self, layer, x, bias=None):
        # Call your custom kernel
        return my_quant_cuda.matmul(x, layer.qweight, layer.scales, ...)

# 3. Write the CUDA kernel (or use existing)
# This is the hard part — see CUTLASS examples
```

## 📈 Which Deployment Path Is Most Common?

> **Note**: These percentages are estimated based on GitHub stars, HuggingFace model downloads, public deployment reports, and community surveys. They are approximate and shift rapidly.

```
Popularity of deployment paths (March 2026):

vLLM + SafeTensors + NVIDIA GPU:    ████████████████████████████  45%
SGLang + SafeTensors + NVIDIA GPU:  ██████████████                22%
llama.cpp + GGUF + CPU/Apple:       ████████████                  18%
TensorRT-LLM + NVIDIA GPU:         ██████                         8%
ONNX RT + various hardware:        ███                            4%
Other (MLC, OpenVINO, etc.):        ██                             3%
```

---

## 🧱 Weight Packing: How INT4 Weights Are Physically Stored

Understanding weight packing is essential when writing or debugging quantized kernels.

```
┌──────── INT4 Weight Packing Schemes ──────────────────────┐
│                                                           │
│  Problem: INT4 is NOT a native hardware data type.        │
│  You can't store a single 4-bit value — the minimum       │
│  addressable unit is 8 bits (1 byte).                     │
│                                                           │
│  PACKING: Store 2 INT4 values per byte (or 8 per INT32)   │
│                                                           │
│  ┌─── 1 byte (uint8) ───┐                                 │
│  │ high nibble│low nibble│                                │
│  │   w1 (4b)  │  w0 (4b) │                                │
│  └────────────┴──────────┘                                │
│                                                           │
│  ┌─── 1 INT32 ──────────────────────────────────────┐     │
│  │ w7 │ w6 │ w5 │ w4 │ w3 │ w2 │ w1 │ w0 │          │     │
│  │ 4b │ 4b │ 4b │ 4b │ 4b │ 4b │ 4b │ 4b │          │     │
│  └──────────────────────────────────────────────┘      │  │
│  Unpack: w0 = (int32_val >> 0) & 0xF                   │  │
│          w1 = (int32_val >> 4) & 0xF                   │  │
│          ...                                            │ │
│                                                           │
│  DIFFERENT FORMATS PACK DIFFERENTLY:                      │
│                                                           │
│  AWQ:   Pack along output channel dimension               │
│         qweight shape: [in_features, out_features/8]      │
│         (8 INT4 weights per INT32, column-major)          │
│                                                           │
│  GPTQ:  Pack along input channel dimension                │
│         qweight shape: [in_features/8, out_features]      │
│         (8 INT4 weights per INT32, row-major)             │
│         → Different kernel needed vs AWQ!                 │
│                                                           │
│  Marlin: Reorders weights for Tensor Core alignment       │
│         Special interleaved layout matching warp access   │
│         → Fastest kernel, but requires format conversion  │
│                                                           │
│  KEY INSIGHT: AWQ and GPTQ weights are NOT interchangeable│
│  even though both are "INT4 with group_size=128."         │
│  The PACKING matters for kernel compatibility!            │
└─────────────────────────────────────────────────────────────┘
```

## 🔬 PTQ vs QAT: When to Use Which

```
┌──────── Post-Training Quantization (PTQ) ──────────────────┐
│                                                            │
│  What: Quantize a PRETRAINED model without further training│
│  Examples: AWQ, GPTQ, SmoothQuant, FP8 calibration         │
│  Cost: Minutes to hours (calibration on 128-512 samples)   │
│  Quality: Good for ≥4-bit, degrades at 2-3 bit             │
│                                                            │
│  When to use:                                              │
│  ✅ You don't have training infrastructure                  │
│  ✅ The model is large (70B+) and training is expensive     │
│  ✅ INT4 or INT8 precision is sufficient                    │
│  ✅ You want fast iteration on quantization algorithms      │
│                                                            │
│  This is what YOU (the researcher) mostly work on.         │
└──────────────────────────────────────────────────────────────┘

┌──────── Quantization-Aware Training (QAT) ─────────────────┐
│                                                            │
│  What: Simulate quantization DURING training/fine-tuning   │
│  Examples: Google AQT (JAX), PyTorch QAT, LLM-QAT          │
│  Cost: Full training run (days/weeks, 100s of GPU-hours)   │
│  Quality: Better than PTQ, especially at ≤3 bits           │
│                                                            │
│  When to use:                                              │
│  ✅ Ultra-low-bit quantization (2-3 bit)                    │
│  ✅ You have training budget and data access                │
│  ✅ Quality at INT4 PTQ isn't sufficient for your use case  │
│  ✅ Deploying a high-value production model worth training  │
│                                                            │
│  Less common in research because of cost.                  │
└──────────────────────────────────────────────────────────────┘

┌──────── Comparison ────────────────────────────────────────┐
│                                                            │
│  Metric          │ PTQ              │ QAT                  │
│  ────────────────│──────────────────│───────────────────────│
│  Cost            │ Low (hours)      │ High (days/weeks)    │
│  Quality at W8A8 │ Excellent        │ Excellent            │
│  Quality at W4A16│ Good             │ Very good            │
│  Quality at W2   │ Poor             │ Acceptable           │
│  Calibration data│ 128-512 samples  │ Full training dataset│
│  Ease of use     │ Easy             │ Requires training    │
│  Industry use    │ ~90% of deploys  │ ~10% (high-value)    │
└──────────────────────────────────────────────────────────────┘
```

## 📐 Calibration Data: What Makes Good Calibration

```
┌──────── Calibration for PTQ ──────────────────────────────┐
│                                                           │
│  Purpose: Run representative inputs through the model to  │
│  collect activation statistics → compute scales/zero-pts  │
│                                                           │
│  WHAT MAKES GOOD CALIBRATION DATA:                        │
│  ✅ Representative of real-world inputs                    │
│  ✅ Diverse (covers different topics/domains)              │
│  ✅ Reasonable length (not too short, not too long)        │
│  ✅ 128-512 samples is usually sufficient for LLMs         │
│  ❌ NOT the same as training data (can be any text)        │
│  ❌ NOT labeled (no ground truth needed)                   │
│                                                           │
│  COMMON CALIBRATION DATASETS:                             │
│  • pileval (used by AutoAWQ — excerpt from The Pile)      │
│  • wikitext-2 (Wikipedia articles)                        │
│  • c4 (Common Crawl subset — used by SmoothQuant)         │
│  • Custom domain data (best for domain-specific models)   │
│                                                           │
│  HOW MANY SAMPLES?                                        │
│  • AWQ: 128 samples typical                               │
│  • GPTQ: 128-256 samples                                  │
│  • SmoothQuant: 512 samples typical                       │
│  • FP8 calibration: 100-500 samples                       │
│  • imatrix (llama.cpp): 100+ chunks recommended           │
│                                                           │
│  DOES CALIBRATION DATA MATTER?                            │
│  • For INT8/FP8: Minimal impact (robust)                  │
│  • For INT4: Moderate impact (wrong domain can hurt)      │
│  • For INT2-3: Significant impact (choose carefully)      │
│                                                           │
│  GOTCHA: Calibration data that's too short (< 64 tokens)  │
│  can produce poor scale estimates. Use full-length samples. │
└─────────────────────────────────────────────────────────────┘
```

## ⚡ Why Activation Quantization Is Hard

```
┌──────── The Activation Quantization Challenge ────────────┐
│                                                           │
│  Weight quantization (W4, W8) is relatively easy:         │
│  • Weights are STATIC — known at quantization time        │
│  • Distribution is approximately Gaussian                 │
│  • Per-channel or per-group scales work well              │
│                                                           │
│  Activation quantization (A8, A4) is HARD:                │
│  • Activations are DYNAMIC — change with every input      │
│  • Outlier channels: some channels have 10-100× larger    │
│    values than others                                     │
│                                                           │
│    ┌─── Activation distribution example ───────────────┐  │
│    │                                                     ││
│    │  Channel 0:  [-1.2, 0.5, -0.8, 0.3, ...]  ← normal│  │
│    │  Channel 1:  [-0.9, 0.7, -0.4, 0.6, ...]  ← normal│  │
│    │  Channel 42: [-68.5, 52.3, -71.2, ...]     ← OUTLIER││
│    │                                                     ││
│    │  If you use per-tensor scale for INT8:              ││
│    │  scale = max(|all values|) / 127 = 71.2 / 127      │ │
│    │  Normal channels get ~2-bit effective precision!    ││
│    └─────────────────────────────────────────────────────┘│
│                                                           │
│  SOLUTIONS:                                               │
│  • SmoothQuant: Migrate difficulty from activations to    │
│    weights (multiply act by s, divide weight by s)        │
│  • Per-token quantization: Different scale per token      │
│  • Per-channel activation scales: More accurate           │
│  • Dynamic quantization: Compute scales at runtime        │
│    (slower but adapts to each input)                      │
│  • FP8 (E4M3): Wider dynamic range than INT8 — handles    │
│    outliers better without smoothing                      │
│                                                           │
│  This is why:                                             │
│  • W4A16 is common (weight quant easy, skip act quant)    │
│  • W8A8 is common (INT8 act is manageable with smoothing) │
│  • W4A4 is rare (4-bit act is extremely challenging)      │
│  • FP8 is gaining popularity (handles outliers natively)  │
└─────────────────────────────────────────────────────────────┘
```

## 🏗️ Model Architecture Impact on Quantization

```
┌──────── How Architecture Affects Quantization ────────────┐
│                                                           │
│  MODEL SIZE:                                              │
│  • Larger models (70B+) are MORE resilient to quantization│
│    (redundancy absorbs precision loss)                    │
│  • Small models (1-3B) are MORE sensitive                 │
│    (every parameter matters more)                         │
│  • Rule of thumb: 70B @ Q4 ≈ 7B @ Q8 in quality           │
│                                                           │
│  DENSE vs MoE:                                            │
│  • MoE models (Mixtral, DeepSeek) may be more resilient   │
│    — only a subset of experts active per token            │
│  • But MoE has MORE total parameters → higher memory need │
│  • Expert-level quantization: different bits per expert   │
│                                                           │
│  SENSITIVE LAYERS (keep at higher precision):             │
│  • Embedding layer (first layer, maps tokens to vectors)  │
│  • LM head (final layer, maps back to vocabulary)         │
│  • First and last transformer blocks                      │
│  • Attention layers (more sensitive than MLP typically)   │
│  → Mixed-precision: sensitive layers at FP16, rest at INT4│
│                                                           │
│  VISION-LANGUAGE MODELS (VLMs):                           │
│  • Vision encoder and language model may need different   │
│    quantization strategies                                │
│  • Cross-attention layers are often more sensitive        │
│  • Vision encoder may tolerate INT8 but not INT4          │
│                                                           │
│  GQA vs MHA:                                              │
│  • GQA models (Llama 3) have smaller KV-cache             │
│  • Less pressure for KV-cache quantization                │
│  • But KV-cache quant (FP8) still helps batch capacity    │
└─────────────────────────────────────────────────────────────┘
```

## 💰 Cost Analysis: The Economics of Quantization

```
┌──────── Quantization Cost Savings ────────────────────────┐
│                                                           │
│  Example: Serving Llama 3 70B (March 2026 cloud pricing)  │
│                                                           │
│  ┌────────────┬──────────┬────────────┬────────────────┐  │
│  │ Precision  │ GPUs     │ $/hr (est.)│ Throughput     │  │
│  ├────────────┼──────────┼────────────┼────────────────┤  │
│  │ FP16       │ 4× H100  │ ~$16/hr    │ ~2000 tok/s   │   │
│  │ FP8        │ 2× H100  │ ~$8/hr     │ ~2500 tok/s   │   │
│  │ AWQ INT4   │ 1× H100  │ ~$4/hr     │ ~3000 tok/s   │   │
│  │ AWQ INT4   │ 1× L40S  │ ~$1.5/hr   │ ~1500 tok/s   │   │
│  │ GGUF Q4_K_M│ 1× L40S  │ ~$1.5/hr   │ ~1200 tok/s   │   │
│  └────────────┴──────────┴────────────┴────────────────┘  │
│                                                           │
│  KEY TAKEAWAYS:                                           │
│  • INT4 quantization can give 4-10× cost reduction        │
│  • Same model on cheaper GPU often beats expensive GPU    │
│    (AWQ on L40S < FP16 on H100)                           │
│  • Cost per million tokens:                               │
│    FP16: ~$0.30/M tokens                                  │
│    INT4: ~$0.05/M tokens                                  │
│  • Break-even: Quantization time (minutes) pays for       │
│    itself within the first few hours of serving           │
│                                                           │
│  Note: Prices are approximate and vary by cloud provider. │
│  Always benchmark YOUR specific model and workload.       │
└─────────────────────────────────────────────────────────────┘
```

---

**Next**: [07 — Company Landscape →](./07_company_landscape.md)
