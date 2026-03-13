[← Back to Table of Contents](./README.md)

*Last updated: March 2026*

# 12 — Practical Code Guide

> End-to-end code examples: from PyTorch model → quantize → export → deploy → serve.

## 🗺️ Complete Pipelines

This guide provides working code for the most common deployment scenarios. Each pipeline is self-contained.

---

## Pipeline 1: AWQ Quantization → vLLM Serving

**Scenario**: You have a fine-tuned Llama model, want to serve it with best throughput on NVIDIA GPUs.

### Step 1: Quantize with AutoAWQ

```python
# pip install autoawq transformers

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "meta-llama/Llama-3.1-8B-Instruct"
quant_path = "./llama3-8b-awq"

# Load model (FP16)
model = AutoAWQForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="float16",
    safetensors=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Configure quantization
quant_config = {
    "zero_point": True,       # Asymmetric quantization
    "q_group_size": 128,      # Group size for per-group quantization
    "w_bit": 4,               # 4-bit weights
    "version": "GEMM",        # GEMM kernel (vs GEMV for batch=1)
}

# Quantize! (needs calibration data internally)
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data="pileval",     # Built-in calibration dataset
    # Or provide your own: calib_data=["text1", "text2", ...]
)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

# Output structure:
# llama3-8b-awq/
# ├── config.json
# ├── quantize_config.json     ← AWQ parameters
# ├── model.safetensors        ← INT4 quantized weights
# ├── tokenizer.json
# └── tokenizer_config.json
```

### Step 2: Verify Quantized Model

```python
# Quick sanity check
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model = AutoAWQForCausalLM.from_quantized(
    "./llama3-8b-awq",
    fuse_layers=True,         # Fuse QKV + MLP layers
)
tokenizer = AutoTokenizer.from_pretrained("./llama3-8b-awq")

# Test generation
inputs = tokenizer("The capital of France is", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

### Step 3: Serve with vLLM

```bash
# pip install vllm

# Option A: Command line (simplest)
vllm serve ./llama3-8b-awq \
  --quantization awq \
  --max-model-len 8192 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --port 8000

# This starts an OpenAI-compatible server at http://localhost:8000
```

```python
# Option B: Python API
from vllm import LLM, SamplingParams

llm = LLM(
    model="./llama3-8b-awq",
    quantization="awq",
    max_model_len=8192,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    enforce_eager=False,       # Use CUDA graphs
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256,
)

# Batch inference
prompts = [
    "Explain quantum computing in simple terms:",
    "Write a Python function to sort a list:",
    "What is the meaning of life?",
]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt[:50]}...")
    print(f"Output: {output.outputs[0].text[:100]}...")
    print("---")
```

### Step 4: Call the Server

```python
# Client code (works with vLLM server)
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="./llama3-8b-awq",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain AWQ quantization in 3 sentences."},
    ],
    temperature=0.7,
    max_tokens=256,
    stream=True,  # Streaming!
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## Pipeline 2: GPTQ Quantization → SGLang Serving

### Step 1: Quantize with AutoGPTQ

```python
# pip install auto-gptq optimum transformers

from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

model_path = "meta-llama/Llama-3.1-8B-Instruct"
quant_path = "./llama3-8b-gptq"

tokenizer = AutoTokenizer.from_pretrained(model_path)

# Prepare calibration data
import random
from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
calibration_texts = [text for text in dataset["text"] if len(text) > 100]
random.shuffle(calibration_texts)
calibration_data = [
    tokenizer(text, return_tensors="pt", max_length=2048, truncation=True)
    for text in calibration_texts[:128]
]

# Configure GPTQ
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=True,     # Use activation order (better quality, slower quant)
    damp_percent=0.1,
    sym=True,          # Symmetric quantization
)

# Load and quantize
model = AutoGPTQForCausalLM.from_pretrained(
    model_path,
    quantize_config=quantize_config,
    torch_dtype="float16",
)
model.quantize(calibration_data)

# Save
model.save_quantized(quant_path, use_safetensors=True)
tokenizer.save_pretrained(quant_path)
```

### Step 2: Serve with SGLang

```bash
# pip install sglang[all]

python -m sglang.launch_server \
  --model-path ./llama3-8b-gptq \
  --quantization gptq \
  --port 8001 \
  --tp 1 \
  --mem-fraction-static 0.85
```

```python
# SGLang structured generation (unique feature!)
import sglang as sgl

@sgl.function
def extract_info(s, text):
    s += sgl.system("Extract structured information from the text.")
    s += sgl.user(f"Text: {text}")
    s += sgl.assistant(
        sgl.gen("result", max_tokens=200, 
                regex=r'\{"name": "[^"]+", "age": \d+, "city": "[^"]+"\}')
    )
    # ↑ Constrained generation with regex — SGLang excels at this!

# Run
sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:8001"))
result = extract_info.run(text="John Smith is 30 years old and lives in NYC.")
print(result["result"])  # Guaranteed valid JSON matching the regex!
```

---

## Pipeline 3: FP8 Quantization → TensorRT-LLM

### Step 1: Quantize to FP8 with NVIDIA ModelOpt

```python
# pip install nvidia-modelopt

import modelopt.torch.quantization as mtq
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "meta-llama/Llama-3.1-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="float16",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Calibration function
def calibrate(model):
    calib_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In machine learning, quantization reduces model precision.",
        # ... more calibration sentences
    ]
    for text in calib_texts:
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        model(**inputs)

# Quantize to FP8
mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop=calibrate)

# Export for TensorRT-LLM
from modelopt.torch.export import export_tensorrt_llm_checkpoint
export_tensorrt_llm_checkpoint(
    model,
    decoder_type="llama",
    dtype="float16",
    export_dir="./llama3-8b-fp8-ckpt",
    inference_tensor_parallel=2,  # 2-GPU TP
)
```

### Step 2: Build TensorRT-LLM Engine

```bash
# Build engine from exported checkpoint
trtllm-build \
  --checkpoint_dir ./llama3-8b-fp8-ckpt \
  --output_dir ./llama3-8b-fp8-engine \
  --gemm_plugin float16 \
  --max_batch_size 64 \
  --max_input_len 4096 \
  --max_seq_len 8192 \
  --workers 2  # For 2-GPU TP
```

### Step 3: Run TRT-LLM

```python
import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner

runner = ModelRunner.from_dir(
    engine_dir="./llama3-8b-fp8-engine",
    rank=tensorrt_llm.mpi_rank(),
)

# Generate
outputs = runner.generate(
    batch_input_ids=[tokenizer.encode("Hello, world!")],
    max_new_tokens=100,
    end_id=tokenizer.eos_token_id,
    pad_id=tokenizer.pad_token_id,
    temperature=0.7,
    top_p=0.9,
)

text = tokenizer.decode(outputs[0][0])
print(text)
```

---

## Pipeline 4: GGUF → llama.cpp (Local Deployment)

### Step 1: Convert & Quantize to GGUF

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make -j

# Option A: Convert HuggingFace model to GGUF
python convert_hf_to_gguf.py \
  ~/models/Llama-3.1-8B-Instruct/ \
  --outfile llama3-8b-f16.gguf \
  --outtype f16

# Option B: Quantize to various levels
./llama-quantize llama3-8b-f16.gguf llama3-8b-q4_k_m.gguf Q4_K_M

# Option C: With importance matrix (better quality at low bits)
# First, generate importance matrix from calibration data
./llama-imatrix \
  -m llama3-8b-f16.gguf \
  -f wiki.train.raw \
  -o imatrix.dat \
  --chunks 100

# Then quantize with importance matrix
./llama-quantize \
  --imatrix imatrix.dat \
  llama3-8b-f16.gguf \
  llama3-8b-iq4_xs.gguf \
  IQ4_XS
```

### Step 2: Run Inference

```bash
# Interactive chat
./llama-cli \
  -m llama3-8b-q4_k_m.gguf \
  --chat-template llama3 \
  -c 8192 \         # Context length
  -ngl 99 \         # Offload all layers to GPU
  -t 8 \            # CPU threads (for CPU layers)
  --temp 0.7 \
  --top-p 0.9

# Server mode (OpenAI-compatible)
./llama-server \
  -m llama3-8b-q4_k_m.gguf \
  --chat-template llama3 \
  -c 8192 \
  -ngl 99 \
  --port 8080 \
  --host 0.0.0.0
```

```python
# Python binding
# pip install llama-cpp-python
from llama_cpp import Llama

llm = Llama(
    model_path="llama3-8b-q4_k_m.gguf",
    n_ctx=8192,
    n_gpu_layers=-1,  # All layers on GPU
    verbose=False,
)

output = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is quantization?"},
    ],
    max_tokens=256,
    temperature=0.7,
)
print(output["choices"][0]["message"]["content"])
```

---

## Pipeline 5: ONNX Export → ONNX Runtime (Cross-Platform)

### Step 1: Export to ONNX

```python
# pip install optimum[onnxruntime-gpu] onnxruntime-gpu

from optimum.onnxruntime import ORTModelForCausalLM
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer
from transformers import AutoTokenizer

model_id = "meta-llama/Llama-3.1-8B-Instruct"
save_dir = "./llama3-8b-onnx"

# Export to ONNX
ort_model = ORTModelForCausalLM.from_pretrained(
    model_id,
    export=True,
    provider="CUDAExecutionProvider",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

ort_model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
```

### Step 2: Quantize ONNX Model

```python
# INT4 quantization with ONNX Runtime
from onnxruntime.quantization import matmul_4bits_quantizer

model_path = "./llama3-8b-onnx/model.onnx"
output_path = "./llama3-8b-onnx-int4/model.onnx"

quantizer = matmul_4bits_quantizer.MatMul4BitsQuantizer(
    model_path,
    block_size=128,
    is_symmetric=True,
    accuracy_level=4,  # 0-4, higher = better quality
)
quantizer.process()
quantizer.model.save_model_to_file(output_path)
```

### Step 3: Run with ONNX Runtime

```python
import onnxruntime as ort
import numpy as np

# Create session with GPU
session = ort.InferenceSession(
    "./llama3-8b-onnx-int4/model.onnx",
    providers=[
        ("CUDAExecutionProvider", {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
        }),
        "CPUExecutionProvider",  # Fallback
    ],
)

# Or use Optimum for easier inference
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer, pipeline

model = ORTModelForCausalLM.from_pretrained(
    "./llama3-8b-onnx-int4",
    provider="CUDAExecutionProvider",
)
tokenizer = AutoTokenizer.from_pretrained("./llama3-8b-onnx-int4")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
result = pipe("The future of AI is", max_new_tokens=100)
print(result[0]["generated_text"])
```

---

## Pipeline 6: Docker Deployment (Production)

### Dockerfile for vLLM

```dockerfile
# Dockerfile.vllm
FROM vllm/vllm-openai:latest

# Copy your quantized model (or download at runtime)
# COPY ./llama3-8b-awq /models/llama3-8b-awq

# Set default arguments
ENV MODEL_NAME="TheBloke/Llama-2-7B-AWQ"
ENV QUANTIZATION="awq"
ENV MAX_MODEL_LEN="8192"
ENV TENSOR_PARALLEL_SIZE="1"

EXPOSE 8000

# NOTE: Using shell form (not JSON array) so that environment variables
# like ${MODEL_NAME} are expanded at runtime by the shell.
CMD python -m vllm.entrypoints.openai.api_server \
     --model ${MODEL_NAME} \
     --quantization ${QUANTIZATION} \
     --max-model-len ${MAX_MODEL_LEN} \
     --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
     --port 8000
```

### Docker Compose for Multi-Service

```yaml
# docker-compose.yml
version: "3.8"

services:
  vllm-server:
    image: vllm/vllm-openai:latest
    ports:
      - "8000:8000"
    volumes:
      - ./models:/models
      - huggingface-cache:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: >
      python -m vllm.entrypoints.openai.api_server
      --model /models/llama3-8b-awq
      --quantization awq
      --max-model-len 8192
      --port 8000
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:latest
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      vllm-server:
        condition: service_healthy

volumes:
  huggingface-cache:
```

---

## 🧪 Benchmarking & Evaluation

### Benchmark Quantized vs Original

```python
# pip install lm-eval

# Evaluate original model
# lm_eval --model hf \
#   --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
#   --tasks mmlu,hellaswag,winogrande \
#   --batch_size 8

# Evaluate AWQ quantized model
# lm_eval --model hf \
#   --model_args pretrained=./llama3-8b-awq \
#   --tasks mmlu,hellaswag,winogrande \
#   --batch_size 8

# Evaluate via vLLM server (running model)
# lm_eval --model local-completions \
#   --model_args model=llama3-8b-awq,base_url=http://localhost:8000/v1 \
#   --tasks mmlu \
#   --batch_size auto
```

### Measure Inference Speed

```python
import time
import torch
from vllm import LLM, SamplingParams

def benchmark_vllm(model_path, quantization=None, num_prompts=100):
    llm = LLM(
        model=model_path,
        quantization=quantization,
        max_model_len=2048,
    )
    
    prompts = ["Write a short story about AI: "] * num_prompts
    sampling_params = SamplingParams(max_tokens=256, temperature=0.7)
    
    # Warmup
    llm.generate(prompts[:5], sampling_params)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    
    print(f"Model: {model_path}")
    print(f"Quantization: {quantization or 'None'}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Throughput: {total_tokens / elapsed:.1f} tokens/sec")
    print(f"Avg latency: {elapsed / num_prompts * 1000:.1f} ms/request")
    
    return total_tokens / elapsed

# Compare
fp16_tps = benchmark_vllm("meta-llama/Llama-3.1-8B-Instruct")
awq_tps = benchmark_vllm("./llama3-8b-awq", quantization="awq")
print(f"\nSpeedup: {awq_tps / fp16_tps:.2f}x")
```

### Measure Perplexity

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def measure_perplexity(model, tokenizer, dataset_name="wikitext", 
                       dataset_config="wikitext-2-raw-v1", max_samples=100):
    dataset = load_dataset(dataset_name, dataset_config, split="test")
    
    nlls = []
    for i, sample in enumerate(dataset):
        if i >= max_samples:
            break
        if len(sample["text"].strip()) == 0:
            continue
            
        inputs = tokenizer(
            sample["text"], 
            return_tensors="pt", 
            max_length=2048, 
            truncation=True
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            nlls.append(outputs.loss.item())
    
    avg_nll = sum(nlls) / len(nlls)
    perplexity = torch.exp(torch.tensor(avg_nll)).item()
    
    print(f"Perplexity: {perplexity:.2f}")
    return perplexity
```

---

## 📋 Quick Reference Card

```
┌──────────────────────── DEPLOYMENT CHEAT SHEET ──────────────────────┐
│                                                                      │
│  GOAL                    RECIPE                                      │
│  ────                    ──────                                      │
│  Max throughput (NVIDIA)  FP8 + TRT-LLM + Triton Server              │
│  Good throughput + easy   AWQ INT4 + vLLM                            │
│  Best quality/throughput  SmoothQuant W8A8 + vLLM                    │
│  Multi-GPU (80B+ model)  AWQ/FP8 + vLLM --tp 4                       │
│  Local Mac/PC            GGUF Q4_K_M + llama.cpp (Ollama)            │
│  Cross-platform          ONNX INT4 + ONNX Runtime                    │
│  Mobile phone            ExecuTorch INT4/INT8 or Core ML             │
│  Web browser             MLC-LLM (WebGPU) or ONNX RT Web             │
│  Embedded/MCU            TFLite INT8 + TFLM (not for LLMs)           │
│  AMD GPU                 AWQ INT4 + vLLM --device rocm               │
│  Structured output       GPTQ/AWQ + SGLang (compressed FSM)          │
│  Multi-model serving     Any quant + Triton Inference Server         │
│  Quick experiment        HF Transformers + BitsAndBytes (NF4)        │
│                                                                      │
│  QUANTIZE               COMMAND                                      │
│  ────────               ───────                                      │
│  AWQ                    autoawq: AutoAWQForCausalLM.quantize()       │
│  GPTQ                   auto-gptq: AutoGPTQForCausalLM.quantize()    │
│  FP8                    nvidia-modelopt: mtq.quantize(FP8_DEFAULT)   │
│  GGUF                   llama.cpp: llama-quantize model.gguf Q4_K_M  │
│  BitsAndBytes           transformers: load_in_4bit=True              │
│  SmoothQuant            smoothquant or vLLM built-in                 │
│                                                                      │
│  UPLOAD TO HF            COMMAND                                     │
│  ────────────            ───────                                     │
│  huggingface-cli upload username/model-awq ./llama3-8b-awq           │
│                                                                      │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 🎓 Summary: Your Research → Production Path

```
┌──────────── THE COMPLETE PICTURE ──────────────────────────┐
│                                                            │
│  1. RESEARCH (You)                                         │
│     ├── Develop quantization algorithm (Python/PyTorch)    │
│     ├── Evaluate on benchmarks (perplexity, accuracy)      │
│     └── Publish paper                                      │
│                                                            │
│  2. IMPLEMENTATION                                         │
│     ├── Write quantization script (autoawq-style)          │
│     ├── Create HuggingFace-compatible output               │
│     │   (SafeTensors + quantize_config.json)               │
│     └── Optional: Write custom Triton/CUDA kernel          │
│                                                            │
│  3. INTEGRATION                                            │
│     ├── PR to vLLM/SGLang to support your format           │
│     ├── PR to llama.cpp for GGUF support                   │
│     ├── PR to ONNX RT for new quantized op                 │
│     └── PR to HuggingFace Transformers                     │
│                                                            │
│  4. DEPLOYMENT (Users of your research)                    │
│     ├── quantize with your tool                            │
│     ├── upload to HuggingFace Hub                          │
│     ├── serve with vLLM/SGLang/TRT-LLM                     │
│     └── or run locally with llama.cpp/Ollama               │
│                                                            │
│  5. IMPACT MEASUREMENT                                     │
│     ├── Downloads on HuggingFace                           │
│     ├── Integration in major engines                       │
│     ├── Community adoption (GGUF conversions, etc.)        │
│     └── Industry deployment                                │
│                                                            │
│  This is the path AWQ and GPTQ followed.                   │
│  Your next algorithm can follow the same path!             │
└──────────────────────────────────────────────────────────────┘
```

---

## 🎚️ Practical Mixed-Precision Notes

When deploying quantized models, not all layers should be quantized equally. Here are practical tips:

```python
# Example: Mixed-precision with AutoAWQ
# Keep sensitive layers at higher precision

quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",
    # Some tools support per-layer config:
    # "modules_to_not_convert": ["lm_head", "embed_tokens"]
}

# In llama.cpp, imatrix automatically assigns more bits to sensitive layers
# Example: Q4_K_M uses mix of Q4_K and Q6_K based on layer importance
#   - Attention layers may get Q6_K (more bits)
#   - MLP layers may get Q4_K (fewer bits)
#   - This is why K_M variants outperform pure Q4_0

# For GPTQ, you can specify per-layer bit widths using some forks:
# layer_config = {
#     "model.embed_tokens": {"bits": 8},   # Keep embeddings at INT8
#     "model.layers.*.self_attn.*": {"bits": 4},  # Attention at INT4
#     "model.layers.*.mlp.*": {"bits": 4},         # MLP at INT4
#     "lm_head": {"bits": 8},              # Keep LM head at INT8
# }
```

**Key Rules of Thumb**:
- Always keep `embed_tokens` and `lm_head` at higher precision (FP16 or INT8)
- First and last transformer blocks are more sensitive — consider INT8 instead of INT4
- Attention layers are generally more sensitive than MLP layers
- The `Q4_K_M` in GGUF achieves this automatically via importance matrix

> 📎 *See [Doc 06 — Quantization to Deployment](./06_quantization_to_deployment.md) for the theory behind model architecture impact on quantization.*

---

**← Back to**: [README — Table of Contents](./README.md)
