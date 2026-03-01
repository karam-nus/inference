[← Back to Table of Contents](./README.md)

*Last updated: March 2026*

# 09 — Serving & Middleware

> After inference engines: how models are actually served to users at scale.

## 🎯 Serving vs Inference — The Critical Distinction

```
┌───────────────────────────────────────────────────────────┐
│                                                           │
│  INFERENCE = Running the model on one input               │
│  (forward pass, token generation)                         │
│                                                           │
│  SERVING = Managing inference at scale                    │
│  (routing, batching, scaling, monitoring, load balancing) │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │                    SERVING LAYER                     │ │
│  │                                                     │ │
│  │  API Gateway → Load Balancer → Model Router         │ │
│  │       │              │             │                │ │
│  │       ▼              ▼             ▼                │ │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐           │ │
│  │  │ Engine  │  │ Engine  │  │ Engine  │            │ │
│  │  │ Replica │  │ Replica │  │ Replica │            │ │
│  │  │  (GPU0) │  │  (GPU1) │  │  (GPU2) │            │ │
│  │  │         │  │         │  │         │            │ │
│  │  │INFERENCE│  │INFERENCE│  │INFERENCE│            │ │
│  │  └─────────┘  └─────────┘  └─────────┘           │ │
│  └─────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────┘
```

## 📊 Serving Stack Layers

```
┌───────────────────── FULL SERVING STACK ─────────────────────┐
│                                                               │
│  Layer 7: APPLICATION                                        │
│  │  ChatGPT UI, API clients, Slack bots, RAG pipelines       │
│  │                                                            │
│  Layer 6: API GATEWAY / PROXY                                │
│  │  Kong, NGINX, Envoy, AWS API Gateway                      │
│  │  → Rate limiting, auth, SSL termination                   │
│  │                                                            │
│  Layer 5: LOAD BALANCER / ROUTER                             │
│  │  NGINX, HAProxy, K8s Service, NVIDIA Dynamo               │
│  │  → Route requests to best available replica                │
│  │                                                            │
│  Layer 4: MODEL SERVING FRAMEWORK                            │
│  │  Triton Inference Server, KServe, TFServing, vLLM Server  │
│  │  → Multi-model, batching, model management, health checks │
│  │                                                            │
│  Layer 3: INFERENCE ENGINE                                   │
│  │  vLLM, SGLang, TensorRT-LLM, llama.cpp                   │
│  │  → Efficient token generation, KV-cache, scheduling       │
│  │                                                            │
│  Layer 2: RUNTIME / BACKEND                                  │
│  │  CUDA, ROCm, ONNX Runtime, ggml                          │
│  │  → Execute operations on hardware                         │
│  │                                                            │
│  Layer 1: HARDWARE                                           │
│  │  GPU, CPU, TPU, NPU                                       │
│  │                                                            │
│  └────────────────────────────────────────────────────────    │
└───────────────────────────────────────────────────────────────┘
```

## 🔍 Serving Solutions — Deep Comparison

### NVIDIA Triton Inference Server

| Aspect | Details |
|--------|---------|
| **What** | General-purpose model serving platform (not just LLMs) |
| **Supports** | TensorRT, ONNX Runtime, PyTorch, TensorFlow, Python, vLLM, TRT-LLM |
| **Key Features** | Dynamic batching, model ensemble, model analyzer, multi-model |
| **Protocol** | HTTP/REST, gRPC, binary tensor protocol |
| **Scaling** | K8s native, GPU sharing, concurrent model execution |
| **Best For** | Production multi-model serving (LLM + vision + embedding) |

```
┌──────── Triton Architecture ──────────────────┐
│                                                 │
│  Client → HTTP/gRPC → Triton Server            │
│                          │                      │
│           ┌──────────────┼──────────────┐      │
│           ▼              ▼              ▼      │
│     ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│     │ TRT-LLM  │  │ ONNX RT  │  │ PyTorch  │ │
│     │ Backend  │  │ Backend  │  │ Backend  │ │
│     │ (LLM)    │  │(Embedding│  │ (Custom) │ │
│     └──────────┘  └──────────┘  └──────────┘ │
│           │              │              │      │
│           ▼              ▼              ▼      │
│        GPU 0          GPU 1          GPU 2     │
│                                                 │
│  Model Repository:                              │
│  models/                                        │
│  ├── llama-7b/                                  │
│  │   ├── config.pbtxt    ← Model config        │
│  │   └── 1/              ← Version 1            │
│  │       └── model.plan  ← TensorRT engine      │
│  ├── embedder/                                   │
│  │   ├── config.pbtxt                            │
│  │   └── 1/                                      │
│  │       └── model.onnx                          │
│  └── reranker/                                   │
│      ├── config.pbtxt                            │
│      └── 1/                                      │
│          └── model.pt                            │
└──────────────────────────────────────────────────┘
```

### NVIDIA Dynamo — LLM Orchestration Layer (2025)

| Aspect | Details |
|--------|---------|
| **What** | Multi-node, multi-GPU LLM serving orchestrator |
| **Key Innovation** | Disaggregated serving: separate prefill and decode phases |
| **Architecture** | Planner + Worker model; smart routing across GPU cluster |
| **Integration** | Works with vLLM and TRT-LLM as backend engines |
| **Best For** | Large-scale multi-node LLM deployments |

```
┌──────── NVIDIA Dynamo Architecture ───────────┐
│                                                 │
│  ┌────────────┐                                │
│  │   Client    │                                │
│  └──────┬─────┘                                │
│         ▼                                       │
│  ┌────────────────┐                            │
│  │   API Gateway   │                            │
│  └──────┬─────────┘                            │
│         ▼                                       │
│  ┌────────────────────────────────────┐        │
│  │        Dynamo Planner              │        │
│  │  ┌────────────────────────────┐    │        │
│  │  │ Request Router             │    │        │
│  │  │ • KV-cache aware routing   │    │        │
│  │  │ • Prefill/decode splitting │    │        │
│  │  │ • Load-based scheduling    │    │        │
│  │  └────────────────────────────┘    │        │
│  └──────────┬────────────┬────────────┘        │
│             │            │                      │
│    ┌────────▼───┐  ┌─────▼────────┐            │
│    │  Prefill    │  │   Decode      │            │
│    │  Workers    │  │   Workers     │            │
│    │ (GPU 0-3)   │  │  (GPU 4-7)    │            │
│    │             │  │               │            │
│    │ Process     │  │ Generate      │            │
│    │ prompt,     │  │ tokens,       │            │
│    │ build KV    │  │ use KV cache  │            │
│    └─────────────┘  └───────────────┘            │
│                                                  │
│  Key: Prefill is compute-bound (batch together) │
│       Decode is memory-bound (run separately)    │
└──────────────────────────────────────────────────┘
```

### KServe (formerly KFServing)

| Aspect | Details |
|--------|---------|
| **What** | Kubernetes-native model serving platform |
| **Supports** | TensorFlow, PyTorch, ONNX, Triton, custom containers |
| **Key Features** | Auto-scaling (inc. scale-to-zero), canary rollouts, model explainability |
| **Protocol** | V2 Inference Protocol (Open Inference Protocol) |
| **Best For** | K8s-native ML platform teams |

```yaml
# KServe InferenceService manifest
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: llama-7b-awq
spec:
  predictor:
    model:
      modelFormat:
        name: pytorch
      storageUri: "s3://models/llama-7b-awq"
      resources:
        limits:
          nvidia.com/gpu: "1"
    containers:
    - name: kserve-container
      image: vllm/vllm-openai:latest
      args:
        - --model=/mnt/models
        - --quantization=awq
        - --max-model-len=4096
      resources:
        limits:
          nvidia.com/gpu: "1"
```

### vLLM as a Server

```
vLLM is both an INFERENCE ENGINE and a SERVING SOLUTION

┌──────── vLLM Server Architecture ──────────────┐
│                                                   │
│  vllm serve meta-llama/Llama-3-8B-AWQ            │
│  --quantization awq                               │
│  --tensor-parallel-size 2                         │
│  --max-model-len 8192                             │
│                                                   │
│  ┌─────────────────────────────────────────┐     │
│  │        OpenAI-Compatible API             │     │
│  │   POST /v1/chat/completions              │     │
│  │   POST /v1/completions                   │     │
│  │   POST /v1/embeddings                    │     │
│  ├─────────────────────────────────────────┤     │
│  │        AsyncLLMEngine                    │     │
│  │   ┌──────────────────────────────┐      │     │
│  │   │ Scheduler                    │      │     │
│  │   │ • Continuous batching        │      │     │
│  │   │ • Chunked prefill            │      │     │
│  │   │ • Preemption                 │      │     │
│  │   ├──────────────────────────────┤      │     │
│  │   │ PagedAttention (KV cache)    │      │     │
│  │   ├──────────────────────────────┤      │     │
│  │   │ Model Executor               │      │     │
│  │   │ • TP across GPUs             │      │     │
│  │   │ • Quantized kernels          │      │     │
│  │   └──────────────────────────────┘      │     │
│  └─────────────────────────────────────────┘     │
│                                                   │
│  This IS the serving layer + engine combined      │
└───────────────────────────────────────────────────┘
```

### SGLang as a Server

```python
# SGLang serves with its own runtime
# python -m sglang.launch_server \
#   --model meta-llama/Llama-3-8B-AWQ \
#   --quantization awq \
#   --tp 2

# SGLang unique serving features:
# 1. RadixAttention: Share KV-cache across requests with common prefixes
# 2. Compressed FSM: Faster structured output (JSON, regex)
# 3. Cache-aware scheduling: Route similar requests to same prefix cache
# 4. Multi-modal serving: Unified API for text + vision models
```

### TGI (Text Generation Inference) — HuggingFace

> 📎 *See also [Doc 03 — Inference Engines](./03_inference_engines.md) for how TGI compares architecturally to vLLM and SGLang as an inference engine.*

| Aspect | Details |
|--------|---------|
| **What** | HuggingFace's production LLM serving solution |
| **Language** | Rust (server) + Python (model loading) |
| **Key Features** | Token streaming, continuous batching, HF Hub integration |
| **Quantization** | AWQ, GPTQ, EETQ, BitsAndBytes, FP8, Marlin |
| **Best For** | Quick deployment from HuggingFace Hub |

## 📊 Serving Solution Comparison

| Feature | vLLM | SGLang | TRT-LLM + Triton | TGI | KServe |
|---------|------|--------|-------------------|-----|--------|
| **LLM-specific** | ✅ | ✅ | ✅ | ✅ | ❌ (general) |
| **OpenAI API** | ✅ | ✅ | ✅ | ⚠️ Partial | ❌ |
| **Continuous Batching** | ✅ | ✅ | ✅ | ✅ | Depends on backend |
| **Disaggregated Serving** | ✅ | ✅ | ✅ (Dynamo) | ❌ | ❌ |
| **Multi-model** | ❌ | ❌ | ✅ (Triton) | ❌ | ✅ |
| **Auto-scaling** | ❌ | ❌ | ❌ | ❌ | ✅ (K8s) |
| **Scale-to-zero** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Structured Output** | ✅ | ✅ (best) | ⚠️ | ✅ | ❌ |
| **Multi-GPU (TP)** | ✅ | ✅ | ✅ | ✅ | Depends |
| **Multi-node (PP)** | ✅ | ✅ | ✅ | ❌ | ❌ |
| **Speculative Decode** | ✅ | ✅ | ✅ | ✅ | ❌ |
| **LoRA serving** | ✅ | ✅ | ✅ | ✅ | ❌ |
| **AWQ support** | ✅ | ✅ | ✅ | ✅ | ❌ |
| **FP8 support** | ✅ | ✅ | ✅ | ✅ | ❌ |
| **AMD GPU** | ✅ | ✅ | ❌ | ⚠️ | ❌ |
| **Ease of setup** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |

## 🔗 Middleware Components

### What is "Middleware" in the LLM Stack?

```
Middleware = Software between your application and the inference engine

┌──── MIDDLEWARE COMPONENTS ────────────────────────────┐
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │  LLM Gateway / Router                            │ │
│  │  Examples: LiteLLM, Portkey, Martian              │ │
│  │  Purpose: Unified API across multiple LLM providers│ │
│  │           Fallback, retry, cost tracking           │ │
│  └──────────────────────────────────────────────────┘ │
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │  Prompt Management / Orchestration               │ │
│  │  Examples: LangChain, LlamaIndex, DSPy            │ │
│  │  Purpose: RAG pipelines, agents, tool use         │ │
│  └──────────────────────────────────────────────────┘ │
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │  Observability / Monitoring                       │ │
│  │  Examples: LangSmith, Weights & Biases, Helicone  │ │
│  │  Purpose: Trace requests, log tokens, debug        │ │
│  └──────────────────────────────────────────────────┘ │
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │  Guardrails / Safety                              │ │
│  │  Examples: Guardrails AI, NeMo Guardrails, Llama  │ │
│  │  Purpose: Content filtering, PII detection        │ │
│  └──────────────────────────────────────────────────┘ │
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │  Caching Layer                                    │ │
│  │  Examples: GPTCache, Redis + semantic similarity   │ │
│  │  Purpose: Cache common responses, reduce cost     │ │
│  └──────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

## 🏗️ Production Architecture Patterns

### Pattern 1: Simple Single-Model Serving

```
Best for: Small team, single model, moderate traffic

Client → vLLM Server (1-2 GPUs) → Response
           │
           └─ AWQ INT4 quantized model
              OpenAI-compatible API
              Continuous batching built-in

Deploy: docker run --gpus all vllm/vllm-openai \
          --model TheBloke/Llama-2-7B-AWQ --quantization awq
```

### Pattern 2: Multi-Model with Triton

```
Best for: Multiple models (LLM + embedding + reranker)

Client → Triton Inference Server
           │
           ├─ TRT-LLM Backend: Llama-3-70B (FP8, 4xH100)
           │    └─ Handles /v1/chat/completions
           │
           ├─ ONNX RT Backend: BGE-large embedding (1xL40)
           │    └─ Handles /v1/embeddings
           │
           └─ PyTorch Backend: Reranker (1xL40)
                └─ Handles /rerank
```

### Pattern 3: Large-Scale Disaggregated (NVIDIA Dynamo)

```
Best for: High-traffic, latency-sensitive, large models

                        ┌─────────────┐
Client → API Gateway → │   Dynamo     │
                        │   Planner    │
                        └──┬──────┬───┘
                           │      │
              ┌────────────▼┐  ┌──▼────────────┐
              │ Prefill Pool │  │  Decode Pool   │
              │  4× H100     │  │  8× H100       │
              │  (compute-   │  │  (memory-       │
              │   intensive) │  │   bandwidth)    │
              └──────────────┘  └────────────────┘
                     │                  ▲
                     │   KV-cache       │
                     └──── transfer ────┘
                         (NIXL/RDMA)

Key insight: Prefill and decode have different resource needs!
  Prefill: Process whole prompt → compute-bound → batch heavily
  Decode:  Generate 1 token    → memory-bound   → maximize throughput
```

### Pattern 4: Kubernetes-Native with KServe

```
Best for: ML platform teams, auto-scaling needed

┌──────── Kubernetes Cluster ──────────────────┐
│                                               │
│  KServe Controller                            │
│       │                                       │
│       ▼                                       │
│  InferenceService: llama-3-8b                │
│  ├── Predictor Pod (vLLM container)          │
│  │   └── GPU: 1x A100                        │
│  │                                            │
│  ├── HPA (Horizontal Pod Autoscaler)         │
│  │   └── Scale 1-4 replicas on GPU util      │
│  │                                            │
│  ├── Canary: 10% traffic → new version       │
│  │   └── v2 with FP8 quantization            │
│  │                                            │
│  └── Knative Serving                          │
│      └── Scale-to-zero when idle              │
│                                               │
│  Istio Service Mesh                           │
│  └── mTLS, traffic splitting, observability   │
└───────────────────────────────────────────────┘
```

## 📈 Key Serving Metrics

> 📎 *These metrics (TTFT, TPOT, ITL) are also defined in the [Doc 02 — Glossary](./02_glossary_and_concept_map.md) with additional context.*

| Metric | Definition | Typical Target |
|--------|-----------|---------------|
| **TTFT** | Time to First Token | < 500ms |
| **TPOT** | Time Per Output Token | < 50ms (20+ tok/s) |
| **ITL** | Inter-Token Latency | < 50ms |
| **Throughput** | Total tokens/sec across all requests | Maximize |
| **QPS** | Queries per second | Application dependent |
| **P50/P99 Latency** | Percentile latency | P99 < 2× P50 |
| **GPU Utilization** | Compute usage | > 70% |
| **KV-cache Hit Rate** | Prefix cache reuse | > 30% (SGLang excels) |

## 🔄 How Quantization Affects Serving

```
┌──────── Impact of Quantization on Serving ──────────┐
│                                                       │
│  1. MEMORY: Smaller model = more KV-cache space      │
│     FP16 70B: 140 GB → needs 2-4 GPUs               │
│     AWQ 70B:   35 GB → fits on 1× 80GB GPU          │
│     → More memory for KV-cache → larger batch sizes   │
│     → Higher throughput                               │
│                                                       │
│  2. BANDWIDTH: Fewer bytes to read per token         │
│     Decode is memory-bandwidth bound                  │
│     INT4 reads 4× fewer bytes than FP16              │
│     → Faster per-token generation                     │
│                                                       │
│  3. BATCH SIZE: Smaller model → larger batches       │
│     More concurrent requests served                   │
│     → Better GPU utilization                          │
│                                                       │
│  4. COST: Same performance on cheaper hardware       │
│     FP16 on H100 ≈ INT4 on L40S (much cheaper)      │
│     → 3-5× cost reduction possible                    │
│                                                       │
│  Tradeoff: Quality vs throughput vs cost              │
│  Your research directly determines these tradeoffs!   │
└───────────────────────────────────────────────────────┘
```

---

**Next**: [10 — Edge & Embedded Deployment →](./10_edge_and_embedded.md)
