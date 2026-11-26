# Infernix — Retuned for NVIDIA DGX Spark (`NVIDIA GB10`, `sm_121a`)

**Infernix** is a high-performance, from-scratch C++20/CUDA inference engine for the Qwen3.5, Qwen3.6, Qwen3.8, and **Qwen3.8-Flash-Next** (`Qwen4ExpForCausalLM`) model families.

> **Project Lineage & DGX Spark (`GB10`) Retune:** This project is built on the **Infernix** inference engine (which originally began as a fork of [NInfer](https://github.com/Neroued/ninfer) by [Neroued](https://github.com/Neroued) targeting the desktop RTX 5090 `sm_120a`) and has been **architecturally retuned end-to-end for the NVIDIA DGX Spark (`NVIDIA GB10`, Blackwell `sm_121a`)** on 64-bit Linux `aarch64`. Every CUDA kernel family, `/shapes/` dispatch table (`103/103` shapes), TMA descriptor policy, L2 cache swizzle, unified-memory (UMA) expert placement, and multi-row Multi-Token Prediction (MTP) speculative scheduler has been calibrated specifically for the GB10's **48 SMs**, **24 MiB L2 cache**, **128 GB unified LPDDR5X memory**, and **20-core Grace ARMv9.2 CPU**.

---

## Key Architectural Highlights on NVIDIA DGX Spark (`GB10`)

### 1. Full-Model Unified Memory Residency for `Qwen3.8-Flash-Next` (`FastQ` / `FastQ-RHT16`)
On **DGX Spark's 128 GB unified LPDDR5X**, Infernix pins all 48 layers of routed experts directly in device-accessible unified memory (`INFERNIX_EXPERTS_ON_GPU=1`, zero PCIe miss transfers) and streams the **PLE (Per-Layer Embedding) n-gram weight table** volume with a 16 GiB GPU-direct UMA cache and 20-thread ARMv9.2 prefill workers.

### 2. Microarchitectural Retuning for Blackwell `sm_121a` (48 SMs, 24 MiB L2)
- **Native 2-Term Row-Scaled `NVFP4 A4` TMA (`OMMA.16864.F32.E2M1`, `501.6 TFLOP/s` pipe)**: Trunk and MoE projections dynamically quantize BF16 activations into two per-token row-scaled NVFP4 terms (`FastPath::Nvfp4A4Pair`) using 1-pass register-cached reciprocals and 2-CTA/SM `128×64` TMA tiles with `CU_TENSOR_MAP_L2_PROMOTION_L2_128B`.
- **24 MiB L2 Cache Thrashing Elimination**: Added 2D `Grouped, 8` CTA rasterization across all `BF16`, `FP8`, `Q8`, `NVFP4`, `Q6`, `Q5`, and `Q4` MMA kernels.
- **Direct FP32 Tensor Core Accumulation (`1:1` HMMA Parity on `sm_121a`)**: Replacing `mma_f16_f16acc` with direct FP32-accumulating `mma_f16` across dense causal attention (`fp8`, `k8v4`, `nvfp4`, `int8`, `vq2`) and Gated Delta Net (`chunked/recurrence.cu`) eliminated **100% of local-memory stack spills (`STACK = 0`)** while improving numerical precision.
- **Hardware `F2FP` + `HMUL2` Bit-Exact NVFP4 Codec & 2×/3× Occupancy Narrow MoE**: Hardware `__nv_cvt_fp4x2_to_halfraw2` + `__hmul2` (`0 ULP` exact in IEEE FP16), `48.0 KiB` static SMEM `Nvfp4SlicedInstance<16, 8, 2, 2>` (`2 CTAs/SM`), and register-quantized narrow MoE (`moe_layer.cu`, `STACK = 0`).

### 3. Production Serving & Agentic Capabilities
- **OpenAI & Anthropic APIs**: `/v1/chat/completions`, `/v1/responses`, `/v1/messages`, and `/v1/decide` with streaming, reasoning budgets (`--reasoning-effort`, `--restore-tool-reasoning`), tolerant XML/JSON tool-call recovery (`--tolerant-tool-calls`), and constrained decoding (JSON Schema, regex, GBNF, choice grammars).
- **Hybrid Prefix Cache**: Content-addressed 64-token KV blocks plus recurrent GDN/conv state snapshots across GPU, host RAM, and persistent disk tiers (`--prefix-cache-save-mins`).
- **Seven KV Cache Formats**: `bf16`, `fp8`, `int8`, `k8v4`, `nvfp4`, `k4v2`, and `vq2`.

---

## Supported Model (`FastQ-RHT16`)

| Model | Hugging Face Download | Description |
|---|---|---|
| **`Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-FastQ`** | [eugene-kamenev/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-FastQ](https://huggingface.co/eugene-kamenev/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-FastQ) | `Qwen3.8-Flash-Next` (`Qwen4ExpForCausalLM`) in `FastQ-RHT16` layout (`69.999 GiB` payload: `q8_g32_fp16` HC & output head, `nvfp4_rht16_mse` boundary GDN input projections, `fp8_e4m3fn_g32_e8m0` attention layers, `nvfp4` trunk/shared/mid-GDN) with `--vision`, `--expert-a4-activations`, external PLE n-gram embedding volume, and mixed MTP + n-gram copy speculation. |

---

## Quick Start (DGX Spark / Linux `aarch64`)

### 1. Prerequisites
Builds and runs natively on **64-bit Linux `aarch64`** (Ubuntu 24.04 LTS, CUDA 13.0, GCC 13.3):

```bash
sudo apt-get update && sudo apt-get install -y \
  build-essential cmake ninja-build pkg-config \
  libavformat-dev libavcodec-dev libavutil-dev libswscale-dev \
  libcurl4-openssl-dev
```

### 2. Build
```bash
cmake -B build-native -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=121a
ninja -C build-native -j "$(nproc)" infernix-serve infernix_bench
```

> **WARNING:** The repository [`Dockerfile`](Dockerfile) / Docker image has not been well tested yet and should **not** be used for production runs. Build and run natively on the host (Ubuntu 24.04 LTS, CUDA 13.0) instead.

### 3. Run the Server (`infernix-serve`)

> **IMPORTANT:** The chat template [`tools/chat_templates/qwen3_8.jinja`](tools/chat_templates/qwen3_8.jinja) **must be used** (`--chat-template tools/chat_templates/qwen3_8.jinja`) for proper thinking, tool-calling, and structured JSON output behavior.

Use [`tools/run_server.sh`](tools/run_server.sh) (which builds `infernix-serve` and passes `--chat-template tools/chat_templates/qwen3_8.jinja` automatically):

```bash
./tools/run_server.sh fastq
```

Or launch `build-native/apps/infernix-serve` directly with the default `FastQ-RHT16` command:

```bash
INFERNIX_EXPERTS_ON_GPU=1 \
./build-native/apps/infernix-serve \
  /opt/etc/models/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-FastQ-RHT16/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-FastQ-RHT16.infernix \
  --expert-a4-activations \
  --vision \
  --spec mtp --draft-tokens auto \
  --ngram-draft-tokens 31 \
  --ngram-min-match 8 \
  --ngram-volume /opt/etc/models/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-FastQ-RHT16/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix.ngram \
  --ngram-residency volume \
  --ngram-cache-mib 16384 \
  --kv-dtype int8 \
  --temperature 1.0 \
  --top-p 0.9 \
  --min-p 0.1 \
  --top-k 20 \
  --lm-head-draft \
  --prefill-chunk 12288 \
  --kv-capacity 524288 \
  --default-max-tokens 32768 \
  --preserve-thinking \
  --chat-template tools/chat_templates/qwen3_8.jinja \
  --restore-tool-reasoning \
  --reasoning-effort medium \
  --reasoning-loop conclude \
  --host-context-mib 4096 \
  --tolerant-tool-calls \
  --max-context 262144 \
  --max-concurrency 8 \
  --host 0.0.0.0 \
  --port 8000
```

#### Handling Concurrent Bursts (Agents & Proxies)

`infernix-serve` runs at most `--max-concurrency` active requests at once (maximum `8`). Additional requests wait in a FIFO queue of size `--max-pending-requests` (default `16`, for `24` total in-flight requests at `--max-concurrency 8`; exceeding that returns HTTP `429` `server_overloaded`) and fail with **HTTP `503` (`request_queue_timeout`)** if they wait longer than `--pending-timeout-ms` (default **`30000` ms / 30 s**, covering preparation and queue wait before admission).

For example, a burst of 16 simultaneous requests (`C16`) against `--max-concurrency 8` queues 8 requests behind the active lanes; if those lanes take more than 30 s to free up, the queued requests time out with HTTP `503`. When running `infernix-serve` behind agent harnesses or a proxy where bursts above 8 are normal, raise `--pending-timeout-ms` (and `--max-pending-requests` if bursts exceed `24` total requests):

```bash
--pending-timeout-ms 300000 \
--max-pending-requests 32
```

Once a request is admitted to an active lane, `--pending-timeout-ms` no longer applies. See [HTTP Serving — Execution behavior](docs/serving.md#execution-behavior) for full queueing and scheduling details.

#### Tuning the PLE N-gram Embedding Cache (`--ngram-cache-mib`)

> **Note:** `infernix-serve` uses `--ngram-*` flags for two unrelated features:
> - **PLE (Per-Layer Embedding) weight table** (`--ngram-volume`, `--ngram-residency`, `--ngram-cache-mib`): External Per-Layer Embedding weights read during forward execution in `Qwen3.8-Flash-Next`.
> - **N-gram copy speculation** (`--ngram-draft-tokens`, `--ngram-min-match`): Token-sequence matching from the prompt/generation history that proposes draft continuations alongside `--spec mtp`.

When streaming the PLE n-gram weight table from NVMe (`--ngram-residency volume`), `--ngram-cache-mib` sets the budget for the 4-way set-associative host/UMA row cache (rounded down to the largest power-of-two slot count from `2^16` to `2^28` rows that fits in the budget; `0` uses the default `2^20` rows).

The example command above uses `--ngram-cache-mib 16384` (configurable via `NGRAM_CACHE_MIB=8192 ./tools/run_server.sh fastq`), whereas a tighter budget such as `--ngram-cache-mib 2048` reduces unified memory footprint shared with `--kv-capacity` and `--host-context-mib` at the cost of more NVMe read misses. Tune it in power-of-two steps (`2048`–`16384`) for your workload, and run with `--log-level debug` to inspect per-request `n-gram rows` cache hit rates and NVMe read time (see [Qwen3.8-Flash-Next Guide — N-gram table in RAM](docs/qwen3_8-flash-next.md#n-gram-table-in-ram)).

---

## Benchmarks on NVIDIA DGX Spark (`NVIDIA GB10`)

### Benchmark Environment

| Component | Specification |
|---|---|
| **System / Device** | **NVIDIA DGX Spark** |
| **GPU** | **NVIDIA GB10** (Blackwell `sm_121a`, Compute Capability `12.1`, **48 SMs** @ up to `3,003 MHz`, **24 MiB L2 cache**) |
| **Memory Subsystem** | **128 GB Unified LPDDR5X** (`cudaDevAttrIntegrated = 1`, `~121 GiB` usable; `273.0 GB/s` theoretical peak, `247.6 GB/s` measured sustained read) |
| **CPU** | **20-core NVIDIA Grace ARMv9.2 (`aarch64`)** — `10× Cortex-X925` (`r0p1`) + `10× Cortex-A725` (`r0p1`), `48 MiB` L3 cache (`2 × 24 MiB`) |
| **Operating System** | **Ubuntu 24.04.5 LTS** (`Noble Numbat`, 64-bit `aarch64`) |
| **CUDA Toolkit** | **CUDA `13.0`** |
| **Host C++ Compiler** | **GCC `13.3.0`** |

### `llama-benchy` Results (`FastQ-RHT16`, `int8` KV Cache)

Measured via `tool-eval-bench bench --perf-only` (`llama-benchy 0.4.0`, synthetic prompts with `0%` prefix-cache reuse). Model loading takes **~8s**; full engine startup and CUDA graph capture takes **~14–16s** (`68.80 GiB` GPU weights).

| Test | c | pp t/s | tg t/s | TTFT (ms) | Total (ms) |
|---|---:|---:|---:|---:|---:|
| pp2048 tg128 @ d0 | c1 | 3,366* | 84.0 | 608 | 2,120 |
| pp2048 tg128 @ d0 | c2 | 3,076 | 118.3 | 1,197 | 3,334 |
| pp2048 tg128 @ d0 | c4 | 3,143 | 185.5 | 2,339 | 4,988 |
| pp2048 tg128 @ d4096 | c1 | 3,806* | 82.5 | 1,614 | 3,153 |
| pp2048 tg128 @ d4096 | c2 | 3,546 | 112.5 | 3,111 | 5,307 |
| pp2048 tg128 @ d4096 | c4 | 3,524 | 170.0 | 6,298 | 9,081 |
| pp2048 tg128 @ d8192 | c1 | 4,036* | 86.4 | 2,537 | 4,007 |
| pp2048 tg128 @ d8192 | c2 | 3,694 | 106.0 | 5,020 | 7,333 |
| pp2048 tg128 @ d8192 | c4 | 3,697 | 164.9 | 10,115 | 13,040 |

\* `c = 1` prefill column is an estimator output, not wall-clock throughput — only `TTFT`-derived rates are meaningful at `c = 1`.

### `262K` NIAH Gate Results (`FastQ-RHT16`, `2048` Needles, `int8` KV Cache)

Measured via [`tools/niah_gate/niah_gate.py`](tools/niah_gate/niah_gate.py) (`--max-context 262144 --needles 2048 --runs 2 --seed 1234 --reasoning-effort medium`, `171,095`-token prompt, `61.8K`-token output, `4096 MiB` prefix cache):

| Run | Gate Verdict | Prefill | Decode (`tok/s`) | Speculative Acceptance | Total Time |
|---|---|---:|---:|---|---|
| **Calibration / Cold Prefill** (`11.7K` / `171.1K`) | — | `3.68k` / `3.51k tok/s` | — | — | `3.2s` / `48.9s` |
| **Run 1** (`93.4%` cached tail `11.4K`) | **PASS (`2048/2048`, `dup=0`, `wrong=0`, `stop`)** | `3.25k tok/s` | `128.3 tok/s` | `97.3%` (`53,666/55,155` MTP+ngram) | `8m 5.8s` |
| **Run 2** (`100.0%` cached hybrid snapshot) | **PASS (`2048/2048`, `dup=0`, `wrong=0`, `stop`)** | — | **`281.8 tok/s`** | `99.6%` (`59,043/59,306`, `52,145` ngram) | `3m 39.6s` |

### `tool-eval-bench` Results (`HC4-W4Dense`, `tool-eval-bench v3.0.1.dev10+g1cd7309e5`)

| Metric | Value |
|---|---|
| **Score** | **99 / 100** (`136 / 138` points, ★★★★★ Excellent) |
| **Scenarios** | **68 passed**, **0 partial**, **1 failed** |
| **Quality** | **99 / 100** |
| **Responsiveness** | **67 / 100** (median turn: **1.9s**) |
| **Deployability** | **89 / 100** (`α = 0.7`) |

---

## Numerical Quality & Fast-Path Gate

Infernix provides two numerical execution routes:

- **The exact route (`--disable-fast-paths`)**: Produces bit-identical output from run to run and release to release. Where an artifact's formats allow it, it is also request-invariant (concurrent requests, prefix-cache state, and memory pressure do not alter a request's output).
- **Fast paths (enabled by default)**: Hardware-accelerated schedules and dynamic activation quantizations (`nvfp4-a4-pair`, `qsa-prepare-fused`, `gdn-f16-accumulate`, `sampled-mtp-drafts`, `adaptive-speculation`) that run only for the kernels they were qualified on, only after passing the numerical quality gate below, and only where measured faster on GB10.

### The Quality Gate
Every fast path is qualified against a high-precision reference (16-bit activations in every layer and a BF16 KV cache) over a 1.04M-token teacher-forced prefill corpus and 1,024 forced decoding segments (65,536 scored positions):
- **Shared budgets (at the 95% upper confidence bound)**: KL divergence $\le 1.5 \times 10^{-3}$, top-1 token agreement within $0.5$ percentage points, decode total variation (TV) $\le 4 \times 10^{-3}$, and 99.9th-percentile KL growth $\le 25\%$.
- Full specification: [Numerical Acceptance Policy (`docs/maintainer/model-quality.md`)](docs/maintainer/model-quality.md#6-numerical-acceptance-policy-and-fast-paths) and [GB10 Optimization Roadmap (`docs/gb10_roadmap.md`)](docs/gb10_roadmap.md).

---

## Documentation

- **DGX Spark (`GB10`) Engineering & Performance**:
  - [DGX Spark (`NVIDIA GB10`) Optimization Roadmap & Hardware Ceiling Audit](docs/gb10_roadmap.md)
  - [Performance Overview](docs/performance.md)
- **User & Product Guides**:
  - [CLI Guide](docs/cli.md)
  - [HTTP Serving & API Reference](docs/serving.md)
  - [Qwen3.8-Flash-Next Guide](docs/qwen3_8-flash-next.md)
  - [Perplexity & Quality Evaluation](docs/perplexity.md)
  - [Weight Conversion Guide](docs/weight-conversion.md)
- **Maintainer Architecture**:
  - [Documentation Map](docs/README.md)
  - [Engine Architecture](docs/maintainer/engine-architecture.md)
  - [Op Development & Qualification](docs/maintainer/op-development.md)

---

## Credits & Acknowledgments

- **Infernix** ([Wallawalla47](https://github.com/Wallawalla47)) provided the base engine, `.infernix` v3 container format, hybrid prefix cache, and `Qwen3.8-Flash-Next` implementation that this repository retuned for the **NVIDIA DGX Spark (`NVIDIA GB10`, `sm_121a`)**.
- **[NInfer](https://github.com/Neroued/ninfer)** by **[Neroued](https://github.com/Neroued)** ([Ko-fi](https://ko-fi.com/neroued)) created the original C++/CUDA engine and artifact architecture that Infernix grew from.
- **[Strata](https://github.com/Niko1221/Strata)** by **[Niko1221](https://github.com/Niko1221)** pioneered single-GPU `Qwen3.8-Flash-Next` execution and mathematical transcription.
- Thanks also to [IST-DASLab](https://github.com/IST-DASLab) (`llmq`), the authors of HyperQuant and Four Over Six, the Qwen team for the models, NVIDIA for the NVFP4 checkpoints, and all open-source contributors whose research and ideas shaped this engine.
