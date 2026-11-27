#!/usr/bin/env bash
#
# run_server.sh
#
# Builds infernix-serve and starts it for a selected model on DGX Spark GB10.
# Automatically sets --model-id to the commit hash so benchmarks (such as tool-eval-bench)
# can target the exact build version.
#
# Usage:
#   ./tools/run_server.sh [options] <model_alias>
#
# Model aliases:
#   fastq       Qwen3.8-Flash-Next NVIDIA FP4-Local-Infernix-FastQ-RHT16 (FastQ-v5 + GdnInRHT16, default; alias: fastq-rht16)
#   fastq-v5    Qwen3.8-Flash-Next NVIDIA FP4-Local-Infernix-FastQ-v5 (FastQ-v4 + OutputHeadQ8)
#   fastq-v4    Qwen3.8-Flash-Next NVIDIA FP4-Local-Infernix-FastQ-v4 (HC8 + SharedNVFP4 + GdnOutNVFP4 + MidGdnNVFP4)
#   fastq-v3    Qwen3.8-Flash-Next NVIDIA FP4-Local-Infernix-FastQ-v3 (HC8 + SharedNVFP4 + GdnOutNVFP4)
#   fastq-v2    Qwen3.8-Flash-Next NVIDIA FP4-Local-Infernix-FastQ (HC8 + SharedNVFP4)
#   dense8      Qwen3.8-Flash-Next NVIDIA NVFP4 Dense8-Infernix
#   nvfp4       Qwen3.8-Flash-Next Local NVFP4-W4Dense (F4-Local-NVPF4-Dense)
#   iqs_3       Qwen3.8-Flash-Next GSQ-RCO IQ3_S-Infernix  (aliases: iq3s, iq3_s)
#   local-f4    Qwen3.8-Flash-Next NVIDIA FP4-Local-Infernix-New
#   hc4-w4dense Qwen3.8-Flash-Next NVIDIA FP4-Local-Infernix-HC4-W4Dense (alias: hc4)
#
# Examples:
#   ./tools/run_server.sh dense8
#   ./tools/run_server.sh --port 8000 nvfp4
#   ./tools/run_server.sh --max-concurrency 8 iqs_3
#
# Options:
#   --port PORT         Listen port (default: 8000)
#   --host HOST         Listen host (default: 0.0.0.0)
#   --model-id ID       Explicit model ID override (default: extracted commit hash)
#   --max-context N     Max context length (default: 262144)
#   --max-concurrency N Max concurrency (default: 4)
#   --reasoning-effort E  Default template reasoning effort (default: medium)
#   -h, --help          Show this help message and exit
#
# Environment overrides:
#   MODEL_PATH / NGRAM_PATH / NGRAM_RESIDENCY / NGRAM_CACHE_MIB (default: 16384)
#   KV_DTYPE (default: int8; set k8v4 for >512K KV or RAM-resident N-gram)
#   VISION (default: 0 for hc4, 1 for local-f4)
#   SPEC_BACKEND (default: mtp; set dflash2 to use draft model), NGRAM_DRAFT_TOKENS (default: 31),
#   NGRAM_MIN_MATCH (default: 8), INFERNIX_NGRAM_WIDTH_COST (optional override)
#   THINKING_FLAG (default: empty), REASONING_EFFORT (default: medium)
#   EXPERT_A4_ACTIVATIONS (default: 1 for FP4-Local)
#   EXPERT_TAIL_SKIP (default: 0; set 8 for legacy rank-based tail skip)
#   INFERNIX_DRAFT_EXPERT_PRUNE (default: 0.10; set 0.0 for exact top-10 draft verify)
#   SKIP_BUILD=1 to skip rebuilding infernix-serve
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODELS_DIR="/opt/etc/models"
DRAFT_PATH="${MODELS_DIR}/Qwen3.8-Flash-Next-DFlash2-Head/Qwen3.8-Flash-Next-DFlash2-Infernix.infernix"

KV_DTYPE="${KV_DTYPE:-int8}"
NGRAM_DRAFT_TOKENS="${NGRAM_DRAFT_TOKENS:-31}"
NGRAM_MIN_MATCH="${NGRAM_MIN_MATCH:-8}"
NGRAM_CACHE_MIB="${NGRAM_CACHE_MIB:-16384}"
PREFILL_CHUNK="${PREFILL_CHUNK:-16384}"
KV_CAPACITY="${KV_CAPACITY:-524288}"
HOST_CONTEXT_MIB="${HOST_CONTEXT_MIB:-4096}"
THINKING_FLAG="${THINKING_FLAG:-}"
REASONING_EFFORT="${REASONING_EFFORT:-medium}"
REASONING_LOOP="${REASONING_LOOP:-conclude}"
RESTORE_TOOL_REASONING="${RESTORE_TOOL_REASONING:-1}"
CHAT_TEMPLATE_PATH="${CHAT_TEMPLATE_PATH:-${REPO_ROOT}/tools/chat_templates/qwen3_8.jinja}"

HOST="0.0.0.0"
PORT="8000"
MAX_CONTEXT="262144"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"
EXPLICIT_MODEL_ID=""
MODEL_ALIAS="${MODEL_ALIAS:-}"
EXPERT_A4_DEFAULT=0
EXPERT_TAIL_SKIP="${EXPERT_TAIL_SKIP:-0}"
INFERNIX_DRAFT_EXPERT_PRUNE="${INFERNIX_DRAFT_EXPERT_PRUNE:-0.10}"
VISION_DEFAULT=1

# Parse options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --model-id)
            EXPLICIT_MODEL_ID="$2"
            shift 2
            ;;
        --max-context)
            MAX_CONTEXT="$2"
            shift 2
            ;;
        --max-concurrency)
            MAX_CONCURRENCY="$2"
            shift 2
            ;;
        --reasoning-effort)
            REASONING_EFFORT="$2"
            shift 2
            ;;
        -h|--help)
            sed -ne '/^#/!q;s/^# //;2,$p' "$0"
            exit 0
            ;;
        *)
            MODEL_ALIAS="$1"
            shift
            ;;
    esac
done

if [[ -z "${MODEL_ALIAS}" ]]; then
    MODEL_ALIAS="fastq"
fi

# Resolve model alias -> weights, n-gram volume, residency
case "${MODEL_ALIAS}" in
    fastq|fastq-rht16)
        MODEL_DIR="${MODELS_DIR}/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-FastQ-RHT16"
        MODEL_PATH="${MODEL_DIR}/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-FastQ-RHT16.infernix"
        NGRAM_PATH="${MODELS_DIR}/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-FastQ-v5/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix.ngram"
        NGRAM_RESIDENCY="${NGRAM_RESIDENCY:-volume}"
        SPEC_BACKEND="${SPEC_BACKEND:-mtp}"
        EXPERT_A4_DEFAULT=1
        VISION_DEFAULT=1
        ;;
    fastq-v5)
        MODEL_DIR="${MODELS_DIR}/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-FastQ-v5"
        MODEL_PATH="${MODEL_DIR}/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-FastQ-v5.infernix"
        NGRAM_PATH="${MODELS_DIR}/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-FastQ-v5/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix.ngram"
        NGRAM_RESIDENCY="${NGRAM_RESIDENCY:-volume}"
        SPEC_BACKEND="${SPEC_BACKEND:-mtp}"
        EXPERT_A4_DEFAULT=1
        VISION_DEFAULT=1
        ;;
    fastq-v4)
        MODEL_DIR="${MODELS_DIR}/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-FastQ-v4"
        MODEL_PATH="${MODEL_DIR}/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-FastQ-v4.infernix"
        NGRAM_PATH="${MODELS_DIR}/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-FastQ-v5/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix.ngram"
        NGRAM_RESIDENCY="${NGRAM_RESIDENCY:-volume}"
        SPEC_BACKEND="${SPEC_BACKEND:-mtp}"
        EXPERT_A4_DEFAULT=1
        VISION_DEFAULT=1
        ;;
    fastq-v3)
        MODEL_DIR="${MODELS_DIR}/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-FastQ-v3"
        MODEL_PATH="${MODEL_DIR}/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-FastQ-v3.infernix"
        NGRAM_PATH="${MODELS_DIR}/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-FastQ-v5/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix.ngram"
        NGRAM_RESIDENCY="${NGRAM_RESIDENCY:-volume}"
        SPEC_BACKEND="${SPEC_BACKEND:-mtp}"
        EXPERT_A4_DEFAULT=1
        VISION_DEFAULT=1
        ;;
    fastq-v2)
        MODEL_DIR="${MODELS_DIR}/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-FastQ"
        MODEL_PATH="${MODEL_DIR}/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-FastQ.infernix"
        NGRAM_PATH="${MODELS_DIR}/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-FastQ-v5/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix.ngram"
        NGRAM_RESIDENCY="${NGRAM_RESIDENCY:-volume}"
        SPEC_BACKEND="${SPEC_BACKEND:-mtp}"
        EXPERT_A4_DEFAULT=1
        VISION_DEFAULT=1
        ;;
    dense8)
        MODEL_DIR="${MODELS_DIR}/Qwen3.8-Flash-Next-Nvidia-Dense8-Infernix"
        MODEL_PATH="${MODEL_DIR}/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix-00001-of-00003.infernix"
        NGRAM_PATH="${MODEL_DIR}/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram"
        NGRAM_RESIDENCY="${NGRAM_RESIDENCY:-volume}"
        EXPERT_A4_DEFAULT=0
        VISION_DEFAULT=0
        ;;
    nvfp4)
        MODEL_DIR="${MODELS_DIR}/Qwen3.8-Flash-Next-F4-Local-NVPF4-Dense"
        MODEL_PATH="${MODEL_DIR}/Qwen3.8-Flash-Next-Local-NVFP4-W4Dense.infernix"
        NGRAM_PATH="${MODEL_DIR}/Qwen3.8-Flash-Next-Local-NVFP4.ngram"
        NGRAM_RESIDENCY="${NGRAM_RESIDENCY:-auto}"
        EXPERT_A4_DEFAULT=0
        VISION_DEFAULT=0
        ;;
    iqs_3|iq3s|iq3_s)
        MODEL_DIR="${MODELS_DIR}/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix"
        MODEL_PATH="${MODEL_DIR}/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix-00001-of-00002.infernix"
        NGRAM_PATH="${MODEL_DIR}/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.ngram"
        NGRAM_RESIDENCY="${NGRAM_RESIDENCY:-ram}"
        EXPERT_A4_DEFAULT=0
        VISION_DEFAULT=0
        ;;
    local-f4)
        MODEL_DIR="${MODELS_DIR}/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix"
        MODEL_PATH="${MODEL_DIR}/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-00001-of-00003.infernix"
        NGRAM_PATH="${MODEL_DIR}/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix.ngram"
        NGRAM_RESIDENCY="${NGRAM_RESIDENCY:-volume}"
        EXPERT_A4_DEFAULT=1
        VISION_DEFAULT=1
        ;;
    hc4-w4dense|hc4)
        MODEL_DIR="${MODELS_DIR}/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-HC4-W4Dense"
        MODEL_PATH="${MODEL_DIR}/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-HC4-W4Dense.infernix"
        NGRAM_PATH="${MODEL_DIR}/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix.ngram"
        NGRAM_RESIDENCY="${NGRAM_RESIDENCY:-volume}"
        SPEC_BACKEND="${SPEC_BACKEND:-mtp}"
        EXPERT_A4_DEFAULT=1
        VISION_DEFAULT=1
        ;;
    *)
        echo "Error: Unknown model alias '${MODEL_ALIAS}'." >&2
        echo "Available aliases: fastq (fastq-rht16), fastq-v5, fastq-v4, fastq-v3, fastq-v2, dense8, nvfp4, iqs_3 (iq3s/iq3_s), local-f4, hc4-w4dense (hc4)" >&2
        exit 1
        ;;
esac

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "Error: Model weights not found: ${MODEL_PATH}" >&2
    exit 1
fi
if [[ ! -e "${NGRAM_PATH}" ]]; then
    echo "Error: N-gram volume not found: ${NGRAM_PATH}" >&2
    exit 1
fi

# Build infernix-serve (incremental) and run only this binary
BINARY_PATH="${REPO_ROOT}/build-native/apps/infernix-serve"
if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    echo "Building infernix-serve..."
    cmake --build "${REPO_ROOT}/build-native" --target infernix-serve --parallel "$(nproc)"
fi
if [[ ! -x "${BINARY_PATH}" ]]; then
    echo "Error: infernix-serve binary not found after build: ${BINARY_PATH}" >&2
    exit 1
fi

# Model ID defaults to the current commit hash
if [[ -n "${EXPLICIT_MODEL_ID}" ]]; then
    MODEL_ID="${EXPLICIT_MODEL_ID}"
elif git -C "${REPO_ROOT}" rev-parse --short=8 HEAD >/dev/null 2>&1; then
    MODEL_ID="$(git -C "${REPO_ROOT}" rev-parse --short=8 HEAD)"
else
    MODEL_ID="infernix-model"
fi

SPEC_BACKEND="${SPEC_BACKEND:-mtp}"

NGRAM_DETAIL="${NGRAM_RESIDENCY}"
if [[ "${NGRAM_RESIDENCY}" == "volume" ]]; then
    NGRAM_DETAIL="${NGRAM_RESIDENCY}, ${NGRAM_CACHE_MIB} MiB cache"
fi

echo "========================================================================================"
echo " INFERNIX SERVER LAUNCHER (GB10 sm_121a)"
echo "========================================================================================"
echo " Binary          : ${BINARY_PATH}"
echo " Model Alias     : ${MODEL_ALIAS}"
echo " Model ID        : ${MODEL_ID}"
echo " Model Weights   : ${MODEL_PATH}"
echo " N-gram Volume   : ${NGRAM_PATH} (${NGRAM_DETAIL})"
echo " KV Dtype        : ${KV_DTYPE}"
echo " Listen Endpoint : http://${HOST}:${PORT}/v1"
echo " Max Context     : ${MAX_CONTEXT} tokens"
echo " Max Concurrency : ${MAX_CONCURRENCY}"
echo " Speculative     : ${SPEC_BACKEND} (auto) + N-gram (${NGRAM_DRAFT_TOKENS} drafts, min-match ${NGRAM_MIN_MATCH})"
echo " Expert Tail-Skip: ${EXPERT_TAIL_SKIP}"
echo " Draft Prune Tau : ${INFERNIX_DRAFT_EXPERT_PRUNE}"
echo "========================================================================================"
echo

export INFERNIX_EXPERTS_ON_GPU=1
export INFERNIX_DRAFT_EXPERT_PRUNE
if [[ -n "${INFERNIX_NGRAM_WIDTH_COST:-}" ]]; then
    export INFERNIX_NGRAM_WIDTH_COST
fi

EXPERT_A4_ARGS=()
if [[ "${EXPERT_A4_ACTIVATIONS:-${EXPERT_A4_DEFAULT}}" == "1" ]]; then
    EXPERT_A4_ARGS=(--expert-a4-activations)
fi

EXPERT_TAIL_SKIP_ARGS=()
if [[ "${EXPERT_TAIL_SKIP}" != "0" ]]; then
    EXPERT_TAIL_SKIP_ARGS=(--expert-tail-skip "${EXPERT_TAIL_SKIP}")
fi

VISION_ARGS=()
if [[ "${VISION:-${VISION_DEFAULT}}" == "1" ]]; then
    VISION_ARGS=(--vision)
fi

THINKING_ARGS=()
if [[ -n "${THINKING_FLAG}" ]]; then
    THINKING_ARGS=("${THINKING_FLAG}")
fi

REASONING_ARGS=()
if [[ -n "${REASONING_EFFORT}" ]]; then
    REASONING_ARGS=(--reasoning-effort "${REASONING_EFFORT}")
fi

RESTORE_ARGS=()
if [[ "${RESTORE_TOOL_REASONING}" == "1" ]]; then
    RESTORE_ARGS=(--restore-tool-reasoning)
fi

SPEC_ARGS=(--spec "${SPEC_BACKEND}" --draft-tokens auto)
if [[ "${SPEC_BACKEND}" == "dflash2" ]]; then
    SPEC_ARGS+=(--draft-model "${DRAFT_PATH}")
fi

exec "${BINARY_PATH}" "${MODEL_PATH}" \
    "${EXPERT_A4_ARGS[@]}" \
    "${EXPERT_TAIL_SKIP_ARGS[@]}" \
    "${VISION_ARGS[@]}" \
    "${SPEC_ARGS[@]}" \
    --ngram-volume "${NGRAM_PATH}" \
    --ngram-residency "${NGRAM_RESIDENCY}" \
    --ngram-cache-mib "${NGRAM_CACHE_MIB}" \
    --kv-dtype "${KV_DTYPE}" \
    --ngram-draft-tokens "${NGRAM_DRAFT_TOKENS}" \
    --ngram-min-match "${NGRAM_MIN_MATCH}" \
    --temperature 1.0 \
    --top-p 0.9 \
    --min-p 0.1 \
    --top-k 20 \
    --lm-head-draft \
    --prefill-chunk "${PREFILL_CHUNK}" \
    --kv-capacity "${KV_CAPACITY}" \
    --default-max-tokens 32768 \
    --preserve-thinking \
    --chat-template "${CHAT_TEMPLATE_PATH}" \
    "${RESTORE_ARGS[@]}" \
    "${THINKING_ARGS[@]}" \
    "${REASONING_ARGS[@]}" \
    --reasoning-loop "${REASONING_LOOP}" \
    --host-context-mib "${HOST_CONTEXT_MIB}" \
    --tolerant-tool-calls \
    --max-context "${MAX_CONTEXT}" \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --model-id "${MODEL_ID}"
