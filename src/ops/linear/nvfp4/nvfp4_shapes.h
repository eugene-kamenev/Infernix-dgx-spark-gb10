#pragma once
#include "infernix/ops/linear.h"
#include "ops/linear/nvfp4/nvfp4_launch.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"

namespace infernix::ops::detail {
struct Nvfp4LinearShape {
    std::int32_t n, k;
    Nvfp4Launch a16;
    void (*a4)(const Tensor&, const Weight&, Tensor&, Nvfp4A4Workspace, cudaStream_t);
    bool (*uses_a4)(std::int32_t min_tokens, std::int32_t max_tokens);
    void (*pair)(const Tensor&, const Weight&, Tensor&, Nvfp4A4Workspace, cudaStream_t) = nullptr;
    std::int32_t pair_min_tokens = 0;
};

extern const Nvfp4LinearShape kNvfp4N14336K5120;
extern const Nvfp4LinearShape kNvfp4N16384K5120;
extern const Nvfp4LinearShape kNvfp4N34816K5120;
extern const Nvfp4LinearShape kNvfp4N5120K6144;
extern const Nvfp4LinearShape kNvfp4N5120K17408;
// Qwen3.8-Flash-Next dense Tier-1 shapes (A16-only).
extern const Nvfp4LinearShape kNvfp4N16384K2560;
extern const Nvfp4LinearShape kNvfp4N13952K2560;
extern const Nvfp4LinearShape kNvfp4N12800K2560;
extern const Nvfp4LinearShape kNvfp4N2560K6144;
// The [16384,2560] GDN projection on the same routes, rows [0,10240) written to qkv [10240,T] and
// rows [10240,16384) to z [6144,T] (gdn_input_proj).
void launch_nvfp4_n16384_k2560_split(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                     cudaStream_t stream,
                                     LinearPolicy policy = LinearPolicy::A16Only,
                                     LinearRoutes routes = LinearRoutes::Exact,
                                     WorkspaceArena* workspace = nullptr);
// The [13952,2560] QSA projection on the same routes, writing q [6144,T], gate [6144,T], k [512,T],
// v [512,T], iq [512,T], and ik [128,T] directly (attn_input_proj).
void launch_nvfp4_n13952_k2560_split(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, Tensor& iq, Tensor& ik,
                                     cudaStream_t stream,
                                     LinearPolicy policy = LinearPolicy::A16Only,
                                     LinearRoutes routes = LinearRoutes::Exact,
                                     WorkspaceArena* workspace = nullptr);
// The same projection over x [2560,B] (B = 1..8 sequences of one position each) with the GDN causal
// convolution in its epilogue (gdn_input_proj_conv_snapshot's routes form).
void launch_nvfp4_n16384_k2560_conv_snapshot(const Tensor& x, const Weight& weight,
                                             const Tensor& conv_weight, Tensor& conv_states,
                                             const Tensor& initial_state_slots,
                                             const Tensor& snapshot_base_slots, Tensor& query,
                                             Tensor& key, Tensor& value, Tensor& z,
                                             cudaStream_t stream,
                                             WorkspaceArena* workspace = nullptr);
// Qwen3.8-Flash-Next W4-dense tier (recipe `qwen3_8_flash_next_nvfp4_w4dense`).
extern const Nvfp4LinearShape kNvfp4N2048K2560;
extern const Nvfp4LinearShape kNvfp4N6144K2560;
extern const Nvfp4LinearShape kNvfp4N512K2560;
extern const Nvfp4LinearShape kNvfp4N128K2560;
extern const Nvfp4LinearShape kNvfp4N10240K2560;
extern const Nvfp4LinearShape kNvfp4N2560K2560;
extern const Nvfp4LinearShape kNvfp4N1280K2560;
extern const Nvfp4LinearShape kNvfp4N2560K640;
// Inkling 2 Mini DFlash2 drafter (block_scale_k16_m128x4_v1, A16Only).
extern const Nvfp4LinearShape kNvfp4N13824K2304;
extern const Nvfp4LinearShape kNvfp4N2304K6912;
} // namespace infernix::ops::detail
