// Goal-2500 Step 6 (tensor-core int8 mma MMQ for indexed-MoE). The core
// `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` PTX wrapper is adapted
// from llama.cpp's ggml-cuda backend:
//   https://github.com/ggml-org/llama.cpp/blob/0eadefebd3f8f92a86d634a0e5b8fffc9dc792c0/ggml/src/ggml-cuda/mma.cuh
// Pinned commit 0eadefebd3f8f92a86d634a0e5b8fffc9dc792c0 (fetched 2026-09-01) --
// port from this exact commit, not a moving `master`. llama.cpp is MIT
// licensed (Copyright (c) 2023-2026 The ggml authors).
//
// The per-thread FRAGMENT LAYOUT below (which (row/col, k-range) each of a
// thread's registers holds) is NOT copied from llama.cpp's `tile::get_i`/
// `get_j` -- an initial attempt to reuse those produced a deterministic
// but wrong result (verified against a host int32 reference, 127/128
// elements mismatched). The mapping actually used here was independently
// derived (kona-a0, 2026-09-01) from the PTX ISA's m16n8k32 integer
// fragment convention and VERIFIED against a host int32 reference with
// random inputs, MATCH x3 (test_mma_primitive.cu) -- the harness is the
// source of truth for this file, not either derivation.
//
// Deliberately self-contained: does NOT include or modify quantized.cu's
// existing `indexed_mul_mat_q_moe` dp4a template. The task-scan +
// odd-even-sort + grid-stride column-tiling preamble that template uses
// will be copied verbatim (not shared/refactored) into this file's own
// kernels when they're built -- see quantized.cu's own comment on why
// that sort exists (a real cross-block task-order race, cornered via
// test_immq.cu) before ever touching it.
//
// Ampere-only for now (sm_80, __CUDA_ARCH__ >= 800): this campaign's env-
// override convention (CEREBRA_MMQ_GEOM/COL_STRIDE precedent) applies here
// too once a second GPU class needs measuring -- never hardcode for one
// device, but do not speculatively build Turing's 4x-m8n8k16 fallback path
// before an actual second target exists to measure it against.

#pragma once

#include "cuda_fp16.h"
#include <stdint.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define CEREBRA_MMA_AVAILABLE
#endif

namespace cerebra_mma {

// Raw block layouts, matching candle-kernels/src/moe/gguf.cuh's
// block_q4_K / block_q8_1 EXACTLY (QK_K=256 branch: half2 dm packs
// d=dm.x, dmin=dm.y; scales[3*QK_K/64]=scales[12]; qs[QK_K/2]=qs[128].
// block_q8_1: half2 ds packs delta=ds.x, sum=ds.y; qs[QK8_0]=qs[32]).
// Named _raw to avoid colliding with the real (identical-layout) structs
// already defined elsewhere in this translation unit when this header is
// included alongside quantized.cu.
struct block_q4_K_raw {
    half2 dm;
    uint8_t scales[12];
    uint8_t qs[128];
};
struct block_q8_1_raw {
    half2 ds;
    int8_t qs[32];
};

// Plain register-array tiles -- `I` rows x `J` int32-packed columns (each
// int32 packs 4 int8 elements along the contraction dimension K). No
// get_i/get_j: the verified fragment layout lives in the pack/unpack
// helpers below, which are the tested source of truth (see file header).
template <int I_, int J_>
struct tile {
    static constexpr int I = I_;
    static constexpr int J = J_;
    static constexpr int ne = I * J / 32;
    int x[ne] = {0};
};

// `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` -- one warp instruction
// computing D[16,8] += A[16,32] @ B[8,32]^T in int8, accumulating int32.
static __device__ __forceinline__ void mma(tile<16, 8> &D, const tile<16, 8> &A, const tile<8, 8> &B) {
#ifdef CEREBRA_MMA_AVAILABLE
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(D.x[0]), "+r"(D.x[1]), "+r"(D.x[2]), "+r"(D.x[3])
        : "r"(A.x[0]), "r"(A.x[1]), "r"(A.x[2]), "r"(A.x[3]), "r"(B.x[0]), "r"(B.x[1]));
#endif
}

// Pack 4 consecutive int8 bytes starting at `base[offset]` into one int32
// (little-endian: base[offset] is the low byte). Local helper, not a PTX
// primitive -- shared by the fragment loaders below.
static __device__ __forceinline__ int32_t pack4_i8(const int8_t *base, int offset) {
    int32_t packed = 0;
#pragma unroll
    for (int b8 = 0; b8 < 4; ++b8) {
        packed |= (static_cast<uint32_t>(static_cast<uint8_t>(base[offset + b8])) << (8 * b8));
    }
    return packed;
}

// VERIFIED fragment layout (test_mma_primitive.cu, MATCH x3 on random
// inputs) for lane = threadIdx.x % 32, group = lane >> 2, tig = lane & 3.
// K=32 splits into two k16 halves; low byte of each packed int32 is the
// lowest K index in its 4-byte group.

// Loads the A fragment (16 rows x 32 int8 K, row-major, `row_stride` in
// int8 elements) for this thread's lane, from `a_rows[row*row_stride +
// k]`.
static __device__ __forceinline__ tile<16, 8> load_a_fragment(const int8_t *__restrict__ a_rows, int row_stride) {
    const int lane = threadIdx.x % 32;
    const int group = lane >> 2;
    const int tig = lane & 3;
    tile<16, 8> a;
    a.x[0] = pack4_i8(a_rows, group * row_stride + 4 * tig);
    a.x[1] = pack4_i8(a_rows, (group + 8) * row_stride + 4 * tig);
    a.x[2] = pack4_i8(a_rows, group * row_stride + 16 + 4 * tig);
    a.x[3] = pack4_i8(a_rows, (group + 8) * row_stride + 16 + 4 * tig);
    return a;
}

// Loads the B fragment (8 cols x 32 int8 K, row-major PER COLUMN i.e.
// `b_cols[col*col_stride + k]` -- matches mma's ".col" operand, which for
// int8 GEMM means B is stored as [N,K] just like A is [M,K], contracted
// against the SAME K axis) for this thread's lane.
static __device__ __forceinline__ tile<8, 8> load_b_fragment(const int8_t *__restrict__ b_cols, int col_stride) {
    const int lane = threadIdx.x % 32;
    const int group = lane >> 2;
    const int tig = lane & 3;
    tile<8, 8> b;
    b.x[0] = pack4_i8(b_cols, group * col_stride + 4 * tig);
    b.x[1] = pack4_i8(b_cols, group * col_stride + 16 + 4 * tig);
    return b;
}

// Scatters the D/accumulator fragment (16x8 int32, `out[row*out_stride +
// col]`) for this thread's lane.
static __device__ __forceinline__ void store_d_fragment(int32_t *__restrict__ out, int out_stride, const tile<16, 8> &d) {
    const int lane = threadIdx.x % 32;
    const int group = lane >> 2;
    const int tig = lane & 3;
    out[group * out_stride + 2 * tig] = d.x[0];
    out[group * out_stride + 2 * tig + 1] = d.x[1];
    out[(group + 8) * out_stride + 2 * tig] = d.x[2];
    out[(group + 8) * out_stride + 2 * tig + 1] = d.x[3];
}

} // namespace cerebra_mma
