// Goal-2500 Step 6: tensor-core int8 mma MMQ, Q4_K indexed-MoE entry
// point. The task-scan + odd-even-sort + grid-stride column-tiling
// preamble below is copied VERBATIM (not refactored) from
// `indexed_mul_mat_q_moe` in quantized.cu -- see that template's own
// comment for why the sort exists (a real cross-block task-order race,
// cornered via test_immq.cu). Do not "clean up" this preamble independent
// of that one; they must stay identical.
//
// The compute core (Q4_K dequant -> int8 mma tiles -> scale-combine) is
// the same logic verified unit-by-unit in test_mma_primitive.cu /
// test_mma_q4k_unit.cu / test_mma_q4k_fullrow.cu / test_mma_q4k_multiblock.cu
// (all MATCH x3 on real hardware before this file was written) -- see
// mmq_mma.cuh's own header for the ported-primitive provenance.
//
// Shape: one warp (32 threads, nwarps=1) per (16-row tile, 8-task tile).
// mmq_y=16, mmq_x=8 are NOT tunable knobs here the way dp4a's mmq_x/y
// are -- they are fixed by the mma.m16n8k32 instruction's own shape.

#pragma once
#include "mmq_mma.cuh"

static __host__ __device__ void mma_get_scale_min_k4(int j, const uint8_t *q, uint8_t &d, uint8_t &m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

extern "C" __global__ void indexed_mul_mat_q4_K_moe_mma(
    const void *__restrict__ all_weights,
    const void *__restrict__ all_inputs, // block_q8_1 array
    const unsigned int *__restrict__ indices,
    float *__restrict__ all_outputs,
    const int n,   // output rows per expert
    const int k,   // input width (must be a multiple of QK_K=256)
    const int batch,
    const int topk,
    const int k_padded, // nrows_y (Q8_1-padded)
    const int input_dim1) {
#define IMMQ_MMA_MAX_TASKS 1024
    const unsigned int expert_id = blockIdx.z;
    const int total_tasks = batch * topk;

    __shared__ unsigned short task_list[IMMQ_MMA_MAX_TASKS];
    __shared__ int task_count;
    const int tid_flat = (int)threadIdx.x; // nwarps=1 -- flat == threadIdx.x
    if (tid_flat == 0) {
        task_count = 0;
    }
    __syncthreads();
    for (int t = tid_flat; t < total_tasks; t += 32) {
        if (indices[t] == expert_id) {
            const int slot = atomicAdd(&task_count, 1);
            if (slot < IMMQ_MMA_MAX_TASKS) {
                task_list[slot] = (unsigned short)t;
            }
        }
    }
    __syncthreads();
    const int count = task_count < IMMQ_MMA_MAX_TASKS ? task_count : IMMQ_MMA_MAX_TASKS;
    if (count == 0) {
        return;
    }

    // CANONICALIZE the list order -- verbatim rationale from
    // indexed_mul_mat_q_moe: the atomicAdd append order is nondeterministic
    // PER BLOCK, and every block covering a different row tile of this
    // expert must agree which task sits at which column.
    for (int phase = 0; phase < count; ++phase) {
        const int start = phase & 1;
        for (int p = start + 2 * tid_flat; p + 1 < count; p += 2 * 32) {
            const unsigned short a = task_list[p];
            const unsigned short b = task_list[p + 1];
            if (a > b) {
                task_list[p] = b;
                task_list[p + 1] = a;
            }
        }
        __syncthreads();
    }

    const int QK_K_LOCAL = 256;
    const size_t weight_expert_stride_bytes = (size_t)(n * k) / QK_K_LOCAL * sizeof(cerebra_mma::block_q4_K_raw);
    const cerebra_mma::block_q4_K_raw *x =
        (const cerebra_mma::block_q4_K_raw *)((const char *)all_weights + (size_t)expert_id * weight_expert_stride_bytes);
    const cerebra_mma::block_q8_1_raw *y = (const cerebra_mma::block_q8_1_raw *)all_inputs;

    const int blocks_per_row_x = k / QK_K_LOCAL;
    const int blocks_per_col_y = k_padded / 32; // QK8_1=32

    const int row_dst_0 = blockIdx.x * 16;
    if (row_dst_0 >= n) {
        return;
    }
    const int rows_here = min(16, n - row_dst_0);

    __shared__ float acc[16 * 8];
    __shared__ int8_t a_nibbles[16 * 32];
    __shared__ int8_t b_quants[8 * 32];
    __shared__ int32_t dot_int[16 * 8];

    for (int col_tile = blockIdx.y; col_tile * 8 < count; col_tile += gridDim.y) {
        const int col_dst_0 = col_tile * 8;
        const int tasks_here = min(8, count - col_dst_0);

        for (int idx = tid_flat; idx < 16 * 8; idx += 32) acc[idx] = 0.0f;
        __syncthreads();

        for (int ib0 = 0; ib0 < blocks_per_row_x; ++ib0) {
            for (int il = 0; il < 4; ++il) {
                for (int half = 0; half < 2; ++half) {
                    const int is = 2 * il + half;

                    if (tid_flat < 32) {
                        const int row_local = tid_flat / 2, t2 = tid_flat % 2;
                        const int row = row_dst_0 + min(row_local, rows_here - 1);
                        const cerebra_mma::block_q4_K_raw &bxi = x[(size_t)row * blocks_per_row_x + ib0];
                        for (int b = 0; b < 16; ++b) {
                            const int b_local = t2 * 16 + b;
                            const uint8_t byte = bxi.qs[32 * il + b_local];
                            a_nibbles[row_local * 32 + b_local] = static_cast<int8_t>(half == 0 ? (byte & 0x0F) : (byte >> 4));
                        }
                    }
                    if (tid_flat < 32) {
                        const int task_local = tid_flat / 4, quarter = tid_flat % 4;
                        const int col_local = min(col_dst_0 + task_local, count - 1);
                        const int t = (int)task_list[col_local];
                        const int input_row = (input_dim1 == 1) ? (t / topk) : t;
                        const cerebra_mma::block_q8_1_raw &byi = y[(size_t)input_row * blocks_per_col_y + ib0 * 8 + is];
                        for (int b = 0; b < 8; ++b) {
                            b_quants[task_local * 32 + quarter * 8 + b] = byi.qs[quarter * 8 + b];
                        }
                    }
                    __syncthreads();

                    cerebra_mma::tile<16, 8> a = cerebra_mma::load_a_fragment(a_nibbles, 32);
                    cerebra_mma::tile<8, 8> b = cerebra_mma::load_b_fragment(b_quants, 32);
                    cerebra_mma::tile<16, 8> d;
                    cerebra_mma::mma(d, a, b);
                    cerebra_mma::store_d_fragment(dot_int, 8, d);
                    __syncthreads();

                    for (int idx = tid_flat; idx < 16 * 8; idx += 32) {
                        const int row_local = idx / 8, task_local = idx % 8;
                        const int row = row_dst_0 + min(row_local, rows_here - 1);
                        const cerebra_mma::block_q4_K_raw &bxi = x[(size_t)row * blocks_per_row_x + ib0];
                        const float dall = __low2float(bxi.dm);
                        const float dmin = __high2float(bxi.dm);
                        uint8_t sc, m;
                        mma_get_scale_min_k4(is, bxi.scales, sc, m);

                        const int col_local = min(col_dst_0 + task_local, count - 1);
                        const int t = (int)task_list[col_local];
                        const int input_row = (input_dim1 == 1) ? (t / topk) : t;
                        const cerebra_mma::block_q8_1_raw &byi = y[(size_t)input_row * blocks_per_col_y + ib0 * 8 + is];
                        // BUG FIX (2026-09-01, Goal-2500 Step 6 M2 real-data
                        // breakage): the dot term must be scaled by this
                        // q8_1 block's own DELTA (ds.x), matching shipped
                        // vec_dot_q4_K_q8_1_impl_vmmq's `d8[i] * (dot1 *
                        // sc[i])` -- ds.y (sum) alone, previously used here,
                        // only feeds the min-correction term. Missing the
                        // delta multiply was invisible against synthetic
                        // test data (every synthetic q8_1 block used a
                        // constant delta, so kernel and host reference
                        // agreed with each other despite both being wrong)
                        // but produced catastrophic errors against real
                        // data with varying per-block deltas (worst_rel
                        // 144970 measured against real production bytes).
                        const float q8_delta = __low2float(byi.ds);
                        const float q8_sum = __high2float(byi.ds);

                        acc[idx] += dall * (dot_int[idx] * q8_delta * (float)sc) - dmin * (q8_sum * (float)m);
                    }
                    __syncthreads();
                }
            }
        }

        for (int idx = tid_flat; idx < 16 * 8; idx += 32) {
            const int row_local = idx / 8, task_local = idx % 8;
            if (row_local >= rows_here || task_local >= tasks_here) continue;
            const int row = row_dst_0 + row_local;
            const int col_local = col_dst_0 + task_local;
            const int t = (int)task_list[col_local];
            all_outputs[(size_t)t * n + row] = acc[idx];
        }
        __syncthreads();
    }
#undef IMMQ_MMA_MAX_TASKS
}
