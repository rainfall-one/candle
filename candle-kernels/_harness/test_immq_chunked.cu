// Standalone diff harness for the chunked MMQ launch loop in
// candle-core's cuda.rs (indexed_moe_forward_fused_q8_1_input) --
// originally built for Goal-2500 Step 7.5 (2026-08-31, the token-chunked
// input_dim1 == 1 arm), extended 2026-09-04 by the cerebra
// spec-nondeterminism investigation with the two cases whose absence let
// a real corruption bug ship:
//
//   Case 1 (Step 7.5 original): input_dim1 == 1, total_tasks > 1024,
//     token-boundary chunk loop vs the task-major reference -- MATCH
//     required.
//   Case 2 (NEW): input_dim1 == topk (one activation row per TASK --
//     the down projection's real shape), total_tasks > 1024, the
//     per-task chunk loop (each chunk relaunched as batch=chunk_tasks,
//     topk=1, input_dim1=1) vs the task-major reference run at the TRUE
//     shape -- MATCH required. This is the arm whose absence from the
//     host chunk gate caused live nondeterministic generation.
//   Case 3 (NEW, negative control + fix proof): hot-expert routing skew
//     (>1024 of the tasks on ONE expert), input_dim1 == 1. First the
//     OLD buggy path -- a single UNCHUNKED tiled-MMQ launch -- which
//     MUST come back corrupt (the kernel's 1024-entry shared task list
//     truncates; dropped tasks' outputs are never written; the new
//     in-kernel tripwire also prints TRUNCATION). A harness that cannot
//     observe the bug proves nothing, so corruption here is asserted,
//     not tolerated. Then the token-chunk loop on the SAME data --
//     MATCH required (each chunk holds <= 1024 tasks total, so even a
//     100%-skewed expert stays under the per-launch cap).
//
// Ground truth throughout: indexed_moe_forward_q4k_q8_1 (task-major,
// no shared task list, no task-count limit, handles both input_dim1
// conventions natively).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#include "../src/quantized.cu"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

static const int NUM_EXPERTS = 16;
static const int N = 128;
static const int K = 2048;
static const int BATCH = 200; // batch * topk = 1600 > IMMQ_MAX_TASKS (1024)
static const int TOPK = 8;
static const int K_PADDED = K;
static const int TOTAL_TASKS = BATCH * TOPK;
static const int MAX_TASKS_PER_LAUNCH = 1024; // IMMQ_MAX_TASKS's value, host side

static size_t y_blocks_per_row() { return K_PADDED / QK8_1; }
static size_t row_bytes() { return y_blocks_per_row() * sizeof(block_q8_1); }

static void fill_weights(std::vector<unsigned char> &h_w) {
    const size_t blocks_per_expert = (size_t)N * K / QK_K;
    h_w.resize(NUM_EXPERTS * blocks_per_expert * sizeof(block_q4_K));
    for (size_t b = 0; b < NUM_EXPERTS * blocks_per_expert; ++b) {
        block_q4_K * blk = (block_q4_K *)(h_w.data() + b * sizeof(block_q4_K));
        blk->dm = __floats2half2_rn(0.01f + 0.0001f * (rand() % 100), 0.001f * (rand() % 50));
        for (int i = 0; i < K_SCALE_SIZE; ++i) blk->scales[i] = (unsigned char)(rand() % 256);
        for (int i = 0; i < QK_K / 2; ++i) blk->qs[i] = (unsigned char)(rand() % 256);
    }
}

static void fill_activations(std::vector<unsigned char> &h_y, int total_rows) {
    h_y.resize((size_t)total_rows * row_bytes());
    for (size_t b = 0; b < (size_t)total_rows * y_blocks_per_row(); ++b) {
        block_q8_1 * blk = (block_q8_1 *)(h_y.data() + b * sizeof(block_q8_1));
        float sum = 0.0f;
        for (int i = 0; i < QK8_1; ++i) {
            int q = (rand() % 255) - 127;
            blk->qs[i] = (int8_t)q;
            sum += q;
        }
        const float d = 0.01f;
        blk->ds = __floats2half2_rn(d, d * sum);
    }
}

// Task-major reference, ONE launch over the full set at the TRUE shape.
static void run_reference(const void *d_w, const void *d_y, const unsigned int *d_ids,
                          float *d_out, int input_dim1) {
    dim3 grid(N, BATCH, TOPK);
    dim3 blk(WARP_SIZE, 4, 1);
    indexed_moe_forward_q4k_q8_1<<<grid, blk>>>(d_w, d_y, d_ids, d_out,
        N, K, BATCH, TOPK, K_PADDED, input_dim1);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
}

// One tiled-MMQ launch at whatever (batch, topk, input_dim1) the caller
// hands it -- shared by the chunk loops and case 3's unchunked control.
static void launch_mmq(const void *d_w, const unsigned char *input_ptr,
                       const unsigned int *ids_ptr, float *out_ptr,
                       int batch, int topk, int input_dim1) {
    dim3 grid((N + IMMQ4_Y - 1) / IMMQ4_Y, 8, NUM_EXPERTS);
    dim3 blk(WARP_SIZE, IMMQ4_NWARPS, 1);
    indexed_mul_mat_q4_K_moe<<<grid, blk>>>(d_w, input_ptr, ids_ptr, out_ptr,
        N, K, batch, topk, K_PADDED, input_dim1);
    CK(cudaGetLastError());
}

// cuda.rs's token-boundary chunk loop (input_dim1 == 1), by hand.
static void run_chunked_token(const void *d_w, const unsigned char *d_y,
                              const unsigned int *d_ids, float *d_out) {
    const int chunk_tokens = MAX_TASKS_PER_LAUNCH / TOPK;
    int token_start = 0;
    while (token_start < BATCH) {
        const int chunk_batch = std::min(chunk_tokens, BATCH - token_start);
        const int chunk_task_start = token_start * TOPK;
        launch_mmq(d_w,
                   d_y + (size_t)token_start * row_bytes(),
                   d_ids + chunk_task_start,
                   d_out + (size_t)chunk_task_start * N,
                   chunk_batch, TOPK, 1);
        token_start += chunk_batch;
    }
    CK(cudaDeviceSynchronize());
}

// cuda.rs's NEW per-task chunk loop (input_dim1 == topk), by hand: each
// chunk relaunches as its own (batch = chunk_tasks, topk = 1,
// input_dim1 = 1) problem -- in-kernel input_row = t / 1 = t, exactly
// the per-task row this shape stores.
static void run_chunked_per_task(const void *d_w, const unsigned char *d_y,
                                 const unsigned int *d_ids, float *d_out) {
    int task_start = 0;
    while (task_start < TOTAL_TASKS) {
        const int chunk_tasks = std::min(MAX_TASKS_PER_LAUNCH, TOTAL_TASKS - task_start);
        launch_mmq(d_w,
                   d_y + (size_t)task_start * row_bytes(),
                   d_ids + task_start,
                   d_out + (size_t)task_start * N,
                   chunk_tasks, 1, 1);
        task_start += chunk_tasks;
    }
    CK(cudaDeviceSynchronize());
}

// Elementwise diff; returns the count of bad elements (rel > 2% AND
// abs > 0.05, the harness family's established thresholds).
static int diff_outputs(const std::vector<float> &ref, const std::vector<float> &got,
                        const char *label) {
    double worst_abs = 0.0, worst_rel = 0.0;
    long long first_bad = -1;
    int bad_count = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double a = ref[i], b = got[i];
        const double ad = fabs(a - b);
        const double rd = ad / (fabs(a) + 1e-6);
        if (ad > worst_abs) worst_abs = ad;
        if (rd > worst_rel) worst_rel = rd;
        if (rd > 0.02 && ad > 0.05) {
            if (first_bad < 0) first_bad = (long long)i;
            ++bad_count;
        }
    }
    printf("%s: elems=%zu worst_abs=%g worst_rel=%g bad=%d\n",
           label, ref.size(), worst_abs, worst_rel, bad_count);
    if (first_bad >= 0) {
        printf("%s: first_bad task=%lld row=%lld ref=%g got=%g\n",
               label, first_bad / N, first_bad % N, ref[first_bad], got[first_bad]);
    }
    return bad_count;
}

struct DeviceBufs {
    void *w = nullptr, *y = nullptr;
    unsigned int *ids = nullptr;
    float *out_ref = nullptr, *out_got = nullptr;
    size_t out_elems = 0;
};

static DeviceBufs upload(const std::vector<unsigned char> &h_w,
                         const std::vector<unsigned char> &h_y,
                         const std::vector<unsigned int> &h_ids) {
    DeviceBufs d;
    d.out_elems = (size_t)TOTAL_TASKS * N;
    CK(cudaMalloc(&d.w, h_w.size()));
    CK(cudaMalloc(&d.y, h_y.size()));
    CK(cudaMalloc(&d.ids, h_ids.size() * sizeof(unsigned int)));
    CK(cudaMalloc(&d.out_ref, d.out_elems * sizeof(float)));
    CK(cudaMalloc(&d.out_got, d.out_elems * sizeof(float)));
    // 0x7E fill: any output row a launch never writes stays a huge,
    // unmistakably-wrong float instead of a plausible stale value.
    CK(cudaMemset(d.out_got, 0x7E, d.out_elems * sizeof(float)));
    CK(cudaMemcpy(d.w, h_w.data(), h_w.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d.y, h_y.data(), h_y.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d.ids, h_ids.data(), h_ids.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
    return d;
}

static void download(const DeviceBufs &d, std::vector<float> &ref, std::vector<float> &got) {
    ref.resize(d.out_elems);
    got.resize(d.out_elems);
    CK(cudaMemcpy(ref.data(), d.out_ref, d.out_elems * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(got.data(), d.out_got, d.out_elems * sizeof(float), cudaMemcpyDeviceToHost));
}

static void release(DeviceBufs &d) {
    cudaFree(d.w); cudaFree(d.y); cudaFree(d.ids);
    cudaFree(d.out_ref); cudaFree(d.out_got);
}

int main() {
    srand(42);
    int failures = 0;
    std::vector<unsigned char> h_w;
    fill_weights(h_w);

    // ---- Case 1: input_dim1 == 1, uniform routing, token-chunked. ----
    {
        std::vector<unsigned char> h_y;
        fill_activations(h_y, BATCH); // one row per TOKEN
        std::vector<unsigned int> h_ids(TOTAL_TASKS);
        for (int t = 0; t < TOTAL_TASKS; ++t) h_ids[t] = rand() % NUM_EXPERTS;
        DeviceBufs d = upload(h_w, h_y, h_ids);
        run_reference(d.w, d.y, d.ids, d.out_ref, 1);
        run_chunked_token(d.w, (const unsigned char *)d.y, d.ids, d.out_got);
        std::vector<float> ref, got;
        download(d, ref, got);
        if (diff_outputs(ref, got, "case1 dim1=1 token-chunked") != 0) {
            printf("case1: FAIL (must MATCH)\n");
            ++failures;
        } else {
            printf("case1: MATCH\n");
        }
        release(d);
    }

    // ---- Case 2: input_dim1 == topk (per-task rows), per-task chunked. ----
    {
        std::vector<unsigned char> h_y;
        fill_activations(h_y, TOTAL_TASKS); // one row per TASK
        std::vector<unsigned int> h_ids(TOTAL_TASKS);
        for (int t = 0; t < TOTAL_TASKS; ++t) h_ids[t] = rand() % NUM_EXPERTS;
        DeviceBufs d = upload(h_w, h_y, h_ids);
        run_reference(d.w, d.y, d.ids, d.out_ref, TOPK);
        run_chunked_per_task(d.w, (const unsigned char *)d.y, d.ids, d.out_got);
        std::vector<float> ref, got;
        download(d, ref, got);
        if (diff_outputs(ref, got, "case2 dim1=topk per-task-chunked") != 0) {
            printf("case2: FAIL (must MATCH)\n");
            ++failures;
        } else {
            printf("case2: MATCH\n");
        }
        release(d);
    }

    // ---- Case 3: hot-expert skew -- unchunked MUST corrupt, chunked MUST match. ----
    {
        std::vector<unsigned char> h_y;
        fill_activations(h_y, BATCH); // one row per TOKEN (dim1 == 1 shape)
        std::vector<unsigned int> h_ids(TOTAL_TASKS);
        // 1,400 of 1,600 tasks on expert 3 -- past the kernel's
        // 1,024-entry shared task list; the remainder spread uniformly.
        for (int t = 0; t < TOTAL_TASKS; ++t) {
            h_ids[t] = (t < 1400) ? 3u : (unsigned)(rand() % NUM_EXPERTS);
        }
        DeviceBufs d = upload(h_w, h_y, h_ids);
        run_reference(d.w, d.y, d.ids, d.out_ref, 1);

        // 3a: the OLD buggy path -- one unchunked launch at the full
        // 1,600-task shape. The hot expert truncates (expect the
        // in-kernel TRUNCATION printf) and dropped tasks keep the 0x7E
        // fill. Corruption is ASSERTED: if this ever comes back clean,
        // the harness has lost its ability to see the bug.
        launch_mmq(d.w, (const unsigned char *)d.y, d.ids, d.out_got, BATCH, TOPK, 1);
        CK(cudaDeviceSynchronize());
        std::vector<float> ref, got;
        download(d, ref, got);
        const int unchunked_bad = diff_outputs(ref, got, "case3a hot-expert UNCHUNKED (bug demo)");
        if (unchunked_bad == 0) {
            printf("case3a: FAIL (unchunked hot-expert launch must corrupt -- harness can no longer observe the bug)\n");
            ++failures;
        } else {
            printf("case3a: corrupt as expected (bad=%d)\n", unchunked_bad);
        }

        // 3b: the fixed path -- token-chunked on the same data.
        CK(cudaMemset(d.out_got, 0x7E, d.out_elems * sizeof(float)));
        run_chunked_token(d.w, (const unsigned char *)d.y, d.ids, d.out_got);
        download(d, ref, got);
        if (diff_outputs(ref, got, "case3b hot-expert token-chunked") != 0) {
            printf("case3b: FAIL (must MATCH)\n");
            ++failures;
        } else {
            printf("case3b: MATCH\n");
        }
        release(d);
    }

    if (failures != 0) {
        printf("FAILURES=%d\n", failures);
        return 1;
    }
    printf("ALL CASES PASS\n");
    return 0;
}
