use super::{GgmlDType, QStorage};
use crate::quantized::k_quants::GgmlType;
use crate::{backend::BackendDevice, cuda_backend::WrapErr};
use crate::{builder_arg as barg, CudaDevice, CudaStorage, Result, Tensor};
use half::{bf16, f16};

use cudarc::driver::{CudaSlice, CudaView, PushKernelArg};

#[derive(Clone, Debug)]
struct PaddedCudaSlice {
    inner: CudaSlice<u8>,
    len: usize,
}

#[derive(Clone, Debug)]
pub struct QCudaStorage {
    data: PaddedCudaSlice,
    dtype: GgmlDType,
    device: CudaDevice,
}

static FORCE_DMMV: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_force_dmmv(f: bool) {
    FORCE_DMMV.store(f, std::sync::atomic::Ordering::Relaxed)
}

pub const WARP_SIZE: usize = 32;
pub const MMQ_X_Q4_0_AMPERE: usize = 4;
pub const MMQ_Y_Q4_0_AMPERE: usize = 32;
pub const NWARPS_Q4_0_AMPERE: usize = 4;
pub const GGML_CUDA_MMV_X: usize = 32;
pub const GGML_CUDA_MMV_Y: usize = 1;
pub const CUDA_QUANTIZE_BLOCK_SIZE: usize = 256;
pub const CUDA_DEQUANTIZE_BLOCK_SIZE: usize = 256;
pub const MATRIX_ROW_PADDING: usize = 512;

fn ceil_div(p: usize, q: usize) -> usize {
    p.div_ceil(q)
}

fn pad(p: usize, q: usize) -> usize {
    ceil_div(p, q) * q
}

/// Task-count floor for the tiled dp4a MMQ tier's DEFAULT dispatch
/// (Goal-2500 enablement, 2026-09-05): real hardware evidence shows this
/// tier wins at prefill scale (~20k tasks, -31 to -45% TTFT/MoE-time)
/// but REGRESSES at decode widths (64-256 tasks, -7.9% to -17.6% decode
/// tok/s) -- the per-launch task-list scan+sort cost this tier's own
/// `moe_grouped_build_task_list`-family helper pays doesn't amortize
/// below some real task count. 1,024 sits above every decode width
/// measured (<=256 tasks even at N=32) and below every prefill chunk
/// (4,096 tasks) -- comfortably inside the gap between the two
/// evidenced regimes, not derived from a sweep of the boundary itself.
///
fn moe_grouped_min_tasks() -> usize {
    moe_grouped_min_tasks_from_raw(std::env::var("CEREBRA_MOE_GROUPED_MIN_TASKS").ok().as_deref())
}

/// Pure half of [`moe_grouped_min_tasks`] -- unit-testable without
/// mutating the process environment (this file's CUDA tests run
/// against real hardware; splitting the parse out keeps this predicate
/// testable on any machine, no device required).
///
/// # Errors
/// Never -- an unparsable or zero override falls back to the default
/// rather than propagating, matching this crate's own env-override
/// conventions elsewhere (an operator typo should not crash inference).
fn moe_grouped_min_tasks_from_raw(raw: Option<&str>) -> usize {
    raw.and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v >= 1)
        .unwrap_or(1024)
}

/// Resolves `CEREBRA_MOE_GROUPED`'s per-dtype precedence for the tiled
/// MMQ MoE tier -- pure and unit-testable (no CUDA, no env access),
/// factored out of the three near-identical `match w_dtype` arms in
/// [`indexed_moe_forward_fused_q8_1_input`] rather than duplicated per
/// dtype.
///
/// `env` is the raw `CEREBRA_MOE_GROUPED` value; `task_count_gate` is
/// [`moe_grouped_min_tasks`]'s own threshold check for this call;
/// `this_dtype_key` is this dtype's own bisect string (e.g. `"q4k"`);
/// `other_dtype_keys` are the sibling hot dtypes' bisect strings, which
/// this dtype must be EXCLUDED for (correctness-bisection semantics: an
/// operator forcing one dtype on is deliberately ruling the others
/// out, not leaving them at their own default).
///
/// Precedence, in order: `""` (unset) defers to `task_count_gate` --
/// the production default, no env needed. `"0"` is an explicit
/// force-OFF (added after this enablement's supervisor review found it
/// silently fell through to the SAME behavior as unset, which is not
/// what an operator setting `"0"` would expect) -- task-major
/// regardless of task count. `"1"` force-ON for every hot dtype at any
/// task count. `this_dtype_key` force-ON for just this dtype.
/// `other_dtype_keys` force-OFF (bisection exclusion). Anything else
/// unrecognized falls back to `task_count_gate`, same as unset -- an
/// operator typo should not silently disable the tier either.
fn mmq_dtype_eligible(env: &str, task_count_gate: bool, this_dtype_key: &str, other_dtype_keys: &[&str]) -> bool {
    match env {
        "" => task_count_gate,
        "0" => false,
        "1" => true,
        key if key == this_dtype_key => true,
        key if other_dtype_keys.contains(&key) => false,
        _ => task_count_gate,
    }
}

fn quantize_q8_1(
    src: &CudaView<f32>,
    dst: &mut CudaSlice<u8>,
    k: usize,
    ky: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let kx_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks = ceil_div(kx_padded, CUDA_QUANTIZE_BLOCK_SIZE);

    let total_rows = ky;
    // Get Q8_1 metadata.
    let q8_1_block_size = GgmlDType::Q8_1.block_size();
    let q8_1_type_size = GgmlDType::Q8_1.type_size();

    // Calculate the size of the output buffer in bytes.
    let num_blocks_per_row = kx_padded / q8_1_block_size;
    let dst_row_size_bytes = num_blocks_per_row * q8_1_type_size;

    const CHUNK_SIZE: usize = 65535; // gridDim.y limit
    let func = dev.get_or_load_func("quantize_q8_1", &candle_kernels::QUANTIZED)?;

    let mut rows_processed = 0;
    while rows_processed < total_rows {
        // --- calculate the number of rows for this chunk ---
        let remaining_rows = total_rows - rows_processed;
        // This is our gridDim.y, now <= 65535
        let rows_in_chunk = std::cmp::min(CHUNK_SIZE, remaining_rows);

        // --- slice the source (f32) tensor by elements ---
        let src_start_elem = rows_processed * k;
        let src_num_elems = rows_in_chunk * k;
        let src_chunk = src.slice(src_start_elem..(src_start_elem + src_num_elems));

        // --- slice the destination (u8) tensor by bytes ---
        let dst_start_byte = rows_processed * dst_row_size_bytes;
        let dst_num_bytes = rows_in_chunk * dst_row_size_bytes;
        let dst_chunk = dst.slice(dst_start_byte..(dst_start_byte + dst_num_bytes));

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_blocks as u32, rows_in_chunk as u32, 1),
            block_dim: (CUDA_QUANTIZE_BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = func.builder();
        builder.arg(&src_chunk);
        builder.arg(&dst_chunk);
        barg!(builder, k as i32, kx_padded as i32);
        unsafe { builder.launch(cfg) }.w()?;

        rows_processed += rows_in_chunk;
    }

    Ok(())
}

// BF16-input sibling of `quantize_q8_1` -- identical chunking and
// launch geometry, dispatching to the `quantize_q8_1_bf16` kernel so a
// BF16 activation row quantizes directly (no separate BF16->F32 cast
// pass; that cast was one graph node per MoE layer per token in the
// consuming project's captured decode graph).
fn quantize_q8_1_from_bf16(
    src: &CudaView<bf16>,
    dst: &mut CudaSlice<u8>,
    k: usize,
    ky: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let kx_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks = ceil_div(kx_padded, CUDA_QUANTIZE_BLOCK_SIZE);

    let total_rows = ky;
    let q8_1_block_size = GgmlDType::Q8_1.block_size();
    let q8_1_type_size = GgmlDType::Q8_1.type_size();
    let num_blocks_per_row = kx_padded / q8_1_block_size;
    let dst_row_size_bytes = num_blocks_per_row * q8_1_type_size;

    const CHUNK_SIZE: usize = 65535; // gridDim.y limit
    let func = dev.get_or_load_func("quantize_q8_1_bf16", &candle_kernels::QUANTIZED)?;

    let mut rows_processed = 0;
    while rows_processed < total_rows {
        let remaining_rows = total_rows - rows_processed;
        let rows_in_chunk = std::cmp::min(CHUNK_SIZE, remaining_rows);

        let src_start_elem = rows_processed * k;
        let src_num_elems = rows_in_chunk * k;
        let src_chunk = src.slice(src_start_elem..(src_start_elem + src_num_elems));

        let dst_start_byte = rows_processed * dst_row_size_bytes;
        let dst_num_bytes = rows_in_chunk * dst_row_size_bytes;
        let dst_chunk = dst.slice(dst_start_byte..(dst_start_byte + dst_num_bytes));

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_blocks as u32, rows_in_chunk as u32, 1),
            block_dim: (CUDA_QUANTIZE_BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = func.builder();
        builder.arg(&src_chunk);
        builder.arg(&dst_chunk);
        barg!(builder, k as i32, kx_padded as i32);
        unsafe { builder.launch(cfg) }.w()?;

        rows_processed += rows_in_chunk;
    }

    Ok(())
}

/// The activation-input storage `indexed_moe_forward` quantizes on the
/// fly -- F32 (the historical path) or BF16 (quantized directly, no
/// widening cast pass). Views, not whole slices: the caller slices at
/// the input layout's start offset, which the historical path silently
/// ignored (latent -- its F32 inputs were always fresh offset-0 cast
/// outputs; a BF16 narrow view exposed it as garbage activations,
/// caught live 2026-08-28).
enum IndexedMoeInput<'a> {
    F32(CudaView<'a, f32>),
    Bf16(CudaView<'a, bf16>),
    /// Activation rows ALREADY quantized to Q8_1 blocks by the caller
    /// (e.g. a producer kernel fusing its epilogue with quantization) --
    /// the internal quantize pass is skipped entirely. The U8 tensor's
    /// last dim is `k_padded / 32 * 36` bytes per row, exactly the
    /// buffer this function would otherwise build.
    Q8_1(CudaView<'a, u8>),
}

fn dequantize_f32(
    data: &PaddedCudaSlice,
    dtype: GgmlDType,
    elem_count: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let nb = elem_count.div_ceil(256);
    let (kernel_name, is_k, block_dim, num_blocks) = match dtype {
        GgmlDType::Q4_0 => ("dequantize_block_q4_0_f32", false, 32, nb),
        GgmlDType::Q4_1 => ("dequantize_block_q4_1_f32", false, 32, nb),
        GgmlDType::Q5_0 => (
            "dequantize_block_q5_0_f32",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q5_1 => (
            "dequantize_block_q5_1_f32",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q8_0 => ("dequantize_block_q8_0_f32", false, 32, nb),
        GgmlDType::Q2K => ("dequantize_block_q2_K_f32", true, 64, nb),
        GgmlDType::Q3K => ("dequantize_block_q3_K_f32", true, 64, nb),
        GgmlDType::Q4K => ("dequantize_block_q4_K_f32", true, 32, nb),
        GgmlDType::Q5K => ("dequantize_block_q5_K_f32", true, 64, nb),
        GgmlDType::Q6K => ("dequantize_block_q6_K_f32", true, 64, nb),
        GgmlDType::Q8K => ("dequantize_block_q8_K_f32", true, 32, nb),
        _ => crate::bail!("unsupported dtype for dequantize {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &candle_kernels::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f32>(elem_count)? };
    // See e.g.
    // https://github.com/ggerganov/llama.cpp/blob/cbbd1efa06f8c09f9dff58ff9d9af509cc4c152b/ggml-cuda.cu#L7270
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (block_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    if is_k {
        let mut builder = func.builder();
        builder.arg(&data.inner);
        builder.arg(&dst);
        unsafe { builder.launch(cfg) }.w()?;
    } else {
        let nb32 = match dtype {
            GgmlDType::Q5_0 | GgmlDType::Q5_1 => elem_count,
            _ => elem_count / 32,
        };
        let mut builder = func.builder();
        builder.arg(&data.inner);
        builder.arg(&dst);
        barg!(builder, nb32 as i32);
        unsafe { builder.launch(cfg) }.w()?;
    }
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

fn dequantize_f16(
    data: &PaddedCudaSlice,
    dtype: GgmlDType,
    elem_count: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let nb = elem_count.div_ceil(256);
    let (kernel_name, is_k, block_dim, num_blocks) = match dtype {
        GgmlDType::Q4_0 => ("dequantize_block_q4_0_f16", false, 32, nb),
        GgmlDType::Q4_1 => ("dequantize_block_q4_1_f16", false, 32, nb),
        GgmlDType::Q5_0 => (
            "dequantize_block_q5_0_f16",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q5_1 => (
            "dequantize_block_q5_1_f16",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q8_0 => ("dequantize_block_q8_0_f16", false, 32, nb),
        GgmlDType::Q2K => ("dequantize_block_q2_K_f16", true, 64, nb),
        GgmlDType::Q3K => ("dequantize_block_q3_K_f16", true, 64, nb),
        GgmlDType::Q4K => ("dequantize_block_q4_K_f16", true, 32, nb),
        GgmlDType::Q5K => ("dequantize_block_q5_K_f16", true, 64, nb),
        GgmlDType::Q6K => ("dequantize_block_q6_K_f16", true, 64, nb),
        GgmlDType::Q8K => ("dequantize_block_q8_K_f16", true, 32, nb),
        _ => crate::bail!("unsupported dtype for dequantize {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &candle_kernels::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f16>(elem_count)? };
    // See e.g.
    // https://github.com/ggerganov/llama.cpp/blob/cbbd1efa06f8c09f9dff58ff9d9af509cc4c152b/ggml-cuda.cu#L7270
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (block_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    if is_k {
        let mut builder = func.builder();
        builder.arg(&data.inner);
        builder.arg(&dst);
        unsafe { builder.launch(cfg) }.w()?;
    } else {
        let nb32 = match dtype {
            GgmlDType::Q5_0 | GgmlDType::Q5_1 => elem_count,
            _ => elem_count / 32,
        };
        let mut builder = func.builder();
        builder.arg(&data.inner);
        builder.arg(&dst);
        barg!(builder, nb32 as i32);
        unsafe { builder.launch(cfg) }.w()?;
    }
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

fn dequantize_mul_mat_vec(
    data: &PaddedCudaSlice,
    y: &CudaView<f32>,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let data_elems = data.len / dtype.type_size() * dtype.block_size();
    if data_elems < ncols * nrows {
        crate::bail!("unexpected data size {}, ncols {ncols} {nrows}", data_elems)
    }
    if y.len() != ncols {
        crate::bail!("unexpected y size {}, ncols {ncols} {nrows}", y.len())
    }
    let kernel_name = match dtype {
        GgmlDType::Q4_0 => "dequantize_mul_mat_vec_q4_0_cuda",
        GgmlDType::Q4_1 => "dequantize_mul_mat_vec_q4_1_cuda",
        GgmlDType::Q5_0 => "dequantize_mul_mat_vec_q5_0_cuda",
        GgmlDType::Q5_1 => "dequantize_mul_mat_vec_q5_1_cuda",
        GgmlDType::Q8_0 => "dequantize_mul_mat_vec_q8_0_cuda",
        GgmlDType::Q2K => "dequantize_mul_mat_vec_q2_k",
        GgmlDType::Q3K => "dequantize_mul_mat_vec_q3_k",
        GgmlDType::Q4K => "dequantize_mul_mat_vec_q4_k",
        GgmlDType::Q5K => "dequantize_mul_mat_vec_q5_k",
        GgmlDType::Q6K => "dequantize_mul_mat_vec_q6_k",
        _ => crate::bail!("unsupported dtype for quantized matmul {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &candle_kernels::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f32>(nrows)? };
    let block_num_y = ceil_div(nrows, GGML_CUDA_MMV_Y);
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (block_num_y as u32, 1, 1),
        block_dim: (WARP_SIZE as u32, GGML_CUDA_MMV_Y as u32, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(&data.inner);
    builder.arg(y);
    builder.arg(&dst);
    barg!(builder, ncols as i32, nrows as i32);
    unsafe { builder.launch(cfg) }.w()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

fn mul_mat_vec_via_q8_1(
    data: &PaddedCudaSlice,
    y: &CudaView<f32>,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    b_size: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let data_elems = data.len / dtype.type_size() * dtype.block_size();
    if data_elems < ncols * nrows {
        crate::bail!("unexpected data size {}, ncols {ncols} {nrows}", data_elems)
    }
    if y.len() != ncols * b_size {
        crate::bail!("unexpected y size {}, ncols {ncols} {nrows}", y.len())
    }
    if b_size == 0 || b_size > 8 {
        crate::bail!("only bsize between 1 and 8 are supported, got {b_size}")
    }
    // Start by quantizing y
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    let y_size_in_bytes =
        b_size * ncols_padded * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
    // Zero-init ONLY when a padded tail exists: `quantize_q8_1` writes
    // exactly `ncols` worth of blocks per row and the matmul kernel
    // reads the padded width, so a tail must be zeroed -- but when
    // `ncols` is already a multiple of MATRIX_ROW_PADDING there is no
    // tail and the memset is pure overhead (inside a CUDA graph capture
    // it also records a memset node per call -- audited live at
    // 240/decode-step on rainfall-one's Cerebra, 2026-08-28).
    // SAFETY (alloc branch): no padded tail exists, so `quantize_q8_1`
    // below overwrites every byte before any read.
    let mut y_q8_1 = if ncols_padded == ncols {
        unsafe { dev.alloc::<u8>(y_size_in_bytes) }?
    } else {
        dev.alloc_zeros::<u8>(y_size_in_bytes)?
    };
    quantize_q8_1(y, &mut y_q8_1, ncols, b_size, dev)?;

    let kernel_name = match dtype {
        GgmlDType::Q4_0 => "mul_mat_vec_q4_0_q8_1_cuda",
        GgmlDType::Q4_1 => "mul_mat_vec_q4_1_q8_1_cuda",
        GgmlDType::Q5_0 => "mul_mat_vec_q5_0_q8_1_cuda",
        GgmlDType::Q5_1 => "mul_mat_vec_q5_1_q8_1_cuda",
        GgmlDType::Q8_0 => "mul_mat_vec_q8_0_q8_1_cuda",
        GgmlDType::Q2K => "mul_mat_vec_q2_K_q8_1_cuda",
        GgmlDType::Q3K => "mul_mat_vec_q3_K_q8_1_cuda",
        GgmlDType::Q4K => "mul_mat_vec_q4_K_q8_1_cuda",
        GgmlDType::Q5K => "mul_mat_vec_q5_K_q8_1_cuda",
        GgmlDType::Q6K => "mul_mat_vec_q6_K_q8_1_cuda",
        _ => crate::bail!("unsupported dtype for quantized matmul {dtype:?}"),
    };
    let kernel_name = format!("{kernel_name}{b_size}");
    let func = dev.get_or_load_func(&kernel_name, &candle_kernels::QUANTIZED)?;
    // SAFETY: every `mul_mat_vec_*_q8_1_cuda` kernel writes each of its
    // `nrows * b_size` output elements exactly once (per-row guard then
    // unconditional store) -- no element is read-modify-written, so the
    // zero-init this replaced was pure overhead (and one memset graph
    // node per call inside a capture).
    let dst = unsafe { dev.alloc::<f32>(nrows * b_size) }?;
    // https://github.com/ggerganov/llama.cpp/blob/facb8b56f8fd3bb10a693bf0943ae9d69d0828ef/ggml-cuda/mmvq.cu#L98
    let (nblocks, nwarps) = match b_size {
        1 => (nrows as u32, 4),
        2..=4 => ((nrows as u32).div_ceil(2), 4),
        5..=8 => ((nrows as u32).div_ceil(2), 2),
        _ => crate::bail!("unexpected bsize {b_size}"),
    };
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (nblocks, 1, 1),
        block_dim: (WARP_SIZE as u32, nwarps, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(&data.inner);
    builder.arg(&y_q8_1);
    builder.arg(&dst);
    barg!(
        builder,
        /* ncols_x */ ncols as i32,
        /* nrows_x */ nrows as i32,
        /* nrows_y */ ncols_padded as i32,
        /* nrows_dst */ nrows as i32
    );
    unsafe { builder.launch(cfg) }.w()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

#[allow(clippy::too_many_arguments)]
fn mul_mat_via_q8_1(
    data: &PaddedCudaSlice,
    y: &CudaView<f32>,
    dtype: GgmlDType,
    x_rows: usize,
    x_cols: usize,
    y_rows: usize,
    y_cols: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let data_elems = data.len / dtype.type_size() * dtype.block_size();
    if data_elems < x_rows * x_cols {
        crate::bail!("unexpected lhs size {}, {x_rows} {x_cols}", data_elems)
    }
    if y.len() != y_rows * y_cols {
        crate::bail!("unexpected y size {}, {y_rows} {y_cols}", y.len())
    }
    if x_cols != y_rows {
        crate::bail!("unexpected x/y size {x_rows} {x_cols} {y_rows} {y_cols}")
    }
    let k = x_cols;
    // Start by quantizing y
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let y_size_in_bytes =
        k_padded * y_cols * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
    // Padded-tail-only zero-init -- see mul_mat_vec_via_q8_1's own
    // comment for the reasoning (identical situation).
    // SAFETY (alloc branch): no padded tail, quantize_q8_1 overwrites
    // every byte before any read.
    let mut y_q8_1 = if k_padded == k {
        unsafe { dev.alloc::<u8>(y_size_in_bytes) }?
    } else {
        dev.alloc_zeros::<u8>(y_size_in_bytes)?
    };
    quantize_q8_1(y, &mut y_q8_1, k, y_cols, dev)?;

    let (kernel_name, mmq_x, mmq_y) = match dtype {
        GgmlDType::Q4_0 => ("mul_mat_q4_0", 64, 128),
        GgmlDType::Q4_1 => ("mul_mat_q4_1", 64, 128),
        GgmlDType::Q5_0 => ("mul_mat_q5_0", 128, 64),
        GgmlDType::Q5_1 => ("mul_mat_q5_1", 128, 64),
        GgmlDType::Q8_0 => ("mul_mat_q8_0", 128, 64),
        GgmlDType::Q2K => ("mul_mat_q2_K", 64, 128),
        GgmlDType::Q3K => ("mul_mat_q3_K", 128, 128),
        GgmlDType::Q4K => ("mul_mat_q4_K", 64, 128),
        GgmlDType::Q5K => ("mul_mat_q5_K", 64, 128),
        GgmlDType::Q6K => ("mul_mat_q6_K", 64, 64),
        _ => crate::bail!("unsupported dtype for quantized matmul {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &candle_kernels::QUANTIZED)?;
    // SAFETY: the mul_mat_q* (MMQ) kernels write every in-bounds dst
    // element exactly once (bounds-guarded tiles, final store of the
    // fully-reduced sum -- never read-modify-write), so zero-init was
    // pure overhead.
    let dst = unsafe { dev.alloc::<f32>(x_rows * y_cols) }?;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (
            ceil_div(x_rows, mmq_y) as u32,
            ceil_div(y_cols, mmq_x) as u32,
            1,
        ),
        block_dim: (WARP_SIZE as u32, 4, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(/* vx */ &data.inner);
    builder.arg(/* vy */ &y_q8_1);
    builder.arg(/* dst */ &dst);
    barg!(
        builder,
        /* ncols_x */ x_cols as i32,
        /* nrows_x */ x_rows as i32,
        /* ncols_y */ y_cols as i32,
        /* nrows_y */ k_padded as i32,
        /* nrows_dst */ x_rows as i32
    );
    unsafe { builder.launch(cfg) }.w()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

// Goal-2500 Step 6 alias-bug fix (2026-09-01): a documented Rainfall-fork
// driver defect (see `CUDARC_DISABLE_ASYNC_ALLOC`'s own comment,
// driver/safe/core.rs) makes `cuMemAllocAsync` unreliable on this
// hardware/driver/passthrough combination -- a freshly-cycled pool buffer
// that a KERNEL scatter-writes into (this MoE forward's `out`, formerly a
// fresh `unsafe { dev.alloc }` every call) can silently receive wrong
// data. Root-caused via the mma tensor-core tier's rigor (byte-level
// checksums, offline replay, launch-time instrumentation); dp4a is
// exposed to the identical defect (nothing about it is mma-specific) but
// has not corrupted in practice -- evidence the defect is triggered by a
// kernel's OWN scatter-write into a freshly-cycled buffer specifically,
// not by generic device memory traffic against the same pool.
//
// FIX: a persistent, per-(device,stream) workspace `Tensor`, grown
// on-demand and reused across every call -- `out`'s kernel write target
// is stable device memory that (after warmup) is never freshly cycled
// through the driver's pool at all, sidestepping the defect's window
// entirely rather than depending on the `CUDARC_DISABLE_ASYNC_ALLOC`
// escape hatch (benchmarked separately: that blanket fix OOMs the whole
// engine at real concurrency, N=128, because `cuMemAlloc` doesn't pool --
// not viable here).
//
// Zero-alloc sharing works because candle's OWN `Arc<RwLock<Storage>>`
// (at the `Tensor` layer, not `CudaStorage`/`CudaSlice`, which own their
// memory directly and have no non-owning view variant) does the Arc
// bookkeeping: this function mutates the workspace tensor's storage in
// place via crate-internal accessors, then returns `workspace.narrow(0,
// 0, outsize)` -- a normal, PUBLIC Tensor op producing a view that
// shares the SAME Arc, with no ownership hand-rolling and no risk of a
// double-free or premature-free (there is exactly one owner, the static
// map, and every returned Tensor is just another ref-counted handle to
// it).
//
// KEYED BY (DeviceId, stream pointer), not a single process-wide
// instance: a second CUDA device, a second engine instance, or any
// caller on a different stream must get its OWN workspace entry --
// sharing across streams would be the exact cross-stream aliasing this
// whole investigation exists to eliminate. The Mutex only serializes
// host-side lookup/replace; safety comes from each entry being used by
// exactly one stream, not from the lock.
//
// SIZED BY GROWTH, not a hardcoded ceiling: this is a general candle-core
// function, not something that may bake in one model's geometry (a
// standing rule -- never tune a general library function to the observed
// instance). Each call computes its demanded size; if the keyed entry is
// absent or too small, a larger `Tensor` replaces it. Growth is safe
// because of Arc semantics alone: any in-flight narrowed view (e.g. a
// deferred `pending_moe`) holds its own ref to the OLD tensor, which
// stays alive via that ref until its last consumer drops it, regardless
// of the map moving on to a new one. Steady state (the overwhelming
// majority of calls, once every distinct shape class has been seen once)
// performs zero allocation.
//
// TEARDOWN: a workspace entry holds device memory until its last Arc ref
// (map entry + any outstanding views) drops -- for the live server this
// is effectively "until process exit," which is fine (matches every
// other piece of persistent engine state). Test code that creates many
// short-lived `CudaDevice`s should expect this map to accumulate one
// entry per distinct (device,stream) it touches; the growth-replace path
// is also how a test harness recovers if that ever matters (a smaller
// re-keyed workspace is not created, entries are only ever grown or
// reused, never shrunk).
//
// ALIAS-BUG CORRECTION (2026-09-01): this section originally claimed the
// three cases below were exhaustive -- "single outstanding view" was an
// unstated assumption, not something the three traced cases actually
// established. A FOURTH real call pattern was found that breaks it:
// `RoutedProjections::Separate` (quantized_experts.rs) issues TWO
// `indexed_moe_forward` calls (gate, then up) BEFORE consuming either --
// both returned views alias the SAME `out` buffer, so the second call's
// kernel launch overwrote the first view's memory before it was ever
// read (silent SwiGLU corruption on this non-default path). Fixed by
// making `out` a small SLOT RING (`out_slots` below): each acquisition
// picks the first slot whose storage is not still aliased by a live
// caller-held view (`Tensor::storage_strong_count() <= 1`), growing that
// slot if needed, or appending a new slot if every existing one is still
// aliased. This closes the Separate-path bug AND any future multi-call-
// before-consume pattern by construction -- no per-call-site special
// casing, no dependence on enumerating every caller. The three cases
// below remain true and are why the FUSED path (and any single-call
// site) always finds slot 0 unaliased and never grows past one slot in
// steady state -- they are no longer read as an exhaustiveness claim.
//
// SAFETY OF REUSE (why cross-call/cross-layer sharing of one workspace
// never races on the FUSED/single-view path -- every consumer of a
// workspace-derived view is stream-ordered strictly AFTER the write that
// produced it, on the SAME stream, so the NEXT write reusing that slot's
// memory is always safe by CUDA's single-stream execution-order
// guarantee, never by timing):
//   1. Within one MoE layer, gate_up's kernel writes the workspace, its
//      output is fully consumed (activation + re-quantize) by candle ops
//      BEFORE down's own call reuses the same workspace (moe_ffn.rs's
//      quantized-path glue, sequential ops on one stream).
//   2. Across layers, the decoder defers each layer's MoE residual into
//      the NEXT layer via `pending_moe: Option<Tensor>` (model.rs) --
//      a genuine cross-layer view lifetime, exactly what this design's
//      Arc-sharing must get right. Traced precisely:
//      `residual_rmsnorm(x, pending_delta, ...)` at the TOP of
//      `TransformerLayer::forward_decode_fused` (layer.rs:134-176)
//      consumes the previous layer's deferred `moe_out` (a workspace
//      view) strictly BEFORE that same call's OWN `self.moe.forward_with_q8`
//      (the bottom, step 4) produces a NEW `moe_out` that reuses/regrows
//      the workspace. Sequential kernel launches, one stream -- read
//      completes before reuse, holds at cross-layer granularity by the
//      same argument as the within-layer case.
//   3. The ONE actual multi-stream mechanism in this codebase
//      (`CudaDevice::fork_side_branch`, single call site moe_ffn.rs:329)
//      forks the SHARED expert's front half onto a side stream,
//      capture-time only -- it provably never touches this workspace
//      (disjoint buffers: shared-expert and routed-expert weights are
//      separate objects with separate forward methods; the fork is
//      paused before routed experts run and joined before the combine
//      reads both branches). This workspace must NEVER be handed to that
//      side-branch code path; it structurally cannot reach it today. If
//      a future optimization ever widens the side-branch window to cover
//      routed-expert work, this is exactly the assumption that breaks --
//      the runtime stream-identity assert is the tripwire for that, not
//      a substitute for re-auditing this comment then.
//
// RUNTIME TRIPWIRE: every call asserts the CURRENT dispatch's stream
// matches the workspace entry's recorded stream (redundant with the
// keying above under normal operation, but keying only prevents a NEW
// entry from being shared across streams -- it does not catch a bug that
// somehow reuses an existing key's entry from the wrong stream). This
// guards eager and capture-recording paths; it does NOT run during
// CUDA-graph REPLAY (replay bypasses host code entirely) -- acceptable,
// since replay preserves the captured ordering by construction and the
// risky moments are exactly the eager/capture-time dispatches where the
// assert DOES run.
/// One `out` buffer in a workspace entry's slot ring, with its own
/// independently-tracked capacity -- slots grow independently since a
/// caller pattern that keeps one view alive longer (e.g. `gate_raw`
/// held across the `up` call) shouldn't force every slot to the max
/// size ever requested of any of them.
struct MoeOutSlot {
    tensor: Tensor,
    capacity: usize, // f32 elements
}

struct MoeWorkspaceEntry {
    out_slots: Vec<MoeOutSlot>,
    input_quant: CudaSlice<u8>,
    input_quant_capacity: usize, // bytes
    stream_ptr: usize,
}

type MoeWorkspaceKey = (crate::cuda_backend::DeviceId, usize);

static MOE_WORKSPACE: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<MoeWorkspaceKey, MoeWorkspaceEntry>>> =
    std::sync::OnceLock::new();

fn moe_workspace_key(dev: &CudaDevice) -> MoeWorkspaceKey {
    let stream_ptr = std::sync::Arc::as_ptr(&dev.cuda_stream()) as usize;
    (dev.id(), stream_ptr)
}

/// Ensures a workspace entry exists for `dev`'s (device, stream) key,
/// with an out-slot ready for `out_elems` f32 output elements and
/// `input_bytes` bytes of Q8_1-quantized input scratch.
///
/// Alias-safe slot selection (2026-09-01, see the alias-bug correction
/// above `MoeWorkspaceEntry`): scans `out_slots` for the first slot whose
/// storage is not still aliased by a live caller-held view
/// (`Tensor::storage_strong_count() <= 1` -- the map's own entry is the
/// only reference), growing that slot in place if it's too small. If
/// EVERY existing slot is still aliased, appends a new slot sized to
/// `out_elems`. Slots are never removed once allocated (same
/// never-shrink policy as capacity growth) -- a caller pattern that
/// needs N concurrent live views steady-states at N slots after its
/// first occurrence, zero allocation on every call after that.
///
/// Returns the entry's key plus the chosen slot's index, so the caller
/// can look both back up under the same lock scope needed for the
/// actual kernel dispatch.
fn moe_workspace_ensure(dev: &CudaDevice, out_elems: usize, input_bytes: usize) -> Result<(MoeWorkspaceKey, usize)> {
    let key = moe_workspace_key(dev);
    let map_lock = MOE_WORKSPACE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let mut map = map_lock.lock().expect("moe workspace mutex poisoned");

    if !map.contains_key(&key) {
        let input_quant = unsafe { dev.alloc::<u8>(input_bytes) }?;
        map.insert(
            key,
            MoeWorkspaceEntry { out_slots: Vec::new(), input_quant, input_quant_capacity: input_bytes, stream_ptr: key.1 },
        );
    }
    let entry = map.get_mut(&key).expect("just inserted or already present");

    if entry.input_quant_capacity < input_bytes {
        entry.input_quant = unsafe { dev.alloc::<u8>(input_bytes) }?;
        entry.input_quant_capacity = input_bytes;
    }

    let unaliased_slot = entry.out_slots.iter().position(|slot| slot.tensor.storage_strong_count() <= 1);
    let slot_idx = match unaliased_slot {
        Some(idx) => {
            if entry.out_slots[idx].capacity < out_elems {
                entry.out_slots[idx].tensor = Tensor::zeros(out_elems, crate::DType::F32, &crate::Device::Cuda(dev.clone()))?;
                entry.out_slots[idx].capacity = out_elems;
            }
            idx
        }
        None => {
            let tensor = Tensor::zeros(out_elems, crate::DType::F32, &crate::Device::Cuda(dev.clone()))?;
            entry.out_slots.push(MoeOutSlot { tensor, capacity: out_elems });
            entry.out_slots.len() - 1
        }
    };
    Ok((key, slot_idx))
}

#[allow(clippy::too_many_arguments)]
fn indexed_moe_forward_fused_q8_1_input(
    weight: &CudaView<u8>,
    w_shape: &crate::Shape, //[num_experts, n, k]
    w_dtype: GgmlDType,
    input: IndexedMoeInput<'_>,
    in_shape: &crate::Shape, //[batch, topk or 1, k]
    ids: &CudaView<u32>,
    idx_shape: &crate::Shape, //[batch, topk]
    dev: &CudaDevice,
) -> Result<Tensor> {
    let (num_experts, n, k) = w_shape.dims3()?;
    let batch = in_shape.dims()[0];
    let input_dim1 = in_shape.dims()[1];

    let topk = idx_shape.dims()[1];
    assert!(batch == idx_shape.dims()[0], "batch dim not match!");

    // Quantize input into q8_1.
    let total_rows = batch * input_dim1;
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    // Get Q8_1 metadata.
    let q8_1_block_size = GgmlDType::Q8_1.block_size();
    let q8_1_type_size = GgmlDType::Q8_1.type_size();

    // Calculate the size of the output buffer in bytes.
    let num_blocks_per_row = k_padded / q8_1_block_size;
    let dst_row_size_bytes = num_blocks_per_row * q8_1_type_size;
    let y_size_in_bytes = total_rows * dst_row_size_bytes;
    // Padded-tail-only zero-init -- see mul_mat_vec_via_q8_1's own
    // comment. (The indexed kernel's dot loop additionally never reads
    // past `k`'s own blocks, but the conditional stays conservative.)
    // SAFETY (alloc branch): no padded tail, quantize_q8_1 overwrites
    // every byte before any read.
    // A pre-quantized (Q8_1) input skips the buffer and the quantize
    // pass entirely -- the caller's blocks feed the kernel directly.
    // output buffer -- shape known up front, needed to size the workspace.
    let outsize = batch * topk * n;

    // Alias-bug fix (2026-09-01, see this function's own preceding header
    // comment for the full design/safety argument): both the input-quant
    // scratch and the output live in a persistent, per-(device,stream)
    // workspace, grown on demand, instead of a fresh per-call
    // `dev.alloc`. `moe_workspace_ensure` grows the entry if needed, then
    // the lock is re-acquired and held for the rest of this function so
    // every kernel launch below writes into the SAME entry it was sized
    // for -- no lookup/replace race between the grow and the actual use.
    let (key, slot_idx) = moe_workspace_ensure(dev, outsize, y_size_in_bytes)?;
    let map_lock = MOE_WORKSPACE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let mut map = map_lock.lock().expect("moe workspace mutex poisoned");
    let entry = map.get_mut(&key).expect("moe_workspace_ensure just populated this key");
    assert_eq!(
        entry.stream_ptr, key.1,
        "indexed_moe_forward: workspace entry's recorded stream does not match the current \
         dispatch's stream -- see this function's RUNTIME TRIPWIRE comment above. This should be \
         unreachable given the (device,stream) keying; if it fires, the keying itself has a gap, \
         not just this assert."
    );

    let input_quant_owned: Option<CudaView<'_, u8>> = match &input {
        IndexedMoeInput::Q8_1(view) => {
            if view.len() < y_size_in_bytes {
                crate::bail!(
                    "indexed_moe_forward: pre-quantized input holds {} bytes, needs {y_size_in_bytes}",
                    view.len()
                );
            }
            None
        }
        IndexedMoeInput::F32(view) => {
            // Pass the FULL persistent scratch (capacity >= y_size_in_bytes,
            // possibly larger from an earlier bigger call) -- `quantize_q8_1`
            // only writes the `total_rows`/`k`-derived prefix it computes
            // internally, matching exactly `y_size_in_bytes`; any extra
            // capacity is simply untouched, not read by the caller below
            // either (sliced to `0..y_size_in_bytes` there).
            quantize_q8_1(view, &mut entry.input_quant, k, total_rows, dev)?;
            Some(entry.input_quant.slice(0..y_size_in_bytes))
        }
        IndexedMoeInput::Bf16(view) => {
            quantize_q8_1_from_bf16(view, &mut entry.input_quant, k, total_rows, dev)?;
            Some(entry.input_quant.slice(0..y_size_in_bytes))
        }
    };

    let mut out_storage_guard = entry.out_slots[slot_idx].tensor.storage_mut_and_layout().0;
    let out: &mut CudaSlice<f32> = match &mut *out_storage_guard {
        crate::Storage::Cuda(storage) => storage.as_cuda_slice_mut::<f32>()?,
        _ => unreachable!("moe workspace `out` is always constructed as a CUDA F32 tensor"),
    };
    if !std::env::var("CEREBRA_MMQ_PTR_TRACE").unwrap_or_default().is_empty() {
        use cudarc::driver::DevicePtr;
        let ptr = out.device_ptr(&dev.cuda_stream()).0;
        eprintln!(
            "cerebra out ptr trace: workspace ptr={:#014x} outsize={outsize} slot={slot_idx} capacity={}",
            ptr, entry.out_slots[slot_idx].capacity
        );
    }

    // Warp-per-row variant for SMALL-K rows (rainfall-one, 2026-08-29):
    // with `k / QK_K <= 4` k-blocks per row, the block-per-row shape
    // leaves most of its 128 threads idle (measured ~14% of peak
    // bandwidth on a k=512 Mixture of Experts down projection); the
    // `_wr` kernels give each of the 4 warps its own row instead
    // (grid.x = ceil(n / 4), no shared-memory reduction). Only wired
    // for the dtypes that real down projections use.
    let small_k = k / 256 <= 4;
    // EXPERT-GROUPED dispatch (rainfall-one, 2026-08-30): at large task
    // counts (continuous batching -- batch up to 32, topk 8) the
    // task-major kernels read each selected expert's weights once per
    // TASK, so tasks sharing an expert multiply HBM weight traffic. The
    // `_grp` kernels are expert-major (grid.y = expert): weights are
    // read once per ACTIVE expert, outputs bit-identical (per-task
    // accumulation order unchanged). Threshold: past ~4x the expert
    // count in tasks the reuse is guaranteed; below it the task-major
    // shapes keep their smaller grids. The grouped kernel's shared task
    // list caps at 256 tasks.
    // MEASURED NULL (2026-08-30, A100, 35B MoE, continuous batching):
    // grouped was SLOWER at every batch width (N=8 380->291, N=32
    // 529->483 tok/s aggregate) -- the task-major grid already gets
    // enough L2 weight reuse that the grouped kernel's serialized
    // per-task vec_dots cost more than the HBM reads they save. Kept
    // behind CEREBRA_MOE_GROUPED=1 for future tuning; default off.
    let total_tasks = batch * topk;
    // Goal-2500 Step 7.5 (2026-08-31): the `grp_family` (Q8_0, `_grp`
    // kernels) task-list cap (256, its own kernel's shared-memory limit)
    // is UNRELATED to and smaller than the MMQ template's IMMQ_MAX_TASKS
    // (1024) below -- kept as its own gate, untouched, still bounded at
    // 1024 tasks total (never chunked; Q8_0 experts are not Cerebra's
    // hot MoE dtypes and this path is off by default regardless).
    //
    // Deliberately NOT given the task-count-driven default treatment
    // `mmq_moe` gets below (2026-09-05 enablement) -- this is the naive
    // expert-major register-tile kernel, MEASURED NULL at every decode
    // width tested (see the comment a few lines up), a confirmed dead
    // end kept only for override-driven experimentation, not a
    // candidate for any production default at any task count.
    let grouped = !std::env::var("CEREBRA_MOE_GROUPED").unwrap_or_default().is_empty()
        && total_tasks > 32
        && total_tasks <= 1024
        && matches!(
            w_dtype,
            GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K | GgmlDType::Q8_0
        );
    // Tiled MMQ for the hot expert dtypes (round 3, see the kernel's
    // header comment): weight tiles unpacked to shared memory once per
    // block and reused across a task-column tile -- the shape the
    // per-task vec_dot kernels (rounds 1/2, kept below for Q6K/Q8_0)
    // could not reach. Grid: (row_tiles, col_tile_stride, experts),
    // block (32, IMMQ_NWARPS).
    // Goal-2500 Step 7.5 (2026-08-31): NO upper bound on total_tasks here
    // (unlike `grouped` above) -- prefill's total_tasks (seq_len x topk,
    // e.g. ~20,000 at a 2.5k-token prompt) blew past both the old
    // `grouped` cap and the MMQ kernel's own IMMQ_MAX_TASKS (1024)
    // shared-memory task list, so prefill NEVER reached the tiled MMQ
    // path Steps 1-2 built -- confirmed live via an eager nsys profile
    // at 2.5k context: MoE prefill through the task-major `_wr` kernels
    // was ~68% of GPU time, the tiled MMQ kernels combined only ~2.4%.
    // The launch loop below (`mmq_chunks`) splits any total_tasks count
    // into IMMQ_MAX_TASKS-sized launches, so this dtype gate no longer
    // needs to reject large counts -- it always routes MMQ-eligible
    // dtypes through MMQ when the row-major (`input_dim1 == 1`)
    // precondition chunking needs holds (checked at the call site).
    //
    // Enablement (2026-09-05): the `total_tasks > 32` flat gate above
    // is RETIRED as the tier's authoritative condition -- it applied
    // the SAME threshold at every scale, which is exactly what made a
    // real prefill win (Single's TTFT A/B: -31 to -45%) coexist with a
    // real decode-width regression (this session's ladder: N=4/8/16 at
    // -17.6%/-13.8%/-7.9% decode tok/s) behind one boolean. The tier now
    // engages by TASK COUNT: default ON once `total_tasks` reaches
    // `moe_grouped_min_tasks()` (1,024 by default -- see that function's
    // own doc), no env needed. `CEREBRA_MOE_GROUPED` stays as a pure
    // force-override for experiments: "1" forces every hot dtype on at
    // ANY task count (what this session's own decode-width measurement
    // used); a specific dtype string ("q4k"/"q5k"/"q6k") forces ONLY
    // that dtype on and EXCLUDES the others regardless of task count --
    // unchanged correctness-bisection semantics from before, just no
    // longer the production default path.
    let moe_grouped_env = std::env::var("CEREBRA_MOE_GROUPED").unwrap_or_default();
    let task_count_gate = total_tasks >= moe_grouped_min_tasks();
    let mmq_eligible_dtype = match w_dtype {
        GgmlDType::Q4K => mmq_dtype_eligible(&moe_grouped_env, task_count_gate, "q4k", &["q5k", "q6k"]),
        GgmlDType::Q5K => mmq_dtype_eligible(&moe_grouped_env, task_count_gate, "q5k", &["q4k", "q6k"]),
        // Step 2 of the Goal-2500 campaign (2026-08-30): Q6K down
        // projections through the same tiled MMQ path. "q6k" bisects
        // independently of Q4K/Q5K for correctness isolation.
        GgmlDType::Q6K => mmq_dtype_eligible(&moe_grouped_env, task_count_gate, "q6k", &["q4k", "q5k"]),
        _ => false,
    };
    let mmq_moe = mmq_eligible_dtype;
    // Goal-2500 Step 6 (2026-09-01): tensor-core int8 mma MMQ, THIRD tier
    // ahead of the dp4a mmq_moe tier -- CEREBRA_MMQ_MMA=1 selects it, and
    // ONLY for Q4K (M1's dtype; Q5K/Q6K land in M4). Non-Q4K dtypes and
    // env-off fall straight through to the existing mmq_moe/grp_family/
    // task-major chain below, untouched. `mma_moe` and `mmq_moe` are
    // mutually exclusive by construction (this dtype gate), so `grp_family`
    // below (which already excludes `mmq_moe`) needs no separate mma
    // exclusion.
    let mma_moe =
        !std::env::var("CEREBRA_MMQ_MMA").unwrap_or_default().is_empty() && total_tasks > 32 && matches!(w_dtype, GgmlDType::Q4K);
    // Shape census (2026-09-01, kona-a0 plan step a): env-gated, logs every
    // mma-dispatched call's shape so the live dispatch surface (which call
    // classes actually hit mma_moe -- gate_up vs down-projection, k_padded
    // vs k, both input_dim1 arms) can be enumerated instead of guessed.
    // Cheap enough to leave in permanently behind the env gate.
    if mma_moe && !std::env::var("CEREBRA_MMQ_MMA_TRACE").unwrap_or_default().is_empty() {
        let input_arm = match &input {
            IndexedMoeInput::Q8_1(_) => "Q8_1(caller-provided)",
            IndexedMoeInput::F32(_) => "F32(quantize-on-the-fly)",
            IndexedMoeInput::Bf16(_) => "Bf16(quantize-on-the-fly)",
        };
        eprintln!(
            "cerebra mma_moe dispatch: w_dtype={w_dtype:?} n={n} k={k} k_padded={k_padded} \
             input_dim1={input_dim1} total_tasks={total_tasks} num_experts={num_experts} input_arm={input_arm}"
        );
    }
    // The round-1/2 expert-major vec_dot kernel, kept for Q8_0 when
    // everything is grouped ("1"); dtype-bisect values leave
    // non-selected dtypes on the task-major kernels entirely.
    let grp_family = grouped
        && !mmq_moe
        && !mma_moe
        && moe_grouped_env == "1"
        && matches!(w_dtype, GgmlDType::Q8_0);
    // Dispatch trace (2026-09-05, CEREBRA_MOE_TIER_TRACE=1): prints the
    // resolved tier for every indexed-MoE call -- the precedence-audit
    // instrument for the task-count-gate enablement above. The env
    // lookup is checked ONCE per process (`OnceLock`, supervisor review
    // of this enablement, round 2: this is a per-launch hot path, a
    // string env read on every call is real per-call cost even when
    // tracing is off) -- the branch below then costs one atomic load
    // plus a boolean check when disabled, not a string parse.
    static MOE_TIER_TRACE_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *MOE_TIER_TRACE_ENABLED
        .get_or_init(|| !std::env::var("CEREBRA_MOE_TIER_TRACE").unwrap_or_default().is_empty())
    {
        let tier = if mma_moe {
            "mma_moe"
        } else if mmq_moe {
            "mmq_moe"
        } else if grp_family {
            "grp_family"
        } else {
            "task_major"
        };
        eprintln!(
            "cerebra moe_tier_trace: tier={tier} w_dtype={w_dtype:?} total_tasks={total_tasks} \
             task_count_gate={task_count_gate} moe_grouped_env={moe_grouped_env:?} \
             min_tasks={}",
            moe_grouped_min_tasks()
        );
    }
    // Per-device geometry (never hardcode for one card): both compiled
    // tile shapes are runtime-selectable. CEREBRA_MMQ_GEOM=y64 picks
    // the fatter (8, 64, 8) row tiles; the default y32 (8, 32, 4) is
    // the A100-40GB-measured best. A new device class (H200, ...) is a
    // re-measure of this env, not a kernel rebuild.
    let mmq_y64 = std::env::var("CEREBRA_MMQ_GEOM").as_deref() == Ok("y64");
    let (kernel_name, warp_rows) = if mma_moe {
        // Goal-2500 Lane A / A2 (2026-09-01): the 8-warp/128-row geometry
        // ported from llama.cpp's real Ampere Q4_K MMQ config replaces
        // the killed M1/M2 one-warp/16-row geometry (measured ~41% of
        // dp4a throughput at N=128, col_stride swept -- see campaign
        // memory for the full verdict; that kernel stays in the fork,
        // gated off, as a documented dead end, not deleted). `k` MUST be
        // a multiple of QK_K=256 (asserted below).
        ("indexed_mul_mat_q4_K_moe_mma8w", false)
    } else if mmq_moe {
        let name = match (w_dtype, mmq_y64) {
            (GgmlDType::Q4K, false) => "indexed_mul_mat_q4_K_moe",
            (GgmlDType::Q5K, false) => "indexed_mul_mat_q5_K_moe",
            (GgmlDType::Q4K, true) => "indexed_mul_mat_q4_K_moe_y64",
            (GgmlDType::Q5K, true) => "indexed_mul_mat_q5_K_moe_y64",
            // No y64 twin compiled for Q6K (Step 2 shipped only the
            // A100-measured y32 geometry) -- CEREBRA_MMQ_GEOM=y64 is a
            // no-op for this dtype until a twin is added and measured.
            (GgmlDType::Q6K, _) => "indexed_mul_mat_q6_K_moe",
            _ => unreachable!("mmq gate above"),
        };
        (name, false)
    } else if grp_family {
        let name = match w_dtype {
            GgmlDType::Q8_0 => "indexed_moe_forward_q8_0_q8_1_grp",
            _ => unreachable!("grp_family gate above"),
        };
        (name, true)
    } else {
        match w_dtype {
            GgmlDType::Q2K => ("indexed_moe_forward_q2k_q8_1", false),
            GgmlDType::Q3K => ("indexed_moe_forward_q3k_q8_1", false),
            // Q4K experiment (2026-08-29): also try warp-rows for the k=2048
            // gate/up shape (8 k-blocks -> each warp runs 4 serial
            // iterations, but drops the cross-warp barrier and quarters the
            // block count). Measured on the consuming workload; revert to
            // `false` if it does not hold its gain.
            GgmlDType::Q4K if k / 256 <= 8 => ("indexed_moe_forward_q4k_q8_1_wr", true),
            GgmlDType::Q4K => ("indexed_moe_forward_q4k_q8_1", false),
            GgmlDType::Q5K if small_k => ("indexed_moe_forward_q5k_q8_1_wr", true),
            GgmlDType::Q5K => ("indexed_moe_forward_q5k_q8_1", false),
            GgmlDType::Q6K if small_k => ("indexed_moe_forward_q6k_q8_1_wr", true),
            GgmlDType::Q6K => ("indexed_moe_forward_q6k_q8_1", false),
            GgmlDType::Q8_0 => ("indexed_moe_forward_q8_0_q8_1", false),
            _ => crate::bail!("unsupported dtype for indexed_moe_forward {w_dtype:?}"),
        }
    };
    let func = dev.get_or_load_func(kernel_name, &candle_kernels::QUANTIZED)?;
    let (nblocks, nwarps) = if warp_rows {
        ((n as u32).div_ceil(4), 4)
    } else {
        (n as u32, 4)
    };
    // Goal-2500 Step 6: launch-config assertions for the mma tile's own
    // shape assumptions -- the poison-arm lesson (a silent geometry
    // mismatch profiles as something else entirely, not a crash) applies
    // doubly here since this is the first tier whose row/K tiling isn't
    // just "one row or 4 rows per warp" but a fixed 16x8 hardware tile.
    if mma_moe {
        if k % 256 != 0 {
            crate::bail!("indexed_mul_mat_q4_K_moe_mma: k={k} must be a multiple of QK_K=256");
        }
        if n == 0 {
            crate::bail!("indexed_mul_mat_q4_K_moe_mma: n=0 (no output rows)");
        }
    }
    let cfg = if mma_moe {
        // Goal-2500 Lane A / A2: 8 warps (256 threads) per (128-row tile,
        // col-tile-stride-wide 8-task tile) -- I=128/nthreads=256 fixed
        // by llama.cpp's real Ampere Q4_K MMQ config
        // (mmq-config-ampere.cuh CASE(GGML_TYPE_Q4_K, 256, 1, 128, 8, ...),
        // pinned commit 0eadefebd3f8f92a86d634a0e5b8fffc9dc792c0), not a
        // dp4a-style tunable. Column-tile stride IS env-tunable per
        // device (same CEREBRA_MMQ_MMA_COL_STRIDE precedent); default
        // still unmeasured until A3's sweep picks a real A100 value.
        let col_stride: u32 = std::env::var("CEREBRA_MMQ_MMA_COL_STRIDE")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v| (1..=256).contains(v))
            .unwrap_or(4);
        cudarc::driver::LaunchConfig {
            grid_dim: ((n as u32).div_ceil(128), col_stride, num_experts as u32),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        }
    } else if mmq_moe {
        // Geometry must mirror the kernel's IMMQ constants: Q4K runs
        // (mmq_y 32, nwarps 4), Q5K (mmq_y 64, nwarps 8). grid.y = 8
        // column-tile strides (hot experts loop in-kernel, see the
        // kernel's grid-stride comment).
        // Geometry mirrors the selected kernel's IMMQ constants; the
        // column-tile stride is env-tunable per device too
        // (CEREBRA_MMQ_COL_STRIDE, default 2 -- A100-measured, Step 1
        // autotune of the Goal-2500 campaign: 1049.1 -> 1219.5 tok/s
        // aggregate at N=128, +16.2%, confirmed stable across repeat
        // sweeps and non-monotonic across {1,2,4,8,16}).
        // Q6K has no y64 twin compiled (see kernel_name match above) --
        // its grid geometry must stay y32 regardless of the env flag, or
        // grid math would assume tiles the selected kernel does not have.
        let effective_y64 = mmq_y64 && !matches!(w_dtype, GgmlDType::Q6K);
        let (mmq_y, nwarps_mmq) = if effective_y64 { (64u32, 8u32) } else { (32u32, 4u32) };
        let col_stride: u32 = std::env::var("CEREBRA_MMQ_COL_STRIDE")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v| (1..=256).contains(v))
            .unwrap_or(2);
        cudarc::driver::LaunchConfig {
            grid_dim: ((n as u32).div_ceil(mmq_y), col_stride, num_experts as u32),
            block_dim: (WARP_SIZE as u32, nwarps_mmq, 1),
            shared_mem_bytes: 0,
        }
    } else if grp_family {
        cudarc::driver::LaunchConfig {
            grid_dim: (nblocks, num_experts as u32, 1),
            block_dim: (WARP_SIZE as u32, nwarps, 1),
            shared_mem_bytes: 0,
        }
    } else {
        cudarc::driver::LaunchConfig {
            grid_dim: (nblocks, batch as u32, topk as u32),
            block_dim: (WARP_SIZE as u32, nwarps, 1),
            shared_mem_bytes: 0,
        }
    };

    // Goal-2500 Step 7.5 (2026-08-31): route the MMQ launch through a
    // per-chunk loop when `total_tasks` exceeds the MMQ kernel's own
    // IMMQ_MAX_TASKS (1024, its shared-memory task list's fixed size --
    // see the kernel's own `#define`) -- prefill's total_tasks (seq_len
    // x topk, tens of thousands at real prompt lengths) always exceeded
    // it, so prefill never reached this kernel before (confirmed via
    // eager nsys profile at 2.5k context: MoE prefill through the
    // task-major fallback was ~68% of GPU time). Each chunk is an
    // independent, complete MMQ launch over a disjoint slice of
    // [batch, topk] rows -- trivially correct (chunks never touch each
    // other's ids/input/output regions, and every chunk computes
    // exactly what a smaller standalone call over that row range would).
    // Two chunkable shapes, each with its own boundary rule:
    // - `input_dim1 == 1` (one activation row per token, routed to
    //   `topk` experts): chunk boundaries must land on whole TOKEN rows
    //   (`token_start * topk`) -- the first arm below.
    // - `input_dim1 == topk` (one activation row per TASK): chunk
    //   boundaries are per-task, strictly simpler (slice ids/input/out
    //   at task granularity; each chunk relaunches as its own
    //   `(batch = chunk_tasks, topk = 1, input_dim1 = 1)` problem,
    //   which the kernel's own indexing makes literally identical:
    //   `input_row = t / 1 = t`, ids and outputs are flat per-task
    //   arrays either way) -- the second arm below.
    // HISTORY (2026-09-04, cerebra-spec-nondeterminism investigation):
    // an earlier version of this comment claimed no `input_dim1 != 1`
    // caller exists and let that shape fall through to the single-shot
    // launch. The DOWN projection (`QuantizedExperts::weighted_sum`'s
    // `[seq, top_k, intermediate]` input) is exactly such a caller, and
    // at prefill task counts its unchunked launch overflowed the
    // kernel's 1024-entry per-expert shared task list -- silently
    // truncated in atomicAdd arrival order (nondeterministic dropped
    // tasks, their output rows never written). Confirmed live via
    // dtype bisect: CEREBRA_MOE_GROUPED=q6k (down projection only)
    // diverged 3 modes in 4 identical runs; =q4k (gate_up only,
    // token-chunked) was 4/4 bit-identical. Any OTHER over-cap shape
    // now fails loudly instead of corrupting. Per-task chunking also
    // removes the `unsigned short` task-id overflow any unchunked
    // launch past 65,535 tasks would have hit inside the kernel.
    // Goal-2500 Step 6 (2026-09-01): the mma tier routes through this
    // SAME loop, not a second chunking path -- `func`/`cfg` are already
    // tier-dispatched above, so this loop just launches whichever kernel
    // was selected, chunk after chunk. `IMMQ_MAX_TASKS` is re-derived
    // (not assumed) for the mma kernel too: its own task_list is sized
    // identically (1024 unsigned shorts, same shared-memory budget) --
    // confirmed by reading indexed_mul_mat_q4_K_moe_mma's own
    // IMMQ_MMA_MAX_TASKS definition, not copied blindly. If a future mma
    // dtype ever needs a different in-block cap, this condition and the
    // constant below both need to become per-tier.
    // Goal-2500 Step 6 debug instrument (2026-09-01, kona-a0 plan item 2):
    // launch-time checksum, run as close as possible to actual kernel
    // entry (immediately before the real dispatch below) so its result
    // reflects device memory state at execution time -- offline replay of
    // a live dump's captured bytes proved CORRECT while the SAME live
    // call's own output was wrong, meaning a post-hoc dump (captured
    // AFTER kernel completion) might not be observing what the kernel
    // actually read. Compare this checksum against a host-side XOR of the
    // same dump's weight.bin/input.bin: a mismatch localizes a
    // write-after-read hazard; a match rules that out entirely.
    if mma_moe && !std::env::var("CEREBRA_MMQ_CHECKSUM").unwrap_or_default().is_empty() {
        let checksum_func = dev.get_or_load_func("cerebra_mmq_checksum", &candle_kernels::QUANTIZED)?;
        let checksum_out = dev.alloc_zeros::<u32>(3)?;
        let input_view_for_checksum: CudaView<u8> = match (&input, &input_quant_owned) {
            (IndexedMoeInput::Q8_1(view), _) => view.slice(0..y_size_in_bytes),
            (_, Some(buf)) => buf.slice(0..y_size_in_bytes),
            _ => unreachable!("non-Q8_1 inputs always build an owned quantize buffer above"),
        };
        let weight_len = weight.len() as u32;
        let input_len = input_view_for_checksum.len() as u32;
        let ids_count = ids.len() as u32;
        let mut cs_builder = checksum_func.builder();
        cs_builder.arg(weight);
        barg!(cs_builder, weight_len);
        cs_builder.arg(&input_view_for_checksum);
        barg!(cs_builder, input_len);
        cs_builder.arg(ids);
        barg!(cs_builder, ids_count);
        cs_builder.arg(&checksum_out);
        let cs_cfg = cudarc::driver::LaunchConfig {
            grid_dim: (256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { cs_builder.launch(cs_cfg) }.w()?;
        let checksums: Vec<u32> = dev.cuda_stream().memcpy_dtov(&checksum_out).w()?;
        eprintln!(
            "cerebra mmq checksum (launch-time, total_tasks={total_tasks}): weight_xor={:#010x} input_xor={:#010x} ids_xor={:#010x}",
            checksums[0], checksums[1], checksums[2]
        );
    }
    if (mmq_moe || mma_moe) && total_tasks > 1024 && input_dim1 == 1 {
        const IMMQ_MAX_TASKS: usize = 1024;
        let chunk_tokens = (IMMQ_MAX_TASKS / topk).max(1);
        let input_view: CudaView<u8> = match (&input, &input_quant_owned) {
            (IndexedMoeInput::Q8_1(view), _) => view.slice(0..y_size_in_bytes),
            (_, Some(buf)) => buf.slice(0..y_size_in_bytes),
            _ => unreachable!("non-Q8_1 inputs always build an owned quantize buffer above"),
        };
        let mut token_start = 0usize;
        while token_start < batch {
            let chunk_batch = chunk_tokens.min(batch - token_start);
            let chunk_task_start = token_start * topk;
            let chunk_tasks = chunk_batch * topk;

            let input_chunk = input_view.slice(
                token_start * dst_row_size_bytes..(token_start + chunk_batch) * dst_row_size_bytes,
            );
            let ids_chunk = ids.slice(chunk_task_start..chunk_task_start + chunk_tasks);
            let out_chunk = out.slice(chunk_task_start * n..(chunk_task_start + chunk_tasks) * n);

            let mut builder = func.builder();
            builder.arg(weight);
            builder.arg(&input_chunk);
            builder.arg(&ids_chunk);
            builder.arg(&out_chunk);
            barg!(
                builder,
                n as i32,
                k as i32,
                chunk_batch as i32,
                topk as i32,
                k_padded as i32,
                input_dim1 as i32
            );
            unsafe { builder.launch(cfg) }.w()?;

            token_start += chunk_batch;
        }
    } else if (mmq_moe || mma_moe) && total_tasks > 1024 && input_dim1 == topk {
        // Per-task chunking for the one-row-per-TASK shape (the down
        // projection's `[batch, topk, k]` input) -- see the boundary-rule
        // comment above. Each chunk is a complete, independent launch
        // over a contiguous task range, re-expressed as its own
        // `(batch = chunk_tasks, topk = 1, input_dim1 = 1)` problem:
        // in-kernel `input_row = t / topk` becomes `t / 1 = t`, exactly
        // the per-task row this shape stores, and ids/output indexing is
        // flat per-task in both formulations. No chunk can exceed the
        // kernel's IMMQ_MAX_TASKS (1024) shared task list, restoring the
        // list's no-truncation precondition for every launch.
        const IMMQ_MAX_TASKS: usize = 1024;
        let input_view: CudaView<u8> = match (&input, &input_quant_owned) {
            (IndexedMoeInput::Q8_1(view), _) => view.slice(0..y_size_in_bytes),
            (_, Some(buf)) => buf.slice(0..y_size_in_bytes),
            _ => unreachable!("non-Q8_1 inputs always build an owned quantize buffer above"),
        };
        let mut task_start = 0usize;
        while task_start < total_tasks {
            let chunk_tasks = IMMQ_MAX_TASKS.min(total_tasks - task_start);

            let input_chunk = input_view.slice(
                task_start * dst_row_size_bytes..(task_start + chunk_tasks) * dst_row_size_bytes,
            );
            let ids_chunk = ids.slice(task_start..task_start + chunk_tasks);
            let out_chunk = out.slice(task_start * n..(task_start + chunk_tasks) * n);

            let mut builder = func.builder();
            builder.arg(weight);
            builder.arg(&input_chunk);
            builder.arg(&ids_chunk);
            builder.arg(&out_chunk);
            barg!(
                builder,
                n as i32,
                k as i32,
                chunk_tasks as i32, // batch
                1i32,               // topk
                k_padded as i32,
                1i32 // input_dim1
            );
            unsafe { builder.launch(cfg) }.w()?;

            task_start += chunk_tasks;
        }
    } else if (mmq_moe || mma_moe) && total_tasks > 1024 {
        // Neither chunkable shape -- refuse loudly rather than launch a
        // kernel whose 1024-entry shared task list would silently
        // truncate in nondeterministic atomicAdd order (the exact
        // corruption the 2026-09-04 investigation root-caused). No
        // current caller reaches this arm; if a future shape does, it
        // needs its own chunk-boundary rule added above, not a silent
        // fallthrough.
        crate::bail!(
            "indexed_moe_forward: total_tasks={total_tasks} exceeds the tiled MMQ kernel's \
             1024-task shared list and input_dim1={input_dim1} matches neither chunkable shape \
             (1 or topk={topk}) -- refusing the unchunked launch, which would silently drop \
             tasks in nondeterministic order"
        );
    } else {
        let mut builder = func.builder();
        builder.arg(weight);
        match (&input, &input_quant_owned) {
            (IndexedMoeInput::Q8_1(view), _) => {
                builder.arg(view);
            }
            (_, Some(buf)) => {
                builder.arg(buf);
            }
            _ => unreachable!("non-Q8_1 inputs always build an owned quantize buffer above"),
        }
        builder.arg(ids);
        builder.arg(&*out);

        barg!(
            builder,
            n as i32,
            k as i32,
            batch as i32,
            topk as i32,
            k_padded as i32,
            input_dim1 as i32
        );
        unsafe { builder.launch(cfg) }.w()?;
    }

    // Scoped dump (2026-09-01, kona-a0 plan step c): the shape census
    // (CEREBRA_MMQ_MMA_TRACE) showed the LIVE mma_moe call is the exact
    // same shape family as the already-verified fix (n=1024 k=2048
    // k_padded==k input_dim1==1), differing only in total_tasks (96 live
    // vs 224 in the earlier offline dump) -- ruling out both of kona-a0's
    // candidate uncovered shapes. Capturing real bytes at THIS smaller
    // live scale to check whether the bug is task-count-dependent.
    // Separate env gate (CEREBRA_MMQ_DUMP2) from the removed original
    // hook so this doesn't collide with any leftover env in the shell.
    // Broadened (2026-09-01, kona-a0 triangulation step 1) to also fire on
    // mmq_moe (dp4a) so the SAME live call shape can be captured under the
    // dp4a tier for a real ground-truth comparison against the mma tier's
    // output on identical routing/activations -- deterministic dispatch
    // means the same prompt produces the same ids/activations regardless
    // of which tier computes the result, since routing happens upstream
    // of this kernel choice.
    static DUMPED2: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if (mma_moe || mmq_moe)
        && matches!(w_dtype, GgmlDType::Q4K)
        && n == 1024
        && k == 2048
        && !std::env::var("CEREBRA_MMQ_DUMP2").unwrap_or_default().is_empty()
        && !DUMPED2.swap(true, std::sync::atomic::Ordering::SeqCst)
    {
        use std::io::Write;
        let dump_dir = std::path::Path::new("/tmp/mmq_dump2");
        let _ = std::fs::create_dir_all(dump_dir);
        let dump_stream = dev.cuda_stream();
        let weight_bytes: Vec<u8> = dump_stream.memcpy_dtov(weight).w()?;
        let ids_u32: Vec<u32> = dump_stream.memcpy_dtov(ids).w()?;
        let out_f32: Vec<f32> = dump_stream.memcpy_dtov(&*out).w()?;
        let input_bytes: Vec<u8> = match (&input, &input_quant_owned) {
            (IndexedMoeInput::Q8_1(view), _) => dump_stream.memcpy_dtov(view).w()?,
            (_, Some(buf)) => dump_stream.memcpy_dtov(buf).w()?,
            _ => unreachable!("non-Q8_1 inputs always build an owned quantize buffer above"),
        };
        let meta = format!(
            "w_dtype={w_dtype:?} num_experts={num_experts} n={n} k={k} batch={batch} topk={topk} \
             k_padded={k_padded} input_dim1={input_dim1} total_tasks={total_tasks} \
             weight_bytes={} ids_len={} out_len={} input_bytes={}\n",
            weight_bytes.len(),
            ids_u32.len(),
            out_f32.len(),
            input_bytes.len()
        );
        if let Ok(mut f) = std::fs::File::create(dump_dir.join("meta.txt")) {
            let _ = f.write_all(meta.as_bytes());
        }
        let _ = std::fs::write(dump_dir.join("weight.bin"), &weight_bytes);
        let _ = std::fs::write(
            dump_dir.join("ids.bin"),
            ids_u32.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
        );
        let _ = std::fs::write(
            dump_dir.join("out.bin"),
            out_f32.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
        );
        let _ = std::fs::write(dump_dir.join("input.bin"), &input_bytes);
        eprintln!("cerebra: mmq dump2 written to /tmp/mmq_dump2 ({meta})");
    }

    let mut out_shape = in_shape.dims().to_vec();
    out_shape.pop();
    out_shape.push(n);
    out_shape[1] = topk;

    // Release the storage write-lock before narrowing -- `Tensor::narrow`
    // reads the storage via its own (shared) lock internally, and holding
    // the write guard here would deadlock against that.
    drop(out_storage_guard);
    let result = entry.out_slots[slot_idx].tensor.narrow(0, 0, outsize)?.reshape(out_shape)?;
    Ok(result)
}

impl QCudaStorage {
    pub fn indexed_moe_forward(
        &self,
        self_shape: &crate::Shape, //[num_experts, n, k]
        input: &CudaStorage,       //[batch, topk or 1, k]
        input_l: &crate::Layout,
        ids: &CudaStorage, //[batch, topk]
        ids_l: &crate::Layout,
    ) -> Result<Tensor> {
        if matches!(
            self.dtype(),
            GgmlDType::Q8_0
                | GgmlDType::Q2K
                | GgmlDType::Q3K
                | GgmlDType::Q4K
                | GgmlDType::Q5K
                | GgmlDType::Q6K
        ) {
            // Dtype-dispatched activation input: BF16 rows quantize
            // directly (no separate widening cast pass -- see
            // `quantize_q8_1_from_bf16`); F32 keeps the historical path.
            use crate::backend::BackendStorage;
            use crate::DType;
            let input_enum = match input.dtype() {
                DType::F32 => {
                    IndexedMoeInput::F32(input.as_cuda_slice::<f32>()?.slice(input_l.start_offset()..))
                }
                DType::BF16 => {
                    IndexedMoeInput::Bf16(input.as_cuda_slice::<bf16>()?.slice(input_l.start_offset()..))
                }
                // A U8 input is the caller's own Q8_1 block stream --
                // pre-quantized by a producer kernel, consumed directly.
                DType::U8 => {
                    IndexedMoeInput::Q8_1(input.as_cuda_slice::<u8>()?.slice(input_l.start_offset()..))
                }
                other => crate::bail!(
                    "indexed_moe_forward input must be F32, BF16, or U8 (pre-quantized Q8_1), got {other:?}"
                ),
            };
            let ids_storage = ids.as_cuda_slice::<u32>()?;
            indexed_moe_forward_fused_q8_1_input(
                &self.data.inner.slice(0..),
                self_shape, //[num_experts, n, k]
                self.dtype(),
                input_enum,
                input_l.shape(), //[batch, topk or 1, k]
                &ids_storage.slice(0..),
                ids_l.shape(), //[batch, topk]
                &self.device,
            )
        } else {
            crate::bail!(
                "The given quantized dtype {:?} is not supported for indexed_moe_forward!",
                self.dtype()
            );
        }
    }

    pub fn zeros(device: &CudaDevice, el_count: usize, dtype: GgmlDType) -> Result<Self> {
        let size_in_bytes = ceil_div(el_count, dtype.block_size()) * dtype.type_size();
        let padded_size_in_bytes =
            ceil_div(el_count + MATRIX_ROW_PADDING, dtype.block_size()) * dtype.type_size();
        let inner = device.alloc_zeros::<u8>(padded_size_in_bytes)?;
        Ok(QCudaStorage {
            data: PaddedCudaSlice {
                inner,
                len: size_in_bytes,
            },
            device: device.clone(),
            dtype,
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<CudaStorage> {
        fn deq<T: GgmlType>(buffer: &[u8], n: usize, dst: &mut [f32]) {
            let slice = unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const T, n) };
            let vec = slice.to_vec();
            T::to_float(&vec, dst)
        }

        let fast_kernel = matches!(
            self.dtype,
            GgmlDType::Q4_0
                | GgmlDType::Q4_1
                | GgmlDType::Q5_0
                | GgmlDType::Q5_1
                | GgmlDType::Q8_0
                | GgmlDType::Q2K
                | GgmlDType::Q3K
                | GgmlDType::Q4K
                | GgmlDType::Q5K
                | GgmlDType::Q6K
                | GgmlDType::Q8K
        );
        if fast_kernel {
            return dequantize_f32(&self.data, self.dtype, elem_count, self.device());
        }
        // Run the dequantization on cpu.

        let buffer = self
            .device
            .clone_dtoh(&self.data.inner.slice(..self.data.len))?;
        let mut out = vec![0.0; elem_count];
        let block_len = elem_count / self.dtype.block_size();
        match self.dtype {
            GgmlDType::F32 => deq::<f32>(&buffer, block_len, &mut out),
            GgmlDType::F16 => deq::<half::f16>(&buffer, block_len, &mut out),
            GgmlDType::BF16 => deq::<half::bf16>(&buffer, block_len, &mut out),
            GgmlDType::Q4_0 => deq::<crate::quantized::BlockQ4_0>(&buffer, block_len, &mut out),
            GgmlDType::Q4_1 => deq::<crate::quantized::BlockQ4_1>(&buffer, block_len, &mut out),
            GgmlDType::Q5_0 => deq::<crate::quantized::BlockQ5_0>(&buffer, block_len, &mut out),
            GgmlDType::Q5_1 => deq::<crate::quantized::BlockQ5_1>(&buffer, block_len, &mut out),
            GgmlDType::Q8_0 => deq::<crate::quantized::BlockQ8_0>(&buffer, block_len, &mut out),
            GgmlDType::Q8_1 => deq::<crate::quantized::BlockQ8_1>(&buffer, block_len, &mut out),
            GgmlDType::Q2K => deq::<crate::quantized::BlockQ2K>(&buffer, block_len, &mut out),
            GgmlDType::Q3K => deq::<crate::quantized::BlockQ3K>(&buffer, block_len, &mut out),
            GgmlDType::Q4K => deq::<crate::quantized::BlockQ4K>(&buffer, block_len, &mut out),
            GgmlDType::Q5K => deq::<crate::quantized::BlockQ5K>(&buffer, block_len, &mut out),
            GgmlDType::Q6K => deq::<crate::quantized::BlockQ6K>(&buffer, block_len, &mut out),
            GgmlDType::Q8K => deq::<crate::quantized::BlockQ8K>(&buffer, block_len, &mut out),
        }

        self.device
            .storage_from_cpu_storage(&crate::CpuStorage::F32(out))
    }

    pub fn dequantize_f16(&self, elem_count: usize) -> Result<CudaStorage> {
        dequantize_f16(&self.data, self.dtype, elem_count, self.device())
    }

    pub fn quantize(&mut self, src: &CudaStorage) -> Result<()> {
        // Run the quantization on cpu.
        let src = match &src.slice {
            crate::cuda_backend::CudaStorageSlice::F32(data) => self.device.clone_dtoh(data)?,
            _ => crate::bail!("only f32 can be quantized"),
        };
        let src_len = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_len, self.dtype)?;
        qcpu_storage.quantize(&src)?;
        let data = qcpu_storage.data()?;
        let padded_len =
            data.len() + MATRIX_ROW_PADDING * self.dtype.type_size() / self.dtype.block_size();
        let mut inner = unsafe { self.device.alloc::<u8>(padded_len)? };
        self.device
            .memcpy_htod(&*data, &mut inner.slice_mut(..data.len()))?;
        self.data = PaddedCudaSlice {
            inner,
            len: data.len(),
        };
        Ok(())
    }

    pub fn quantize_imatrix(
        &mut self,
        src: &CudaStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        // Run the quantization on cpu.
        let src = match &src.slice {
            crate::cuda_backend::CudaStorageSlice::F32(data) => self.device.clone_dtoh(data)?,
            _ => crate::bail!("only f32 can be quantized"),
        };
        let src_len = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_len, self.dtype)?;
        qcpu_storage.quantize_imatrix(&src, imatrix_weights, n_per_row)?;
        let data = qcpu_storage.data()?;
        let padded_len =
            data.len() + MATRIX_ROW_PADDING * self.dtype.type_size() / self.dtype.block_size();
        let mut inner = unsafe { self.device.alloc::<u8>(padded_len)? };
        self.device
            .memcpy_htod(&*data, &mut inner.slice_mut(..data.len()))?;
        self.data = PaddedCudaSlice {
            inner,
            len: data.len(),
        };
        Ok(())
    }

    pub fn quantize_imatrix_onto(
        &mut self,
        src: &crate::CpuStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        // Run the quantization on cpu.
        let src_len = src.as_slice::<f32>()?.len();
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_len, self.dtype)?;

        if let QStorage::Cpu(storage) = &mut qcpu_storage {
            storage.from_float_imatrix(src.as_slice::<f32>()?, imatrix_weights, n_per_row);
        } else {
            unreachable!()
        }

        let data = qcpu_storage.data()?;
        let padded_len =
            data.len() + MATRIX_ROW_PADDING * self.dtype.type_size() / self.dtype.block_size();
        let mut inner = unsafe { self.device.alloc::<u8>(padded_len)? };
        self.device
            .memcpy_htod(&*data, &mut inner.slice_mut(..data.len()))?;
        self.data = PaddedCudaSlice {
            inner,
            len: data.len(),
        };
        Ok(())
    }

    pub fn quantize_onto(&mut self, src: &crate::CpuStorage) -> Result<()> {
        // Run the quantization on cpu.
        let src_len = src.as_slice::<f32>()?.len();
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_len, self.dtype)?;

        if let QStorage::Cpu(storage) = &mut qcpu_storage {
            storage.from_float(src.as_slice::<f32>()?);
        } else {
            unreachable!()
        }

        let data = qcpu_storage.data()?;
        let padded_len =
            data.len() + MATRIX_ROW_PADDING * self.dtype.type_size() / self.dtype.block_size();
        let mut inner = unsafe { self.device.alloc::<u8>(padded_len)? };
        self.device
            .memcpy_htod(&*data, &mut inner.slice_mut(..data.len()))?;
        self.data = PaddedCudaSlice {
            inner,
            len: data.len(),
        };
        Ok(())
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.data.len
    }

    pub fn fwd(
        &self,
        self_shape: &crate::Shape,
        storage: &CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        let max_bm = if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            1
        } else {
            8
        };
        let use_vec_kernel = match layout.shape().dims() {
            [b, m, _k] => b * m <= max_bm,
            [b, _k] => *b <= max_bm,
            _ => false,
        };
        if use_vec_kernel {
            self.dequantize_matmul_vec(self_shape, storage, layout)
        } else {
            self.dequantize_matmul(self_shape, storage, layout)
        }
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        let mut out = vec![0u8; self.data.len];
        self.device
            .memcpy_dtoh(&self.data.inner.slice(..self.data.len), &mut out)?;
        Ok(out)
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        use cudarc::driver::DevicePtr;
        Ok(self.data.inner.device_ptr(self.data.inner.stream()).0 as *const u8)
    }
}

impl QCudaStorage {
    fn dequantize_matmul_vec(
        &self,
        self_shape: &crate::Shape,
        rhs: &CudaStorage,
        rhs_l: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        let (nrows, ncols) = self_shape.dims2()?;
        let rhs = rhs.as_cuda_slice::<f32>()?;
        let rhs = match rhs_l.contiguous_offsets() {
            Some((o1, o2)) => rhs.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "dmmv" }.bt())?,
        };
        let (b_size, k) = match rhs_l.shape().dims() {
            [b, m, k] => (b * m, *k),
            [b, k] => (*b, *k),
            _ => crate::bail!("unexpected rhs shape in dmmv {:?}", rhs_l.shape()),
        };
        if ncols != k {
            crate::bail!("mismatch on matmul dim {self_shape:?} {:?}", rhs_l.shape())
        }

        let out = if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            dequantize_mul_mat_vec(&self.data, &rhs, self.dtype, ncols, nrows, self.device())?
        } else {
            mul_mat_vec_via_q8_1(
                &self.data,
                &rhs,
                self.dtype,
                ncols,
                nrows,
                b_size,
                self.device(),
            )?
        };
        let mut out_shape = rhs_l.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(nrows);
        Ok((out, out_shape.into()))
    }

    fn dequantize_matmul(
        &self,
        self_shape: &crate::Shape,
        storage: &CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        use crate::backend::BackendStorage;
        let (n, k) = self_shape.dims2()?;
        let (b, m, k2) = match layout.shape().dims() {
            &[b, m, k2] => (b, m, k2),
            &[m, k2] => (1, m, k2),
            s => crate::bail!("unexpected shape for input {s:?}"),
        };
        if k2 != k {
            crate::bail!("mismatch on matmul dim {self_shape:?} {:?}", layout.shape())
        }

        let out = if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            let data_f32 = self.dequantize(n * k)?;
            let rhs_l = crate::Layout::new((k, n).into(), vec![1, k], 0).broadcast_as((b, k, n))?;
            storage.matmul(&data_f32, (b, m, n, k), layout, &rhs_l)?
        } else {
            let storage = storage.as_cuda_slice::<f32>()?;
            let storage = match layout.contiguous_offsets() {
                Some((o1, o2)) => storage.slice(o1..o2),
                None => Err(crate::Error::RequiresContiguous {
                    op: "quantized-matmul",
                }
                .bt())?,
            };
            mul_mat_via_q8_1(
                &self.data,
                &storage,
                self.dtype,
                /* x_rows */ n,
                /* x_cols */ k,
                /* y_rows */ k,
                /* y_cols */ b * m,
                self.device(),
            )?
        };
        let mut out_shape = layout.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(n);
        Ok((out, out_shape.into()))
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    device: &CudaDevice,
    data: &[T],
) -> Result<super::QStorage> {
    let data = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, core::mem::size_of_val(data))
    };
    let dtype = T::DTYPE;
    let padded_len = data.len() + MATRIX_ROW_PADDING * dtype.type_size() / dtype.block_size();
    let mut inner = device.alloc_zeros::<u8>(padded_len)?;
    device.memcpy_htod(data, &mut inner.slice_mut(..data.len()))?;
    Ok(QStorage::Cuda(QCudaStorage {
        data: PaddedCudaSlice {
            inner,
            len: data.len(),
        },
        device: device.clone(),
        dtype,
    }))
}

#[cfg(test)]
mod test {
    use super::*;

    /// [Unit] Enablement (2026-09-05): the default threshold is 1,024
    /// when no override is set, matching the real gap between every
    /// decode width measured (<=256 tasks) and every prefill chunk
    /// (4,096 tasks) -- and an unparsable or zero override falls back
    /// to that default rather than propagating or panicking.
    #[test]
    fn unit_moe_grouped_min_tasks_default_and_override() {
        assert_eq!(moe_grouped_min_tasks_from_raw(None), 1024);
        assert_eq!(moe_grouped_min_tasks_from_raw(Some("512")), 512);
        assert_eq!(moe_grouped_min_tasks_from_raw(Some("not-a-number")), 1024);
        assert_eq!(moe_grouped_min_tasks_from_raw(Some("0")), 1024);
    }

    /// [Regression] Enablement (2026-09-05), the predicate this PR's
    /// whole change rests on: a call at a decode-width task count
    /// (measured up to 256 tasks at N=32) must NOT default-engage the
    /// tiled MMQ tier, while a call at prefill-chunk scale (4,096 tasks)
    /// must. Named after the real hardware finding (decode-width
    /// regression, prefill-scale win) this threshold exists to
    /// separate.
    #[test]
    fn regression_task_count_gate_separates_decode_from_prefill_scale() {
        let min_tasks = moe_grouped_min_tasks_from_raw(None);
        let decode_width_n32_tasks = 32 * 8; // N=32, topk=8 -- this session's own ladder
        let prefill_chunk_tasks = 4096;
        assert!(
            decode_width_n32_tasks < min_tasks,
            "decode-width task count ({decode_width_n32_tasks}) must sit below the default \
             threshold ({min_tasks}), or the tier would default-engage at exactly the width \
             measured as a regression"
        );
        assert!(
            prefill_chunk_tasks >= min_tasks,
            "a prefill chunk ({prefill_chunk_tasks} tasks) must clear the default threshold \
             ({min_tasks}), or the tier would default-engage at the scale it was originally \
             a measured, real win"
        );
    }

    /// [Unit] Full precedence truth table for [`mmq_dtype_eligible`]
    /// (enablement round 2, supervisor's precedence audit on #4348):
    /// `""` defers to the task-count gate, `"0"` force-disables
    /// regardless of task count (added specifically because it used to
    /// silently fall through to the SAME behavior as unset -- an
    /// operator setting it would not have gotten what they asked for),
    /// `"1"` force-enables, this dtype's own bisect key force-enables,
    /// a sibling dtype's key force-disables, and anything unrecognized
    /// falls back to the task-count gate rather than silently
    /// disabling.
    #[test]
    fn unit_mmq_dtype_eligible_full_precedence_table() {
        let others = ["q5k", "q6k"];
        // task_count_gate=true cases:
        assert!(mmq_dtype_eligible("", true, "q4k", &others), "unset defers to the gate");
        assert!(mmq_dtype_eligible("1", true, "q4k", &others), "\"1\" force-enables");
        assert!(mmq_dtype_eligible("q4k", true, "q4k", &others), "own key force-enables");
        assert!(
            mmq_dtype_eligible("unrecognized-typo", true, "q4k", &others),
            "unrecognized value defers to the gate, does not silently disable"
        );
        // task_count_gate=false cases:
        assert!(!mmq_dtype_eligible("", false, "q4k", &others), "unset defers to the gate");
        assert!(mmq_dtype_eligible("1", false, "q4k", &others), "\"1\" force-enables regardless of the gate");
        assert!(mmq_dtype_eligible("q4k", false, "q4k", &others), "own key force-enables regardless of the gate");
        // "0" and sibling-exclusion force-disable REGARDLESS of the gate:
        assert!(!mmq_dtype_eligible("0", true, "q4k", &others), "\"0\" force-disables even when the gate is true");
        assert!(!mmq_dtype_eligible("0", false, "q4k", &others), "\"0\" force-disables when the gate is false too");
        assert!(!mmq_dtype_eligible("q5k", true, "q4k", &others), "a sibling's key excludes this dtype even when the gate is true");
        assert!(!mmq_dtype_eligible("q6k", true, "q4k", &others), "the OTHER sibling's key excludes this dtype too");
    }

    #[test]
    fn cuda_quantize_q8_1() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let el = 256;
        let el_padded = pad(el, MATRIX_ROW_PADDING);
        let y_size_in_bytes =
            el_padded * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
        let mut y_q8_1 = unsafe { dev.alloc::<u8>(y_size_in_bytes)? };
        let vs: Vec<f32> = (0..el).map(|v| v as f32).collect();
        let y = dev.clone_htod(&vs)?;
        quantize_q8_1(&y.as_view(), &mut y_q8_1, el, 1, &dev)?;
        Ok(())
    }

    #[test]
    fn cuda_mmv_q8_1() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let ncols = 256;
        let vs: Vec<f32> = (0..ncols).map(|v| v as f32).collect();
        let y = dev.clone_htod(&vs)?;
        let mut xs = QCudaStorage::zeros(&dev, ncols, GgmlDType::Q4_0)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(y.clone(), dev.clone()))?;
        let cuda_storage = mul_mat_vec_via_q8_1(
            &xs.data,
            &y.as_view(),
            /* dtype */ GgmlDType::Q4_0,
            /* ncols */ ncols,
            /* nrows */ 1,
            /* b_size */ 1,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let vs = dev.clone_dtoh(&vs.as_view())?;
        assert_eq!(vs.len(), 1);
        // for n = 255, n.(n+1).(2n+1) / 6 = 5559680
        // Q8 means 1/256 precision.
        assert_eq!(vs[0], 5561664.5);

        let cuda_storage = dequantize_mul_mat_vec(
            &xs.data,
            &y.as_view(),
            /* dtype */ GgmlDType::Q4_0,
            /* ncols */ ncols,
            /* nrows */ 1,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let vs = dev.clone_dtoh(&vs.as_view())?;
        assert_eq!(vs.len(), 1);
        assert_eq!(vs[0], 5561851.0);
        Ok(())
    }

    #[test]
    fn cuda_mm_q8_1() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let ncols = 256;
        let vs: Vec<f32> = (0..ncols * 4).map(|v| v as f32 / 4.).collect();
        let y = dev.clone_htod(&vs)?;
        let mut xs = QCudaStorage::zeros(&dev, ncols * 4, GgmlDType::Q4_0)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(y.clone(), dev.clone()))?;
        let cuda_storage = mul_mat_via_q8_1(
            &xs.data,
            &y.as_view(),
            /* dtype */ GgmlDType::Q4_0,
            /* x_rows */ 4,
            /* x_cols */ ncols,
            /* y_rows */ ncols,
            /* y_cols */ 4,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let vs = dev.clone_dtoh(&vs.as_view())?;

        /*
           x = torch.tensor([float(v) for v in range(1024)]).reshape(4, 256)
           x @ x.t() / 16
        tensor([[  347480.0000,   869720.0000,  1391960.0000,  1914200.0000],
                [  869720.0000,  2440536.0000,  4011352.0000,  5582166.5000],
                [ 1391960.0000,  4011352.0000,  6630742.0000,  9250132.0000],
                [ 1914200.0000,  5582166.5000,  9250132.0000, 12918099.0000]])
                */
        assert_eq!(vs.len(), 16);
        assert_eq!(vs[0], 347604.0);
        assert_eq!(vs[1], 888153.06);
        assert_eq!(vs[4], 869780.7);
        assert_eq!(vs[5], 2483145.0);
        assert_eq!(vs[11], 9407368.0);
        assert_eq!(vs[14], 9470856.0);
        assert_eq!(vs[15], 13138824.0);
        Ok(())
    }

    // The following test used to fail under compute-sanitizer until #2526.
    #[test]
    fn cuda_mm_q8_1_pad() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let (x_rows, ncols, y_cols) = (4, 16, 2048);
        let vs: Vec<f32> = (0..ncols * y_cols).map(|v| v as f32 / 256.).collect();
        let y = dev.clone_htod(&vs)?;
        let mut xs = QCudaStorage::zeros(&dev, ncols * x_rows, GgmlDType::Q4_0)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(y.clone(), dev.clone()))?;
        let cuda_storage = mul_mat_via_q8_1(
            &xs.data,
            &y.as_view(),
            /* dtype */ GgmlDType::Q4_0,
            /* x_rows */ x_rows,
            /* x_cols */ ncols,
            /* y_rows */ ncols,
            /* y_cols */ y_cols,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let _vs = dev.clone_dtoh(&vs.as_view())?;
        Ok(())
    }
}
