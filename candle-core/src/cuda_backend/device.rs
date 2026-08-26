use crate::backend::{BackendDevice, BackendStorage};
use crate::{CpuStorage, CpuStorageRef, DType, Layout, Result, Shape};
pub use candle_kernels as kernels;
pub use cudarc;
use cudarc::driver::CudaFunction;
use float8::F8E4M3;
use half::{bf16, f16};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use super::{CudaError, CudaStorage, CudaStorageSlice, WrapErr};

/// Unique identifier for cuda devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl DeviceId {
    fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

struct CudaRng(cudarc::curand::CudaRng);
unsafe impl Send for CudaRng {}

/// Device-resident `1.0`/`0.0` constants for cuBLAS's `alpha`/`beta` GEMM
/// coefficients, one pair per dtype this backend's matmul actually issues
/// coefficients for (f32, f16, bf16 -- f64 matmul is not used by any
/// consumer of this fork and is left on cuBLAS's default host pointer
/// mode, which is incompatible with the device-pointer mode this struct
/// exists to support).
///
/// # Why this exists (rainfall-one fork, not upstream candle-core)
///
/// Every call site in this module passes `alpha=1`, `beta=0` -- candle's
/// matmul never uses any other coefficient. Upstream candle-core takes
/// the ADDRESS OF A HOST-LOCAL COPY of those literals
/// (`&cfg.gemm.alpha as *const _`) and relies on cuBLAS's default
/// `CUBLAS_POINTER_MODE_HOST` to dereference it. Confirmed live
/// (rainfall-rajesh, 2026-08-26, on a GPU-passthrough VM after Webyne
/// fixed a mixed-guest-OS NVIDIA driver install): CUDA graph capture of
/// ANY cuBLAS matmul under HOST pointer mode fails with
/// `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`/`CUBLAS_STATUS_INTERNAL_ERROR`
/// on this environment, ruled out as buffer-address instability,
/// explicit-workspace allocation, capture mode (THREAD_LOCAL/RELAXED/GLOBAL),
/// and legacy-cuBLAS-vs-cuBLASLt -- `CUBLAS_POINTER_MODE_DEVICE` with
/// device-resident coefficients is the one change that fixes it. Since
/// the coefficient is always the same two constants, allocating them ONCE
/// at device construction (rather than per-call) avoids adding any
/// per-matmul allocation to the hot decode path this fork exists to speed
/// up in the first place.
pub(crate) struct CublasOneZero {
    pub(crate) f32_one: cudarc::driver::CudaSlice<f32>,
    pub(crate) f32_zero: cudarc::driver::CudaSlice<f32>,
    pub(crate) f16_one: cudarc::driver::CudaSlice<f16>,
    pub(crate) f16_zero: cudarc::driver::CudaSlice<f16>,
    pub(crate) bf16_one: cudarc::driver::CudaSlice<bf16>,
    pub(crate) bf16_zero: cudarc::driver::CudaSlice<bf16>,
    /// Tracks whether the handle is CURRENTLY in
    /// `CUBLAS_POINTER_MODE_DEVICE`, set/cleared by
    /// [`CudaDevice::with_cublas_device_pointer_mode`]. The gemm helper
    /// functions read this (a plain relaxed atomic load, no FFI) instead
    /// of querying cuBLAS's own handle state via `cublasGetPointerMode`
    /// on every single matmul call. Confirmed live 2026-08-26: an earlier
    /// version of this fix DID query `cublas.get_pointer_mode()` per call,
    /// and eager decode stayed pinned at ~162ms/token even after scoping
    /// DEVICE mode to only the capture window -- eager calls never enter
    /// that window at all, so the per-call FFI round-trip itself (not
    /// pointer mode) was the real regression, happening on every one of
    /// the many matmul calls per decode step (QKV/output projections,
    /// MoE FFN gate/up/down) regardless of which mode was ever active.
    pub(crate) device_pointer_mode_active: std::sync::atomic::AtomicBool,
}

impl CublasOneZero {
    fn new(stream: &Arc<cudarc::driver::CudaStream>) -> Result<Self> {
        Ok(Self {
            f32_one: stream.memcpy_stod(&[1f32]).w()?,
            f32_zero: stream.memcpy_stod(&[0f32]).w()?,
            f16_one: stream.memcpy_stod(&[f16::from_f32(1.0)]).w()?,
            f16_zero: stream.memcpy_stod(&[f16::from_f32(0.0)]).w()?,
            bf16_one: stream.memcpy_stod(&[bf16::from_f32(1.0)]).w()?,
            bf16_zero: stream.memcpy_stod(&[bf16::from_f32(0.0)]).w()?,
            device_pointer_mode_active: std::sync::atomic::AtomicBool::new(false),
        })
    }
}

pub struct ModuleStore {
    mdls: [Option<Arc<cudarc::driver::CudaModule>>; kernels::ALL_IDS.len()],
}

#[derive(Clone)]
pub struct CudaDevice {
    id: DeviceId,
    context: Arc<cudarc::driver::CudaContext>,
    modules: Arc<std::sync::RwLock<ModuleStore>>,
    custom_modules: Arc<std::sync::RwLock<HashMap<String, Arc<cudarc::driver::CudaModule>>>>,
    stream: Arc<cudarc::driver::CudaStream>,
    pub(crate) blas: Arc<cudarc::cublas::CudaBlas>,
    pub(crate) cublas_one_zero: Arc<CublasOneZero>,
    curand: Arc<Mutex<CudaRng>>,
    seed_value: Arc<RwLock<u64>>,
    /// A persistent, page-locked (pinned) host scratch buffer this device's
    /// [`CudaDevice::upload_shape_descriptor`] reuses for every small
    /// dims/strides array the CUDA backend's indexing/gather/scatter/
    /// elementwise kernels upload before every launch -- see that method's
    /// own doc comment for why this exists.
    shape_descriptor_pinned: Arc<Mutex<cudarc::driver::PinnedHostSlice<usize>>>,
}

/// Capacity (in `usize` elements) of [`CudaDevice::shape_descriptor_pinned`]
/// -- generous headroom over the largest `[dims..., strides...]` array any
/// op in this backend builds (typically under 8 elements for the highest-
/// rank tensors this workspace uses; nothing here approaches even a
/// quarter of this).
const SHAPE_DESCRIPTOR_SCRATCH_CAPACITY: usize = 64;

impl std::fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CudaDevice({:?})", self.id)
    }
}

impl CudaDevice {
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn alloc<T: cudarc::driver::DeviceRepr>(
        &self,
        len: usize,
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        self.stream.alloc::<T>(len).w()
    }

    pub fn alloc_zeros<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        self.stream.alloc_zeros::<T>(len).w()
    }

    pub fn memcpy_htod<
        T: cudarc::driver::DeviceRepr,
        Src: cudarc::driver::HostSlice<T> + ?Sized,
        Dst: cudarc::driver::DevicePtrMut<T>,
    >(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<()> {
        self.stream.memcpy_htod(src, dst).w()
    }

    pub fn clone_dtoh<T: cudarc::driver::DeviceRepr, Src: cudarc::driver::DevicePtr<T>>(
        &self,
        src: &Src,
    ) -> Result<Vec<T>> {
        self.stream.clone_dtoh(src).w()
    }

    pub fn memcpy_dtod<
        T,
        Src: cudarc::driver::DevicePtr<T>,
        Dst: cudarc::driver::DevicePtrMut<T>,
    >(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<()> {
        self.stream.memcpy_dtod(src, dst).w()
    }

    pub fn memcpy_dtoh<
        T: cudarc::driver::DeviceRepr,
        Src: cudarc::driver::DevicePtr<T>,
        Dst: cudarc::driver::HostSlice<T>,
    >(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<()> {
        self.stream.memcpy_dtoh(src, dst).w()
    }

    pub fn clone_htod<T: cudarc::driver::DeviceRepr, Src: cudarc::driver::HostSlice<T> + ?Sized>(
        &self,
        src: &Src,
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        self.stream.clone_htod(src).w()
    }
}

pub struct CudaFunc {
    func: CudaFunction,
    stream: Arc<cudarc::driver::CudaStream>,
}

impl std::ops::Deref for CudaFunc {
    type Target = CudaFunction;

    fn deref(&self) -> &Self::Target {
        &self.func
    }
}

impl CudaFunc {
    pub fn into_cuda_function(self) -> CudaFunction {
        self.func
    }
}

#[macro_export]
macro_rules! builder_arg {
    ($b:ident, $($arg:expr),*) => {
        $(
            let __arg = $arg;
            $b.arg(&__arg);
        )*
    };
}

impl CudaFunc {
    pub fn builder(&self) -> cudarc::driver::LaunchArgs<'_> {
        self.stream.launch_builder(&self.func)
    }
}

impl CudaDevice {
    pub fn cuda_stream(&self) -> Arc<cudarc::driver::CudaStream> {
        self.stream.clone()
    }

    /// When turned on, all cuda tensors **created after calling this function** will
    /// not track uses via cuda events.
    ///
    /// # Safety
    ///
    /// It is up to the user to ensure proper synchronization between multiple streams:
    /// - Ensure that no tensor is freed before a use on another stream is finished.
    /// - Ensure that a tensor is not used on another stream before allocation on the
    ///   allocating stream finishes.
    /// - Ensure that a tensor is not written two concurrently by multiple streams.
    pub unsafe fn disable_event_tracking(&self) {
        self.context.disable_event_tracking()
    }

    pub fn is_event_tracking(&self) -> bool {
        self.context.is_event_tracking()
    }

    #[cfg(all(feature = "ug", not(target_arch = "wasm32")))]
    pub fn compile(
        &self,
        func_name: &'static str,
        kernel: candle_ug::lang::ssa::Kernel,
    ) -> Result<CudaFunc> {
        let mut buf = vec![];
        candle_ug::cuda::code_gen::gen(&mut buf, func_name, &kernel)?;
        let cuda_code = String::from_utf8(buf)?;
        let opts = cudarc::nvrtc::CompileOptions {
            use_fast_math: Some(true),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::safe::compile_ptx_with_opts(cuda_code, opts).w()?;
        let module = self.context.load_module(ptx).w()?;
        let func = module.load_function(func_name).w()?;
        Ok(CudaFunc {
            func,
            stream: self.stream.clone(),
        })
    }

    pub fn id(&self) -> DeviceId {
        self.id
    }

    pub fn get_or_load_custom_func(
        &self,
        fn_name: &str,
        module_name: &str,
        ptx: &str,
    ) -> Result<CudaFunc> {
        let ms = self.custom_modules.read().unwrap();
        if let Some(mdl) = ms.get(module_name).as_ref() {
            let func = mdl.load_function(fn_name).w()?;
            return Ok(CudaFunc {
                func,
                stream: self.stream.clone(),
            });
        }
        drop(ms);
        let mut ms = self.custom_modules.write().unwrap();
        let cuda_module = self.context.load_module(ptx.into()).w()?;
        ms.insert(module_name.to_string(), cuda_module.clone());
        let func = cuda_module.load_function(fn_name).w()?;
        Ok(CudaFunc {
            func,
            stream: self.stream.clone(),
        })
    }

    pub fn get_or_load_func(&self, fn_name: &str, mdl: &kernels::Module) -> Result<CudaFunc> {
        let ms = self.modules.read().unwrap();
        if let Some(mdl) = ms.mdls[mdl.index()].as_ref() {
            let func = mdl.load_function(fn_name).w()?;
            return Ok(CudaFunc {
                func,
                stream: self.stream.clone(),
            });
        }
        drop(ms);
        let mut ms = self.modules.write().unwrap();
        let cuda_module = self.context.load_module(mdl.ptx().into()).w()?;
        ms.mdls[mdl.index()] = Some(cuda_module.clone());
        let func = cuda_module.load_function(fn_name).w()?;
        Ok(CudaFunc {
            func,
            stream: self.stream.clone(),
        })
    }

    pub fn cublas_handle(&self) -> Arc<cudarc::cublas::CudaBlas> {
        self.blas.clone()
    }
}

impl CudaDevice {
    pub fn new_with_stream(ordinal: usize) -> Result<Self> {
        let context = cudarc::driver::CudaContext::new(ordinal).w()?;
        let stream = context.new_stream().w()?;
        // rainfall-one fork (2026-08-26): disable cudarc's cross-stream
        // event tracking for this context's whole lifetime, immediately
        // after the one stream this device will ever use is created.
        //
        // Root cause, confirmed live: a captured forward pass failed at
        // its very first tensor read with CUDA_ERROR_STREAM_CAPTURE_ISOLATION
        // ("dependency created on uncaptured work in another stream"),
        // reproducing under every capture mode (THREAD_LOCAL, RELAXED).
        // cudarc's `CudaContext::is_managing_stream_synchronization()`
        // returns true as soon as `new_stream()` has been called even
        // ONCE (it flags "multi-stream mode" on the CALL COUNT, not on
        // concurrent multi-stream USE) -- and once true, EVERY
        // `DevicePtr::device_ptr()` call unconditionally inserts a
        // `cuStreamWaitEvent` for that slice's last-recorded write event
        // (`cudarc::driver::safe::core::CudaSlice::device_ptr`, no
        // same-stream skip). Every persistent tensor this device ever
        // allocates -- model weights, KV-cache blocks, the decode-token
        // buffer -- was written long before any capture attempt begins,
        // so that recorded event predates `cuStreamBeginCapture` by
        // construction. Waiting on it during capture is a dependency on
        // pre-capture (uncaptured) work, illegal under CUDA's capture
        // model regardless of capture mode and regardless of whether the
        // wait targets the SAME physical stream the event was recorded
        // on -- exactly the failure observed.
        //
        // Disabling event tracking is safe here specifically because
        // `new_with_stream` is the ONLY place this crate ever calls
        // `CudaContext::new_stream()` -- one stream is created, once, and
        // every kernel launch/cuBLAS/cuRAND call on this device reuses
        // that same `Arc<CudaStream>` for the rest of the process
        // (`CudaDevice::cuda_stream()`, `CudaFunc::stream`, `self.blas`,
        // `self.curand` are all built from this one `stream.clone()`).
        // cudarc's own safety contract for `disable_event_tracking()`
        // only requires the CALLER to preserve write-before-read and
        // free-after-last-use ordering across MULTIPLE concurrently-used
        // streams -- a condition this device never creates in the first
        // place, so there is nothing left for event tracking to protect
        // against; it exists purely to satisfy cudarc's "was
        // `new_stream()` ever called" heuristic, at the cost of breaking
        // graph capture for every tensor this device touches.
        //
        // # Safety
        // No second stream is ever created against `context` anywhere in
        // this codebase (grepped, both this crate and every downstream
        // consumer) -- see the comment above for why that is exactly
        // cudarc's own stated precondition for this call being sound.
        unsafe {
            context.disable_event_tracking();
        }
        // rainfall-one fork (2026-08-26): a persistent pinned host buffer
        // for CudaDevice::upload_shape_descriptor -- see that method's own
        // doc comment for the capture-illegal pattern it exists to replace
        // (dev.clone_htod() from plain, PAGEABLE host Vecs, present at 15+
        // call sites across this crate's own CUDA op-dispatch machinery,
        // confirmed live 2026-08-26 as the cause of
        // CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED on the very first captured
        // op -- Tensor::index_select). Allocated once here, well before any
        // capture attempt; `alloc_pinned` is a context-level resource
        // allocation (page-locked host memory), never legal to call DURING
        // capture, which is exactly why this buffer is built once at
        // device construction and reused for the device's entire lifetime
        // rather than allocated fresh per call.
        let shape_descriptor_pinned = Arc::new(Mutex::new(
            unsafe { context.alloc_pinned::<usize>(SHAPE_DESCRIPTOR_SCRATCH_CAPACITY) }.w()?,
        ));
        let blas = cudarc::cublas::CudaBlas::new(stream.clone()).w()?;
        // rainfall-one fork: device-pointer coefficients for CUDA-graph-
        // capturable matmul -- see CublasOneZero's own doc comment.
        // Deliberately NOT setting CUBLAS_POINTER_MODE_DEVICE here: cuBLAS
        // documented (and confirmed live, 2026-08-26 -- eager decode
        // measured 162ms/token under permanent DEVICE mode vs 47ms/token
        // under HOST mode, the same ~3.4x regression this workspace
        // already independently measured for CUDARC_DISABLE_ASYNC_ALLOC)
        // fast paths for alpha=1/beta=0 rely on those being COMPILE-TIME
        // host constants; DEVICE mode disables them for every matmul, not
        // just captured ones. Pointer mode instead stays HOST (cuBLAS's
        // own default) for ordinary eager execution, and is toggled to
        // DEVICE only for the duration of an explicit capture via
        // CudaDevice::with_cublas_device_pointer_mode -- see that method's
        // own doc comment.
        let cublas_one_zero = CublasOneZero::new(&stream)?;
        let curand = cudarc::curand::CudaRng::new(299792458, stream.clone()).w()?;
        let module_store = ModuleStore {
            mdls: [const { None }; kernels::ALL_IDS.len()],
        };
        Ok(Self {
            id: DeviceId::new(),
            context,
            stream,
            blas: Arc::new(blas),
            cublas_one_zero: Arc::new(cublas_one_zero),
            curand: Arc::new(Mutex::new(CudaRng(curand))),
            modules: Arc::new(std::sync::RwLock::new(module_store)),
            custom_modules: Arc::new(std::sync::RwLock::new(HashMap::new())),
            seed_value: Arc::new(RwLock::new(299792458)),
            shape_descriptor_pinned,
        })
    }

    /// Upload a small `dims`/`strides`-style descriptor array to the
    /// device, returning a freshly-allocated `CudaSlice<usize>` -- the
    /// Compute Unified Device Architecture (CUDA) graph capture-safe
    /// replacement for `self.stream.clone_htod(values)`, which every
    /// indexing/gather/scatter/broadcast kernel dispatch in this crate's
    /// CUDA backend (`cuda_backend/mod.rs`, 15+ call sites as of this
    /// writing) calls once per launch to upload that launch's own
    /// `[dims..., strides...]` array.
    ///
    /// `clone_htod` allocates its destination via the stream-ordered pool
    /// allocator (itself capture-safe -- confirmed by reading this
    /// backend's own device-allocation code, which already routes through
    /// `cuMemAllocAsync` whenever async allocation is enabled, the default
    /// here), so the allocation was never the problem. The COPY is:
    /// `cuMemcpyHtoDAsync` from ordinary Rust heap memory (a `Vec` built
    /// fresh on the host every call) is only genuinely asynchronous when
    /// its HOST-side source is page-locked; from plain (pageable) memory
    /// the driver falls back to a synchronous copy, and any synchronous
    /// operation issued to a stream mid-capture returns
    /// `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED` -- confirmed live
    /// 2026-08-26 as the exact failure on the first captured op
    /// (`Tensor::index_select`'s `IndexSelect::f`).
    ///
    /// This method stages `values` through [`CudaDevice`]'s persistent
    /// `shape_descriptor_pinned` buffer (page-locked, allocated once at
    /// device construction, long before any capture) instead: a plain
    /// host-side `copy_nonoverlapping` (not a CUDA call at all, always
    /// legal) followed by a genuinely asynchronous `cuMemcpyHtoDAsync`
    /// from that pinned buffer to a freshly-allocated (pool-allocator,
    /// capture-safe) device slice. Deliberately bypasses `PinnedHostSlice`'s
    /// own `HostSlice` trait impl (`clone_htod(&pinned_slice)`) rather than
    /// using it directly: that impl unconditionally issues a
    /// `cuStreamWaitEvent` on the slice's own tracking event on every use,
    /// which -- for a REUSED buffer whose event was last recorded by a
    /// prior, pre-capture call -- is itself a dependency on uncaptured
    /// work, the same `CUDA_ERROR_STREAM_CAPTURE_ISOLATION` class this
    /// fork's `disable_event_tracking` call already eliminated for
    /// ordinary tensors. The raw driver call used here carries no such
    /// tracking.
    ///
    /// # Panics
    /// If `values.len()` exceeds [`SHAPE_DESCRIPTOR_SCRATCH_CAPACITY`] --
    /// every real call site in this crate stays well under that; a caller
    /// exceeding it indicates a new op shape this constant needs raising
    /// for, not a data condition to recover from.
    pub(crate) fn upload_shape_descriptor(&self, values: &[usize]) -> Result<cudarc::driver::CudaSlice<usize>> {
        assert!(
            values.len() <= SHAPE_DESCRIPTOR_SCRATCH_CAPACITY,
            "shape descriptor of {} elements exceeds the {SHAPE_DESCRIPTOR_SCRATCH_CAPACITY}-element pinned scratch buffer",
            values.len(),
        );
        let mut pinned = self
            .shape_descriptor_pinned
            .lock()
            .expect("shape_descriptor_pinned mutex poisoned");
        // `PinnedHostSlice`'s own fields are private to cudarc, so its
        // public `as_mut_ptr()` is the only way to reach the underlying
        // pointer -- it also calls the buffer's own (never-recorded, since
        // this method never goes through `HostSlice::stream_synced_slice`)
        // event's `synchronize()` first, a plain host-thread block that
        // returns immediately for an event with no pending work and does
        // not touch the stream being captured, so it costs nothing here
        // and is legal during capture regardless.
        let ptr = pinned.as_mut_ptr().w()?;
        // SAFETY: `ptr` is valid, page-locked memory for
        // SHAPE_DESCRIPTOR_SCRATCH_CAPACITY elements (allocated in
        // `new_with_stream`/`new` and never freed before this device is
        // dropped); `values.len()` is asserted not to exceed that above.
        // A plain host-to-host memory write, not a CUDA API call -- legal
        // unconditionally, including during an active graph capture.
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr(), ptr, values.len());
        }
        // SAFETY: the same region just written, read back immediately
        // afterward on this same thread -- no concurrent host mutation is
        // possible while `pinned`'s mutex guard is held.
        let src = unsafe { std::slice::from_raw_parts(ptr, values.len()) };
        let mut dst = unsafe { self.stream.alloc::<usize>(values.len()) }.w()?;
        // SAFETY: `dst` was just allocated with exactly `values.len()`
        // elements of type `usize`, matching `src`'s length and `T`; `dst`
        // was allocated on `self.stream`, the same stream this copy is
        // issued to.
        let (dst_ptr, _record_dst) =
            cudarc::driver::DevicePtrMut::device_ptr_mut(&mut dst, &self.stream);
        unsafe { cudarc::driver::result::memcpy_htod_async(dst_ptr, src, self.stream.cu_stream()) }.w()?;
        Ok(dst)
    }

    /// Run `f` with this device's cuBLAS handle in
    /// `CUBLAS_POINTER_MODE_DEVICE` (device-resident alpha/beta, required
    /// for CUDA graph capture of matmul -- see [`CublasOneZero`]'s own doc
    /// comment for why HOST mode, cuBLAS's default and what every OTHER
    /// call on this device uses, cannot be captured). Pointer mode is
    /// restored to HOST unconditionally after `f` returns, success or
    /// error -- eager (non-captured) matmul on this device depends on
    /// HOST mode's compile-time alpha=1/beta=0 fast path (confirmed live
    /// 2026-08-26: permanent DEVICE mode cost eager decode a measured
    /// ~3.4x slowdown), so this must never be left toggled on.
    ///
    /// # Errors
    /// Propagates `f`'s own error. If the pointer-mode reset back to HOST
    /// itself fails, that failure is logged to stderr rather than
    /// propagated (mirroring this fork's `run_matmul_workspace_repro`
    /// diagnostic's own established discipline for restoring shared
    /// cuBLAS handle state) -- `f`'s result is still the return value.
    /// Deliberately generic over `f`'s own error type `E`, not this
    /// crate's own `candle_core::Error` -- a caller doing raw capture work
    /// (like this fork's own `run_and_log`/`try_capture_and_replay`
    /// diagnostic, which reports failures as plain `String`s) should not
    /// need a `candle_core::Error` conversion just to use this helper.
    /// Both `set_pointer_mode` calls are non-fatal by design (logged, not
    /// propagated as `E`) -- there is no meaningful way to convert a
    /// `CublasError` into an arbitrary caller-supplied `E` without a
    /// `From` bound this method deliberately avoids requiring; a failure
    /// entering DEVICE mode surfaces naturally anyway, as `f`'s own error
    /// when its captured matmul dereferences a host address as if it were
    /// a device pointer.
    pub fn with_cublas_device_pointer_mode<R, E>(&self, f: impl FnOnce() -> std::result::Result<R, E>) -> std::result::Result<R, E> {
        use cudarc::cublas::sys::cublasPointerMode_t;
        use std::sync::atomic::Ordering;
        if let Err(e) = self.blas.set_pointer_mode(cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE) {
            eprintln!(
                "CudaDevice::with_cublas_device_pointer_mode: WARNING -- failed to set \
                 DEVICE pointer mode: {e} (the closure's own matmul will likely fail \
                 with a host-address-as-device-pointer error instead)"
            );
        }
        // Relaxed: this flag only needs to be visible to gemm calls issued
        // by the SAME thread inside `f` (matching CU_STREAM_CAPTURE_MODE_
        // THREAD_LOCAL's own same-thread assumption) -- no cross-thread
        // synchronization requirement to uphold.
        self.cublas_one_zero.device_pointer_mode_active.store(true, Ordering::Relaxed);
        let result = f();
        self.cublas_one_zero.device_pointer_mode_active.store(false, Ordering::Relaxed);
        if let Err(e) = self.blas.set_pointer_mode(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST) {
            eprintln!(
                "CudaDevice::with_cublas_device_pointer_mode: WARNING -- pointer mode \
                 reset-to-HOST failed: {e} (cuBLAS handle may be left in a bad state \
                 for subsequent eager matmul on this device)"
            );
        }
        result
    }
}

impl BackendDevice for CudaDevice {
    type Storage = CudaStorage;

    fn new(ordinal: usize) -> Result<Self> {
        let context = cudarc::driver::CudaContext::new(ordinal).w()?;
        let stream = context.default_stream();
        // rainfall-one fork: required field, see `CudaDevice::new_with_stream`'s
        // own extensive comment on `shape_descriptor_pinned` -- this
        // constructor (the default per-thread-stream path, never called by
        // Cerebra; see `rpc::device::inference_device`'s own doc comment
        // for why) has no capture-safety requirement of its own, but the
        // field must still be populated for every `CudaDevice` instance.
        let shape_descriptor_pinned = Arc::new(Mutex::new(
            unsafe { context.alloc_pinned::<usize>(SHAPE_DESCRIPTOR_SCRATCH_CAPACITY) }.w()?,
        ));
        let blas = cudarc::cublas::CudaBlas::new(stream.clone()).w()?;
        // rainfall-one fork: device-pointer coefficients for CUDA-graph-
        // capturable matmul -- see CublasOneZero's own doc comment.
        // Deliberately NOT setting CUBLAS_POINTER_MODE_DEVICE here: cuBLAS
        // documented (and confirmed live, 2026-08-26 -- eager decode
        // measured 162ms/token under permanent DEVICE mode vs 47ms/token
        // under HOST mode, the same ~3.4x regression this workspace
        // already independently measured for CUDARC_DISABLE_ASYNC_ALLOC)
        // fast paths for alpha=1/beta=0 rely on those being COMPILE-TIME
        // host constants; DEVICE mode disables them for every matmul, not
        // just captured ones. Pointer mode instead stays HOST (cuBLAS's
        // own default) for ordinary eager execution, and is toggled to
        // DEVICE only for the duration of an explicit capture via
        // CudaDevice::with_cublas_device_pointer_mode -- see that method's
        // own doc comment.
        let cublas_one_zero = CublasOneZero::new(&stream)?;
        let curand = cudarc::curand::CudaRng::new(299792458, stream.clone()).w()?;
        let module_store = ModuleStore {
            mdls: [const { None }; kernels::ALL_IDS.len()],
        };
        Ok(Self {
            id: DeviceId::new(),
            context,
            stream,
            blas: Arc::new(blas),
            cublas_one_zero: Arc::new(cublas_one_zero),
            curand: Arc::new(Mutex::new(CudaRng(curand))),
            modules: Arc::new(std::sync::RwLock::new(module_store)),
            custom_modules: Arc::new(std::sync::RwLock::new(HashMap::new())),
            seed_value: Arc::new(RwLock::new(299792458)),
            shape_descriptor_pinned,
        })
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        // We do not call set_seed but instead create a new curand object. This ensures that the
        // state will be identical and the same random numbers will be generated.
        let mut curand = self.curand.lock().unwrap();
        curand.0 = cudarc::curand::CudaRng::new(seed, self.stream.clone()).w()?;
        *self.seed_value.write().unwrap() = seed;
        Ok(())
    }

    fn get_current_seed(&self) -> Result<u64> {
        Ok(*self.seed_value.read().unwrap())
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Cuda {
            gpu_id: self.context.ordinal(),
        }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        self.id == rhs.id
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<CudaStorage> {
        let elem_count = shape.elem_count();
        let slice = match dtype {
            DType::U8 => {
                let data = self.alloc_zeros::<u8>(elem_count)?;
                CudaStorageSlice::U8(data)
            }
            DType::U32 => {
                let data = self.alloc_zeros::<u32>(elem_count)?;
                CudaStorageSlice::U32(data)
            }
            DType::I16 => {
                let data = self.alloc_zeros::<i16>(elem_count)?;
                CudaStorageSlice::I16(data)
            }
            DType::I32 => {
                let data = self.alloc_zeros::<i32>(elem_count)?;
                CudaStorageSlice::I32(data)
            }
            DType::I64 => {
                let data = self.alloc_zeros::<i64>(elem_count)?;
                CudaStorageSlice::I64(data)
            }
            DType::BF16 => {
                let data = self.alloc_zeros::<bf16>(elem_count)?;
                CudaStorageSlice::BF16(data)
            }
            DType::F16 => {
                let data = self.alloc_zeros::<f16>(elem_count)?;
                CudaStorageSlice::F16(data)
            }
            DType::F32 => {
                let data = self.alloc_zeros::<f32>(elem_count)?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let data = self.alloc_zeros::<f64>(elem_count)?;
                CudaStorageSlice::F64(data)
            }
            DType::F8E4M3 => {
                let data = self.alloc_zeros::<F8E4M3>(elem_count)?;
                CudaStorageSlice::F8E4M3(data)
            }
            DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                return Err(
                    CudaError::InternalError("Dummy types not supported in CUDA backend").into(),
                )
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, lo: f64, up: f64) -> Result<CudaStorage> {
        let elem_count = shape.elem_count();
        let curand = self.curand.lock().unwrap();
        let slice = match dtype {
            // TODO: Add support for F16 and BF16 though this is likely to require some upstream
            // cudarc changes.
            DType::U8
            | DType::U32
            | DType::I16
            | DType::I32
            | DType::I64
            | DType::F16
            | DType::BF16 => Err(CudaError::UnsupportedDtype {
                dtype,
                op: "rand_uniform",
            })
            .w()?,
            DType::F32 => {
                let mut data = unsafe { self.alloc::<f32>(elem_count)? };
                curand.0.fill_with_uniform(&mut data).w()?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let mut data = unsafe { self.alloc::<f64>(elem_count)? };
                curand.0.fill_with_uniform(&mut data).w()?;
                CudaStorageSlice::F64(data)
            }
            DType::F8E4M3 | DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                Err(CudaError::UnsupportedDtype {
                    dtype,
                    op: "rand_uniform",
                })
                .w()?
            }
        };
        let slice = if lo == 0. && up == 1.0 {
            slice
        } else {
            use super::utils::Map1;
            let layout = Layout::contiguous(shape);
            super::Affine(up - lo, lo).map(&slice, self, &layout)?
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn rand_normal(&self, shape: &Shape, dtype: DType, mean: f64, std: f64) -> Result<CudaStorage> {
        // TODO: Add support for F16 and BF16 though this is likely to require some upstream
        // cudarc changes.
        let elem_count = shape.elem_count();
        let curand = self.curand.lock().unwrap();
        // curand can only generate an odd number of values.
        // https://github.com/huggingface/candle/issues/734
        let elem_count_round = if elem_count % 2 == 1 {
            elem_count + 1
        } else {
            elem_count
        };
        let slice = match dtype {
            DType::U8
            | DType::U32
            | DType::I16
            | DType::I32
            | DType::I64
            | DType::F16
            | DType::BF16 => Err(CudaError::UnsupportedDtype {
                dtype,
                op: "rand_normal",
            })
            .w()?,
            DType::F32 => {
                let mut data = unsafe { self.alloc::<f32>(elem_count_round)? };
                curand
                    .0
                    .fill_with_normal(&mut data, mean as f32, std as f32)
                    .w()?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let mut data = unsafe { self.alloc::<f64>(elem_count_round)? };
                curand.0.fill_with_normal(&mut data, mean, std).w()?;
                CudaStorageSlice::F64(data)
            }
            DType::F8E4M3 | DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                Err(CudaError::UnsupportedDtype {
                    dtype,
                    op: "rand_normal",
                })
                .w()?
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();
        let slice = match dtype {
            DType::U8 => {
                let data = self.alloc::<u8>(elem_count)?;
                CudaStorageSlice::U8(data)
            }
            DType::U32 => {
                let data = self.alloc::<u32>(elem_count)?;
                CudaStorageSlice::U32(data)
            }
            DType::I16 => {
                let data = self.alloc::<i16>(elem_count)?;
                CudaStorageSlice::I16(data)
            }
            DType::I32 => {
                let data = self.alloc::<i32>(elem_count)?;
                CudaStorageSlice::I32(data)
            }
            DType::I64 => {
                let data = self.alloc::<i64>(elem_count)?;
                CudaStorageSlice::I64(data)
            }
            DType::BF16 => {
                let data = self.alloc::<bf16>(elem_count)?;
                CudaStorageSlice::BF16(data)
            }
            DType::F16 => {
                let data = self.alloc::<f16>(elem_count)?;
                CudaStorageSlice::F16(data)
            }
            DType::F32 => {
                let data = self.alloc::<f32>(elem_count)?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let data = self.alloc::<f64>(elem_count)?;
                CudaStorageSlice::F64(data)
            }
            DType::F8E4M3 => {
                let data = self.alloc::<F8E4M3>(elem_count)?;
                CudaStorageSlice::F8E4M3(data)
            }
            DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                return Err(
                    CudaError::InternalError("Dummy types not supported in CUDA backend").into(),
                )
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_slice<T: crate::WithDType>(&self, s: &[T]) -> Result<Self::Storage> {
        let slice = match T::cpu_storage_ref(s) {
            CpuStorageRef::U8(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::U8(data)
            }
            CpuStorageRef::U32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::U32(data)
            }
            CpuStorageRef::I16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I16(data)
            }
            CpuStorageRef::I32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I32(data)
            }
            CpuStorageRef::I64(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I64(data)
            }
            CpuStorageRef::BF16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::BF16(data)
            }
            CpuStorageRef::F16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F16(data)
            }
            CpuStorageRef::F32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F32(data)
            }
            CpuStorageRef::F64(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F64(data)
            }
            CpuStorageRef::F8E4M3(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F8E4M3(data)
            }
            CpuStorageRef::F4(_)
            | CpuStorageRef::F6E2M3(_)
            | CpuStorageRef::F6E3M2(_)
            | CpuStorageRef::F8E8M0(_) => {
                return Err(CudaError::UnsupportedDtype {
                    dtype: T::DTYPE,
                    op: "storage_from_slice",
                }
                .into());
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<CudaStorage> {
        let slice = match storage {
            CpuStorage::U8(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::U8(data)
            }
            CpuStorage::U32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::U32(data)
            }
            CpuStorage::I16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I16(data)
            }
            CpuStorage::I32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I32(data)
            }
            CpuStorage::I64(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I64(data)
            }
            CpuStorage::BF16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::BF16(data)
            }
            CpuStorage::F16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F16(data)
            }
            CpuStorage::F32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F32(data)
            }
            CpuStorage::F64(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F64(data)
            }
            CpuStorage::F8E4M3(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F8E4M3(data)
            }
            CpuStorage::F4(_)
            | CpuStorage::F6E2M3(_)
            | CpuStorage::F6E3M2(_)
            | CpuStorage::F8E8M0(_) => {
                return Err(CudaError::UnsupportedDtype {
                    dtype: storage.dtype(),
                    op: "storage_from_cpu_storage",
                }
                .into());
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<CudaStorage> {
        let slice = match storage {
            CpuStorage::U8(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::U8(data)
            }
            CpuStorage::U32(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::U32(data)
            }
            CpuStorage::I16(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::I16(data)
            }
            CpuStorage::I32(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::I32(data)
            }
            CpuStorage::I64(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::I64(data)
            }
            CpuStorage::BF16(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::BF16(data)
            }
            CpuStorage::F16(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::F16(data)
            }
            CpuStorage::F32(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::F32(data)
            }
            CpuStorage::F64(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::F64(data)
            }
            CpuStorage::F8E4M3(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::F8E4M3(data)
            }
            CpuStorage::F4(_)
            | CpuStorage::F6E2M3(_)
            | CpuStorage::F6E3M2(_)
            | CpuStorage::F8E8M0(_) => {
                return Err(CudaError::UnsupportedDtype {
                    dtype: storage.dtype(),
                    op: "storage_from_cpu_storage_owned",
                }
                .into());
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn synchronize(&self) -> Result<()> {
        self.stream.synchronize().map_err(crate::Error::wrap)?;
        Ok(())
    }
}
