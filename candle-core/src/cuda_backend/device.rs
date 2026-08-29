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

/// Fixed-size bump-allocation arena backing every device allocation made
/// during a CUDA-graph capture pass -- see [`CudaDevice::alloc_mode`]'s own
/// doc comment for why this exists (candle's per-op fresh-allocation model
/// is otherwise incompatible with capture on this project's target
/// hardware, both for the sync allocator, illegal mid-capture per CUDA's
/// own rules, and separately for cudarc's async pool allocator, confirmed
/// broken even in eager mode here).
///
/// Built from a dry-run pass's recorded sizes ([`CudaDevice::measured_sizes`])
/// -- one real allocation (`backing`, owned normally, freed normally when
/// this struct drops), then every per-op "allocation" during the actual
/// capture pass becomes a non-owning [`cudarc::driver::CudaStream::upgrade_arena_offset`]
/// view into it instead of a real `cuMemAlloc`/`cuMemAllocAsync` call.
struct CaptureArena {
    /// Owns the one real allocation this arena bump-allocates into. Never
    /// read directly after construction (its raw pointer is cached in
    /// `base_ptr` once, up front) -- kept alive here purely so its `Drop`
    /// frees the backing memory when this arena itself is dropped.
    _backing: cudarc::driver::CudaSlice<u8>,
    base_ptr: cudarc::driver::sys::CUdeviceptr,
    total_bytes: usize,
    cursor: std::sync::atomic::AtomicUsize,
}

/// Rounds `bytes` up to the next multiple of 256 -- CUDA's own typical
/// device-memory alignment granularity, generous enough for every dtype
/// this backend allocates (up to `f64`, 8-byte aligned) with headroom for
/// any op that benefits from wider alignment (e.g. vectorized loads).
fn arena_align(bytes: usize) -> usize {
    (bytes + 255) & !255
}

/// Clears `CudaDevice::capturing_thread` back to `None` when dropped --
/// guarantees this happens on EVERY exit from
/// [`CudaDevice::with_capture_arena`] (an early return on `dry_run`
/// failure, normal completion, or an unwinding panic from either
/// closure), not just the happy path a hand-written reset at each return
/// site could easily miss one of.
struct CapturingThreadGuard<'a>(&'a Mutex<Option<std::thread::ThreadId>>);

impl Drop for CapturingThreadGuard<'_> {
    fn drop(&mut self) {
        *self.0.lock().unwrap() = None;
    }
}

impl CaptureArena {
    /// Bump-allocate `len` elements of `T` (`bytes = len *
    /// size_of::<T>()`) from this arena. Callers (`CudaDevice::alloc`/
    /// `alloc_zeros`) are responsible for requesting sizes in the SAME
    /// order every capture pass -- see `alloc_mode`'s doc for why that
    /// invariant holds (decode at `seq_len == 1` has a fully static op
    /// sequence).
    ///
    /// # Errors
    /// If the arena is exhausted (the live pass requested more total
    /// bytes than the dry run that sized it did) -- this should never
    /// happen if the dry run and the real capture pass ran the exact same
    /// code path, and indicates that invariant was violated somewhere.
    fn bump_alloc<T: cudarc::driver::DeviceRepr>(
        &self,
        stream: &Arc<cudarc::driver::CudaStream>,
        len: usize,
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        let bytes = len * std::mem::size_of::<T>();
        let aligned = arena_align(bytes);
        let offset = self.cursor.fetch_add(aligned, std::sync::atomic::Ordering::SeqCst);
        if std::env::var("CEREBRA_ARENA_DEBUG").is_ok() {
            eprintln!("ARENA_DEBUG real_capture bump_alloc bytes={bytes} aligned={aligned} offset={offset}");
        }
        if offset + bytes > self.total_bytes {
            crate::bail!(
                "CaptureArena exhausted: requested offset {offset} + {bytes} bytes exceeds \
                 arena size {} bytes -- the live capture pass allocated more than the dry run \
                 that sized this arena did, meaning the decode step's allocation sequence is \
                 NOT static as assumed (see CudaDevice::alloc_mode's doc comment)",
                self.total_bytes
            );
        }
        Ok(unsafe { stream.upgrade_arena_offset::<T>(self.base_ptr, offset, len) })
    }
}

/// What [`CudaDevice::alloc`]/[`CudaDevice::alloc_zeros`] do on this call --
/// checked with one relaxed atomic load per call, so the default (`Normal`)
/// hot path this project has spent this whole session optimizing pays
/// negligible extra cost.
///
/// # Why this exists (rainfall-one fork, not upstream candle-core)
///
/// nsys profiling (2026-08-27) showed Cerebra's decode step is
/// launch-bound, not compute-bound: real GPU kernel execution across a
/// full decode request measured ~23ms/token, while the SAME window's
/// host-side kernel-launch dispatch overhead (`cuLaunchKernel` +
/// `cudaLaunchKernel` + `cudaLaunchKernelExC`) measured almost exactly as
/// much (~3,500+ launches per token). CUDA graph capture is the standard
/// fix (replays one pre-recorded launch sequence instead of dispatching
/// thousands of individual kernels) -- both vLLM (PyTorch's graph-safe
/// pooling allocator) and llama.cpp (ggml's static pre-planned compute
/// arena) use it for exactly this reason, and neither can do so with a
/// framework that allocates a fresh output tensor per op the way candle's
/// CUDA backend does by default (`alloc`/`alloc_zeros`/`alloc_uninit`/
/// `zeros_impl` all funnel through this same choke point) -- illegal
/// mid-capture for the sync allocator, and separately confirmed broken
/// even in EAGER mode on this project's specific GPU-passthrough hardware
/// for cudarc's async pool allocator (see `CUBLAS_POINTER_MODE_DEVICE`'s
/// own doc comment above for the sibling ~3.4x-regression finding that
/// established this "permanently-on breaks eager" pattern).
///
/// `Measuring` and `Arena` are both scoped, temporary states (entered via
/// [`CudaDevice::with_capture_arena`], never left active outside that
/// closure) -- mirroring [`CudaDevice::with_cublas_device_pointer_mode`]'s
/// own established "toggle on only for the capture window, always restore
/// afterward" discipline for exactly the same reason: leaving either mode
/// on permanently would break every OTHER (non-decode-step) allocation on
/// this device.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
enum AllocMode {
    /// Default. Every allocation goes through the real
    /// `cudarc::driver::CudaStream::alloc`/`alloc_zeros` exactly as
    /// upstream candle-core does -- zero behavior change from stock.
    Normal = 0,
    /// A dry-run pass: allocations still go through the REAL allocator
    /// (so the dry run itself produces a correct, usable result -- it is
    /// a genuine forward pass, not a no-op), but every requested size is
    /// ALSO appended to `CudaDevice::measured_sizes`, in call order, to
    /// size the arena the next `Arena`-mode pass will bump-allocate into.
    Measuring = 1,
    /// The actual capture pass: allocations bump-allocate into
    /// `CudaDevice::capture_arena` instead of calling the real allocator
    /// at all (illegal mid-capture for the sync path; this is what makes
    /// capture legal in the first place).
    Arena = 2,
}

/// The lazily-created side stream and its fork/join events — see
/// `CudaDevice::stream_override`'s doc. One per device, reused for every
/// sequential fork/join pair.
struct SideBranch {
    stream: Arc<cudarc::driver::CudaStream>,
    fork_event: cudarc::driver::CudaEvent,
    join_event: cudarc::driver::CudaEvent,
}

#[derive(Clone)]
pub struct CudaDevice {
    id: DeviceId,
    context: Arc<cudarc::driver::CudaContext>,
    modules: Arc<std::sync::RwLock<ModuleStore>>,
    custom_modules: Arc<std::sync::RwLock<HashMap<String, Arc<cudarc::driver::CudaModule>>>>,
    stream: Arc<cudarc::driver::CudaStream>,
    /// Optional SIDE stream override (rainfall-one, 2026-08-29): while
    /// `Some`, every stream-consuming path in this backend that goes
    /// through [`Self::active_stream`] — kernel launches via
    /// `get_or_load(_custom)_func`, alloc/memcpy/memset — issues on the
    /// side stream instead of the main one. Entered/left via
    /// [`Self::fork_side_branch`] / [`Self::pause_side_branch`] /
    /// [`Self::join_side_branch`], which fence the two streams with
    /// events so the fork/join edges are real dependencies — inside a
    /// CUDA graph capture this records the side branch as PARALLEL graph
    /// nodes (multi-stream capture via event fork/join, the documented
    /// pattern), letting an independent branch (e.g. a Mixture of
    /// Experts layer's shared expert) execute concurrently with the main
    /// chain on replay. Shared across device clones (one `Arc`), like
    /// every other piece of this device's state.
    stream_override: Arc<RwLock<Option<Arc<cudarc::driver::CudaStream>>>>,
    /// The lazily-created side stream itself plus the two fork/join
    /// events, created once and reused for every fork (event-handle
    /// reuse across sequential fork/join pairs is well-defined both live
    /// and under capture — each record produces its own graph node).
    side_branch: Arc<Mutex<Option<SideBranch>>>,
    pub(crate) blas: Arc<cudarc::cublas::CudaBlas>,
    pub(crate) cublas_one_zero: Arc<CublasOneZero>,
    curand: Arc<Mutex<CudaRng>>,
    seed_value: Arc<RwLock<u64>>,
    /// See [`AllocMode`]'s own doc comment. Checked with a relaxed atomic
    /// load on every `alloc`/`alloc_zeros` call.
    alloc_mode: Arc<std::sync::atomic::AtomicU8>,
    /// Populated only while `alloc_mode` is `Measuring`; drained and
    /// cleared by [`CudaDevice::with_capture_arena`] once the dry run
    /// completes, to build the arena those sizes describe.
    measured_sizes: Arc<Mutex<Vec<usize>>>,
    /// Built fresh by every [`CudaDevice::with_capture_arena`] call, and
    /// then DELIBERATELY LEFT ALIVE after that call returns -- the
    /// captured graph's kernels, and any tensor the caller's `capture`
    /// closure returned, hold pointers into this arena's backing memory,
    /// and remain valid to replay/read only as long as it stays
    /// allocated (confirmed live 2026-08-27: freeing it immediately on
    /// return produced `CUDA_ERROR_ILLEGAL_ADDRESS` on the very next
    /// graph replay or output read). A later `with_capture_arena` call
    /// legitimately replaces it (the assignment inside that method drops
    /// the previous one) -- at most one captured graph's arena is ever
    /// alive at a time, but it is NOT torn down merely because the
    /// call that built it returned.
    capture_arena: Arc<Mutex<Option<CaptureArena>>>,
    /// A fixed cuBLAS workspace, set once via
    /// [`CudaDevice::ensure_cublas_workspace`] and kept alive for the
    /// device's lifetime. Without it, cuBLAS/cublasLt allocate their
    /// workspace via `cudaMallocAsync` PER CALL — measured live
    /// (rainfall-one, 2026-08-28, `cuGraphGetNodes` audit of a captured
    /// Cerebra decode step): 112 MEM_ALLOC + 112 MEM_FREE nodes in the
    /// captured graph, all library-internal, which limits the graph to a
    /// single instantiated exec (`CUDA_ERROR_NOT_SUPPORTED` on any
    /// second `cuGraphInstantiate`) and therefore blocks multi-exec
    /// pipelined replay. Pre-setting the workspace is the same fix
    /// llama.cpp applies for graph-captured decode.
    cublas_workspace: Arc<Mutex<Option<cudarc::driver::CudaSlice<u8>>>>,
    /// Guards against a real concurrent allocation racing a capture
    /// window. `alloc_mode` alone is not enough to make `Measuring`/
    /// `Arena` mode safe under concurrency: it is a single device-wide
    /// flag, so ANY thread calling `alloc`/`alloc_zeros` on this device
    /// while another thread's `with_capture_arena` is active would
    /// observe the SAME `Measuring`/`Arena` state and be wrongly routed
    /// through it too (measured into the wrong dry run, or bump-allocated
    /// into an arena sized for someone else's decode step) -- caught on
    /// self-review 2026-08-27, before this ever reached production
    /// traffic.
    ///
    /// `alloc`/`alloc_zeros` hold a cheap SHARED (read) guard for the
    /// duration of one call; [`Self::with_capture_arena`] holds an
    /// EXCLUSIVE (write) guard for its ENTIRE duration (both the dry run
    /// and the real capture pass). A real concurrent allocation therefore
    /// either completes fully before a capture window can start (holding
    /// its read guard blocks the write guard from being acquired), or
    /// blocks until the capture window finishes (the write guard blocks
    /// all new read guards) -- it can never observe `Measuring`/`Arena`
    /// mode without ALSO being the capture window's own two closures.
    /// `RwLock` over a plain `Mutex` specifically because `alloc`/
    /// `alloc_zeros` are this device's hottest path (this project's own
    /// nsys profiling found ~3,500+ calls per decode token) -- an
    /// uncontended `RwLock` read acquisition costs about the same as the
    /// `alloc_mode` atomic load it sits beside, not meaningfully more.
    capture_gate: Arc<RwLock<()>>,
    /// The thread currently executing inside [`Self::with_capture_arena`]
    /// (its `dry_run`/`capture` closures), if any -- `None` at every
    /// other time. `std::sync::RwLock` is not reentrant: a thread already
    /// holding `capture_gate`'s write lock must NOT also try to acquire
    /// its read lock (undefined behavior per the standard library's own
    /// documentation, deadlocks in practice), which is exactly what
    /// `alloc`/`alloc_zeros` would otherwise do when called (as they
    /// always are) from INSIDE `dry_run`/`capture` -- caught on
    /// self-review immediately after `capture_gate` was added, before
    /// this ever reached CI. `alloc`/`alloc_zeros` check this field ONLY
    /// on the already-rare `alloc_mode != Normal` path (the common
    /// `Normal`-mode hot path never touches this lock at all) to tell
    /// "this call is the SAME thread that owns the current capture
    /// window, no gate needed" apart from "this is a genuinely
    /// independent concurrent caller that happens to observe a capture in
    /// progress, which DOES need to block on `capture_gate` like any
    /// other Normal-mode caller."
    capturing_thread: Arc<Mutex<Option<std::thread::ThreadId>>>,
    /// Pinned host buffers pre-allocated by [`Self::with_capture_arena`]'s
    /// dry-run pass for [`Self::clone_htod_capture_safe`]'s real capture
    /// pass to reuse -- see that method's own doc comment. Cleared at the
    /// start of every `with_capture_arena` call.
    pinned_scratch: Arc<Mutex<Vec<cudarc::driver::PinnedHostSlice<usize>>>>,
    /// Read/reset by [`Self::clone_htod_capture_safe`] and
    /// [`Self::with_capture_arena`] -- indexes into `pinned_scratch` in
    /// the same order the dry run populated it, so the real capture
    /// pass's Nth call reuses the dry run's Nth buffer.
    pinned_scratch_cursor: Arc<std::sync::atomic::AtomicUsize>,
}

impl std::fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CudaDevice({:?})", self.id)
    }
}

impl CudaDevice {
    /// `true` iff the calling thread is the one currently executing
    /// inside [`Self::with_capture_arena`] -- see `capturing_thread`'s own
    /// doc comment for why this matters (distinguishing "I am the
    /// capture window's own dry-run/capture closure" from "I am some
    /// other, genuinely concurrent caller that merely observes
    /// `alloc_mode != Normal`").
    fn is_current_thread_capturing(&self) -> bool {
        *self.capturing_thread.lock().unwrap() == Some(std::thread::current().id())
    }

    /// Records `bytes` in `self.measured_sizes` -- called only from
    /// `alloc`/`alloc_zeros`'s already-verified-`Measuring`-mode,
    /// already-verified-same-thread path.
    fn record_measured(&self, bytes: usize) {
        self.measured_sizes.lock().unwrap().push(bytes);
    }

    /// Bump-allocates from `self.capture_arena` -- called only from
    /// `alloc`/`alloc_zeros`'s already-verified-`Arena`-mode,
    /// already-verified-same-thread path.
    fn arena_alloc<T: cudarc::driver::DeviceRepr>(&self, len: usize) -> Result<cudarc::driver::CudaSlice<T>> {
        let guard = self.capture_arena.lock().unwrap();
        let arena = guard
            .as_ref()
            .expect("CudaDevice::alloc_mode is Arena but capture_arena is None -- with_capture_arena's own invariant was violated");
        arena.bump_alloc::<T>(&self.active_stream(), len)
    }

    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn alloc<T: cudarc::driver::DeviceRepr>(
        &self,
        len: usize,
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        let mode = self.alloc_mode.load(std::sync::atomic::Ordering::Relaxed);
        if mode != AllocMode::Normal as u8 && self.is_current_thread_capturing() {
            // Same thread as `with_capture_arena`'s own write-lock holder
            // -- MUST NOT also acquire `capture_gate`'s read lock here
            // (RwLock is not reentrant; would deadlock against the write
            // lock this same thread already holds).
            return if mode == AllocMode::Arena as u8 {
                self.arena_alloc::<T>(len)
            } else {
                self.record_measured(len * std::mem::size_of::<T>());
                self.active_stream().alloc::<T>(len).w()
            };
        }
        // Either genuinely Normal, or a DIFFERENT thread observing a
        // capture window that is not its own -- block on the gate (waits
        // out any active capture window, exactly like every other
        // Normal-mode caller must), then allocate for real. Once this
        // read guard is held, `alloc_mode` is guaranteed `Normal` -- no
        // writer can be active concurrently with any held read guard.
        let _gate = self.capture_gate.read().unwrap();
        self.active_stream().alloc::<T>(len).w()
    }

    pub fn alloc_zeros<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        let mode = self.alloc_mode.load(std::sync::atomic::Ordering::Relaxed);
        if mode != AllocMode::Normal as u8 && self.is_current_thread_capturing() {
            return if mode == AllocMode::Arena as u8 {
                let mut slice = self.arena_alloc::<T>(len)?;
                self.active_stream().memset_zeros(&mut slice).w()?;
                Ok(slice)
            } else {
                self.record_measured(len * std::mem::size_of::<T>());
                self.active_stream().alloc_zeros::<T>(len).w()
            };
        }
        let _gate = self.capture_gate.read().unwrap();
        self.active_stream().alloc_zeros::<T>(len).w()
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
        self.active_stream().memcpy_htod(src, dst).w()
    }

    pub fn clone_dtoh<T: cudarc::driver::DeviceRepr, Src: cudarc::driver::DevicePtr<T>>(
        &self,
        src: &Src,
    ) -> Result<Vec<T>> {
        self.active_stream().clone_dtoh(src).w()
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
        self.active_stream().memcpy_dtod(src, dst).w()
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
        self.active_stream().memcpy_dtoh(src, dst).w()
    }

    pub fn clone_htod<T: cudarc::driver::DeviceRepr, Src: cudarc::driver::HostSlice<T> + ?Sized>(
        &self,
        src: &Src,
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        self.active_stream().clone_htod(src).w()
    }

    /// Upload `data` (always shape/stride metadata in this file -- every
    /// real call site passes `usize`) to the device -- identical to
    /// [`Self::clone_htod`] outside a capture window (every normal eager
    /// call). During [`Self::with_capture_arena`]'s two passes this takes
    /// a different, capture-safe path, split across them:
    ///
    /// - **Dry run** (`AllocMode::Measuring`, not actually inside
    ///   `cuStreamBeginCapture`/`end_capture` -- only the REAL capture
    ///   pass is): allocates a fresh [`cudarc::driver::PinnedHostSlice`]
    ///   sized exactly for this call, uploads from it (legal: not
    ///   mid-capture), and CACHES it in `pinned_scratch` for the real
    ///   capture pass below to reuse.
    /// - **Real capture** (`AllocMode::Arena`, genuinely mid-capture):
    ///   reuses the NEXT cached buffer from the dry run, in the same call
    ///   order -- writes fresh data into it and uploads, allocating
    ///   nothing new. This two-phase design exists because `clone_htod`
    ///   from ordinary `Vec`-backed (pageable) host memory is illegal
    ///   mid-capture (confirmed live 2026-08-27,
    ///   `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED` from `index_select`'s
    ///   own shape/stride upload, present at every one of this file's
    ///   small-metadata uploads: `SlicePtrOrNull::params_from_layout`,
    ///   `IndexSelect`, `Gather`, `Scatter`/`ScatterAdd`, the ternary
    ///   `WhereCond` op, strided binary-op dispatch) -- and a NAIVE fix
    ///   (allocate a fresh pinned buffer unconditionally whenever
    ///   capturing) hit the IDENTICAL error, also confirmed live: pinned
    ///   host allocation itself (`cuMemHostAlloc`) is ALSO illegal
    ///   mid-capture on this hardware, not just the plain pageable
    ///   upload it was meant to replace. Pre-allocating during the
    ///   (uncaptured) dry run and only WRITING into an already-allocated
    ///   buffer during the real capture sidesteps both illegal calls.
    ///
    /// # Errors
    /// Propagates any `candle_core` error from the pinned allocation, the
    /// write into it, or the device copy itself.
    ///
    /// During the real capture pass, a pool exhausted at the current
    /// index or an entry-length mismatch against `data` -- both meaning
    /// the sizing plan (dry run or cached) and the real capture pass took
    /// genuinely different code paths -- returns a typed error (NOT a
    /// panic: a panic here holds the capture gate's write lock and would
    /// poison it, wedging every later allocation on the device --
    /// confirmed live 2026-08-28). The capture closure then fails
    /// cleanly and the caller falls back to eager decode.
    pub fn clone_htod_capture_safe(&self, data: &[usize]) -> Result<cudarc::driver::CudaSlice<usize>> {
        let mode = self.alloc_mode.load(std::sync::atomic::Ordering::Relaxed);
        if mode == AllocMode::Measuring as u8 && self.is_current_thread_capturing() {
            // SAFETY: written in full immediately below, before any
            // device operation ever reads it.
            let mut pinned = unsafe { self.alloc_pinned::<usize>(data.len()) }?;
            pinned.as_mut_slice().map_err(crate::Error::wrap)?.copy_from_slice(data);
            // Device side via `self.alloc` (NOT `clone_htod`'s internal
            // `stream.alloc`) so the allocation is measured into the
            // arena plan -- the real capture pass below must bump-alloc
            // this same buffer from the arena, and an unmeasured raw
            // stream alloc there would instead record a MEM_ALLOC graph
            // node (confirmed live 2026-08-28 via the node audit: 71
            // such nodes, one per metadata upload, each limiting the
            // captured graph to a single instantiated exec).
            // SAFETY: written in full by the memcpy below before any
            // device operation reads it.
            let mut dst = unsafe { self.alloc::<usize>(data.len()) }?;
            self.memcpy_htod(&pinned, &mut dst)?;
            self.pinned_scratch.lock().unwrap().push(pinned);
            return Ok(dst);
        }
        if mode == AllocMode::Arena as u8 && self.is_current_thread_capturing() {
            let idx = self.pinned_scratch_cursor.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut guard = self.pinned_scratch.lock().unwrap();
            let pool_len = guard.len();
            // Typed errors, NOT panics (changed 2026-08-28): a panic here
            // while the capture gate's write lock is held poisons that
            // lock and wedges every later allocation on the device
            // (confirmed live: one mismatched presized capture killed the
            // whole service). An error unwinds cleanly through the
            // capture closure instead, letting the caller fall back to
            // eager decode and invalidate any cached arena plan.
            let Some(pinned) = guard.get_mut(idx) else {
                crate::bail!(
                    "CudaDevice::clone_htod_capture_safe: pinned scratch pool exhausted at \
                     index {idx} (pool has {pool_len} entries) -- the sizing plan and the real \
                     capture pass made a different number of clone_htod_capture_safe calls"
                )
            };
            if pinned.len() != data.len() {
                crate::bail!(
                    "CudaDevice::clone_htod_capture_safe: pinned scratch pool entry {idx} has \
                     length {} but this call needs {} -- the sizing plan and the real capture \
                     pass requested different sizes at the same call-order position",
                    pinned.len(),
                    data.len()
                );
            }
            // SAFETY: this exact pool entry was allocated during the dry
            // run (AllocMode::Measuring, above) and has only ever been
            // written by that dry run's own `clone_htod` upload -- by the
            // time the real capture pass reaches here, that upload
            // completed synchronously (Measuring mode is never inside an
            // active capture, so its own `clone_htod`'s implicit ordering
            // is real, not deferred). No device operation reads or writes
            // this pinned buffer's memory between then and now.
            let ptr = unsafe { pinned.as_mut_ptr_unsynchronized() };
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len()) };
            // Device side bump-allocated from the arena (`self.alloc` in
            // Arena mode) + an explicit pinned-to-device memcpy -- the
            // memcpy records as a plain (multi-exec-safe) memcpy node,
            // where `clone_htod`'s internal `stream.alloc` would record
            // a MEM_ALLOC node and cap the graph at one exec. The
            // borrow-then-copy split keeps `pinned`'s `&mut` borrow from
            // the pool guard alive only for the memcpy itself.
            // SAFETY: written in full by this recorded memcpy (replayed
            // on every launch) before the step's kernels read it.
            let mut dst = unsafe { self.alloc::<usize>(data.len()) }?;
            self.active_stream().memcpy_htod(&*pinned, &mut dst).w()?;
            return Ok(dst);
        }
        self.clone_htod(data)
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

    /// The stream every stream-consuming path in this backend should
    /// issue on RIGHT NOW: the side stream while a
    /// [`Self::fork_side_branch`] scope is active, the main stream
    /// otherwise. See `stream_override`'s field doc.
    fn active_stream(&self) -> Arc<cudarc::driver::CudaStream> {
        if let Some(side) = self.stream_override.read().unwrap().as_ref() {
            return side.clone();
        }
        self.stream.clone()
    }

    /// Begin issuing subsequent work on this device's SIDE stream, after
    /// fencing it behind everything already issued on the main stream
    /// (event fork). Inside a CUDA graph capture this records the side
    /// work as a PARALLEL branch of the graph. Pair with
    /// [`Self::pause_side_branch`] (stop redirecting, no fence — main
    /// work issued after it runs CONCURRENTLY with the side branch) and
    /// [`Self::join_side_branch`] (main stream waits for the side
    /// branch). Sequential fork/join pairs reuse one side stream and one
    /// event pair — well-defined live and under capture.
    pub fn fork_side_branch(&self) -> Result<()> {
        let mut guard = self.side_branch.lock().unwrap();
        if guard.is_none() {
            *guard = Some(SideBranch {
                stream: self.context.new_stream().w()?,
                fork_event: self.context.new_event(None).w()?,
                join_event: self.context.new_event(None).w()?,
            });
        }
        let branch = guard.as_ref().unwrap();
        branch.fork_event.record(&self.stream).w()?;
        branch.stream.wait(&branch.fork_event).w()?;
        *self.stream_override.write().unwrap() = Some(branch.stream.clone());
        Ok(())
    }

    /// Stop redirecting work to the side stream WITHOUT fencing — work
    /// issued on the main stream after this call runs concurrently with
    /// the side branch until [`Self::join_side_branch`].
    pub fn pause_side_branch(&self) {
        *self.stream_override.write().unwrap() = None;
    }

    /// Fence the main stream behind the side branch (event join). Also
    /// clears the redirect if still active.
    pub fn join_side_branch(&self) -> Result<()> {
        *self.stream_override.write().unwrap() = None;
        let guard = self.side_branch.lock().unwrap();
        let Some(branch) = guard.as_ref() else {
            return Ok(());
        };
        branch.join_event.record(&branch.stream).w()?;
        self.stream.wait(&branch.join_event).w()?;
        Ok(())
    }

    /// `true` iff this device's stream currently has an active CUDA graph
    /// capture in progress (`CU_STREAM_CAPTURE_STATUS_ACTIVE`). Any
    /// operation that internally does a `clone_htod`/`memcpy_htod` from
    /// ordinary (pageable) host memory must check this before doing so --
    /// the driver silently makes that copy SYNCHRONOUS on pageable
    /// memory, illegal mid-capture (`CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`,
    /// confirmed live 2026-08-27 via `index_select`'s own small
    /// shape/stride metadata upload) -- see [`Self::alloc_pinned`] for
    /// the capture-safe alternative. Returns `false` (not an error) if
    /// the capture-status query itself fails, matching this file's
    /// existing convention of treating stream-state queries as
    /// best-effort diagnostics, not fatal.
    pub fn is_capturing(&self) -> bool {
        matches!(
            self.stream.capture_status(),
            Ok(cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE)
        )
    }

    /// Allocates page-locked (pinned) host memory -- see
    /// [`cudarc::driver::PinnedHostSlice`]'s own doc comment. Unlike
    /// ordinary `Vec`-backed host memory, a `clone_htod`/`memcpy_htod`
    /// from a `PinnedHostSlice` is genuinely asynchronous (stream-waits
    /// on an event rather than blocking the CPU), making it safe to call
    /// mid-capture -- see [`Self::is_capturing`]'s own doc comment for
    /// why that distinction matters.
    ///
    /// # Safety
    /// The returned memory is uninitialized -- the caller must write it
    /// before any device operation reads it.
    pub unsafe fn alloc_pinned<T: cudarc::driver::DeviceRepr>(
        &self,
        len: usize,
    ) -> Result<cudarc::driver::PinnedHostSlice<T>> {
        self.context.alloc_pinned::<T>(len).w()
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
            stream: self.active_stream(),
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
                stream: self.active_stream(),
            });
        }
        drop(ms);
        let mut ms = self.custom_modules.write().unwrap();
        let cuda_module = self.context.load_module(ptx.into()).w()?;
        ms.insert(module_name.to_string(), cuda_module.clone());
        let func = cuda_module.load_function(fn_name).w()?;
        Ok(CudaFunc {
            func,
            stream: self.active_stream(),
        })
    }

    pub fn get_or_load_func(&self, fn_name: &str, mdl: &kernels::Module) -> Result<CudaFunc> {
        let ms = self.modules.read().unwrap();
        if let Some(mdl) = ms.mdls[mdl.index()].as_ref() {
            let func = mdl.load_function(fn_name).w()?;
            return Ok(CudaFunc {
                func,
                stream: self.active_stream(),
            });
        }
        drop(ms);
        let mut ms = self.modules.write().unwrap();
        let cuda_module = self.context.load_module(mdl.ptx().into()).w()?;
        ms.mdls[mdl.index()] = Some(cuda_module.clone());
        let func = cuda_module.load_function(fn_name).w()?;
        Ok(CudaFunc {
            func,
            stream: self.active_stream(),
        })
    }

    pub fn cublas_handle(&self) -> Arc<cudarc::cublas::CudaBlas> {
        self.blas.clone()
    }

    /// Allocate a fixed cuBLAS workspace of `bytes` once (idempotent —
    /// later calls with any size are no-ops once one is set) and point
    /// the device's cuBLAS handle at it via `cublasSetWorkspace_v2`.
    /// The buffer stays alive for the device's lifetime.
    ///
    /// Why (see the `cublas_workspace` field doc for the measurements):
    /// without an explicit workspace, cuBLAS/cublasLt `cudaMallocAsync`
    /// their workspace per call; inside a CUDA-graph capture each of
    /// those becomes a MEM_ALLOC/MEM_FREE node pair, and a graph
    /// containing memory nodes can only ever have ONE instantiated
    /// exec — blocking multi-exec pipelined replay. MUST be called
    /// OUTSIDE any capture window (it allocates for real).
    pub fn ensure_cublas_workspace(&self, bytes: usize) -> Result<()> {
        use cudarc::driver::DevicePtr;
        let mut guard = self.cublas_workspace.lock().unwrap();
        if guard.is_some() {
            return Ok(());
        }
        let workspace = unsafe { self.stream.alloc::<u8>(bytes) }.w()?;
        {
            let (ptr, _sync) = workspace.device_ptr(&self.stream);
            // SAFETY: `workspace` is a live device allocation of exactly
            // `bytes` bytes, kept alive by the `Some(...)` store below
            // for the device's (and therefore the handle's) lifetime.
            let status = unsafe {
                cudarc::cublas::sys::cublasSetWorkspace_v2(
                    *self.blas.handle(),
                    ptr as usize as *mut core::ffi::c_void,
                    bytes,
                )
            };
            if status != cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                return Err(crate::Error::Msg(format!(
                    "cublasSetWorkspace_v2 failed: {status:?}"
                )));
            }
        }
        *guard = Some(workspace);
        Ok(())
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
            stream_override: Arc::new(RwLock::new(None)),
            side_branch: Arc::new(Mutex::new(None)),
            blas: Arc::new(blas),
            cublas_one_zero: Arc::new(cublas_one_zero),
            curand: Arc::new(Mutex::new(CudaRng(curand))),
            modules: Arc::new(std::sync::RwLock::new(module_store)),
            custom_modules: Arc::new(std::sync::RwLock::new(HashMap::new())),
            seed_value: Arc::new(RwLock::new(299792458)),
            alloc_mode: Arc::new(std::sync::atomic::AtomicU8::new(AllocMode::Normal as u8)),
            measured_sizes: Arc::new(Mutex::new(Vec::new())),
            capture_arena: Arc::new(Mutex::new(None)),
            cublas_workspace: Arc::new(Mutex::new(None)),
            capture_gate: Arc::new(RwLock::new(())),
            capturing_thread: Arc::new(Mutex::new(None)),
            pinned_scratch: Arc::new(Mutex::new(Vec::new())),
            pinned_scratch_cursor: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        })
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

    /// Runs `dry_run` once in `AllocMode::Measuring` (real allocations,
    /// but every requested size is also recorded in call order), builds a
    /// [`CaptureArena`] sized to exactly what that pass requested, then
    /// runs `capture` once in `AllocMode::Arena` (allocations bump-
    /// allocate into the arena instead of calling the real allocator at
    /// all -- the change that makes CUDA graph capture legal on this
    /// device: no `cuMemAlloc`/`cuMemAllocAsync` call happens during
    /// `capture`, only kernel launches and bump-pointer arithmetic).
    ///
    /// `dry_run` and `capture` **must** run the exact same code path --
    /// same op sequence, same tensor shapes -- or the arena will be
    /// undersized (`CaptureArena::bump_alloc` returns a typed error if
    /// so, rather than corrupting memory: it never reuses/overwrites a
    /// byte range already handed out this pass). In practice both
    /// closures are typically the SAME call (e.g. `|| model.forward(...)`
    /// for the same static decode-step shape) invoked twice; kept as two
    /// separate closures rather than one `FnMut` called twice so a caller
    /// can discard the dry run's own output while keeping the capture
    /// pass's, without needing the result to be meaningfully reusable
    /// across both calls.
    ///
    /// Always restores `alloc_mode` to `Normal` before returning --
    /// mirroring [`Self::with_cublas_device_pointer_mode`]'s own "never
    /// leave a capture-only mode active outside its own closure"
    /// discipline: every OTHER allocation on this device (a concurrent
    /// eager request, a later unrelated capture attempt) depends on this
    /// device defaulting back to real allocation. `capture_arena` itself,
    /// however, is deliberately LEFT populated -- see its own field doc
    /// comment for why (the captured graph and `capture`'s own returned
    /// tensor hold pointers into it, valid only as long as it stays
    /// allocated).
    ///
    /// # Errors
    /// The outer [`Result`] is this crate's own error type, for failures
    /// in the arena-scoping machinery itself (currently only the arena's
    /// one real backing allocation, if the dry run's total measured size
    /// exceeds available device memory). The inner
    /// `std::result::Result<R, E>` is `dry_run`'s or `capture`'s own
    /// result, unwrapped by the caller exactly as if they had called
    /// either closure directly -- if `dry_run` returns `Err`, `capture`
    /// never runs (there is nothing to size the arena from) and that
    /// `Err` is returned immediately as `Ok(Err(..))`.
    /// Build (or replace) the capture arena from an explicit list of
    /// allocation sizes -- the shared tail of [`Self::with_capture_arena`]
    /// and [`Self::with_capture_arena_presized`]. Caller must hold the
    /// capture gate's write lock.
    fn install_capture_arena(&self, sizes: &[usize]) -> Result<()> {
        use std::sync::atomic::Ordering;
        let total_bytes: usize = sizes.iter().map(|&b| arena_align(b)).sum();
        // SAFETY: the arena's own backing bytes are immediately handed to
        // `CaptureArena`, which never exposes them as `T`-typed data
        // until a `bump_alloc` view is constructed over a sub-range a
        // caller is about to write into -- matching every other
        // `alloc`/`alloc_uninit` caller's own established contract in
        // this file (uninitialized until written).
        let backing = unsafe { self.stream.alloc::<u8>(total_bytes.max(1)) }.w()?;
        let base_ptr = {
            use cudarc::driver::DevicePtr;
            backing.device_ptr(&self.stream).0
        };
        *self.capture_arena.lock().unwrap() = Some(CaptureArena {
            _backing: backing,
            base_ptr,
            total_bytes,
            cursor: std::sync::atomic::AtomicUsize::new(0),
        });
        self.pinned_scratch_cursor.store(0, Ordering::Relaxed);
        self.alloc_mode.store(AllocMode::Arena as u8, Ordering::Relaxed);
        Ok(())
    }

    /// The full arena plan measured by the most recent successful
    /// [`Self::with_capture_arena`] dry run: `(arena allocation sizes,
    /// pinned-scratch entry lengths)` -- retrievable until the next
    /// capture call clears them. Cache these (keyed by whatever
    /// determines the capture's allocation shapes) and hand them back to
    /// [`Self::with_capture_arena_presized`] to skip a later,
    /// shape-identical capture's whole dry-run pass. Both halves are
    /// needed: the arena sizes plan the device bump allocator, and the
    /// pinned lengths rebuild the metadata-upload staging pool
    /// (`clone_htod_capture_safe`'s dry-run-built pool -- confirmed live
    /// 2026-08-28: presizing the arena alone left that pool empty, and
    /// the capture pass panicked at its first metadata upload).
    pub fn capture_arena_plan(&self) -> (Vec<usize>, Vec<usize>) {
        let arena_sizes = self.measured_sizes.lock().unwrap().clone();
        let pinned_sizes = self.pinned_scratch.lock().unwrap().iter().map(|p| p.len()).collect();
        (arena_sizes, pinned_sizes)
    }

    /// [`Self::with_capture_arena`] minus the dry run: builds the arena
    /// and the pinned-scratch staging pool straight from a previous
    /// shape-identical capture's [`Self::capture_arena_plan`] and runs
    /// `capture` inside them. If the plan under-provisions the real
    /// capture's allocations, the arena's `bump_alloc` (or the pinned
    /// pool's own exhaustion check) surfaces that as `capture`'s error
    /// or panic -- the caller should invalidate its cache and fall back
    /// to the measuring path. Same locking, mode transitions, and
    /// leave-the-arena-alive semantics as the measuring variant.
    ///
    /// # Errors
    /// Outer [`Result`]: arena-machinery failures (the backing or pinned
    /// allocations). Inner result: `capture`'s own.
    pub fn with_capture_arena_presized<R, E>(
        &self,
        arena_sizes: &[usize],
        pinned_sizes: &[usize],
        capture: impl FnOnce() -> std::result::Result<R, E>,
    ) -> Result<std::result::Result<R, E>> {
        use std::sync::atomic::Ordering;
        let _gate = self.capture_gate.write().unwrap();
        *self.capturing_thread.lock().unwrap() = Some(std::thread::current().id());
        let _capturing_thread_guard = CapturingThreadGuard(&self.capturing_thread);

        // Rebuild the metadata staging pool at the recorded lengths.
        // Content is irrelevant: the capture pass overwrites each entry
        // with its own metadata before the async upload reads it.
        {
            let mut pool = self.pinned_scratch.lock().unwrap();
            pool.clear();
            for &len in pinned_sizes {
                // SAFETY: overwritten in full by the capture pass's own
                // copy_from_slice before any device operation reads it.
                pool.push(unsafe { self.alloc_pinned::<usize>(len) }?);
            }
        }
        self.pinned_scratch_cursor.store(0, Ordering::Relaxed);
        self.install_capture_arena(arena_sizes)?;

        let capture_result = capture();

        self.alloc_mode.store(AllocMode::Normal as u8, Ordering::Relaxed);
        // The arena stays alive after return -- same contract as
        // `with_capture_arena` (see that function's trailing comment).
        Ok(capture_result)
    }

    pub fn with_capture_arena<R, E>(
        &self,
        dry_run: impl FnOnce() -> std::result::Result<R, E>,
        capture: impl FnOnce() -> std::result::Result<R, E>,
    ) -> Result<std::result::Result<R, E>> {
        use std::sync::atomic::Ordering;

        // Exclusive for this call's ENTIRE duration (dry run + arena
        // build + real capture pass) -- blocks until every in-flight
        // `alloc`/`alloc_zeros` call elsewhere finishes (each holds a
        // shared guard), and blocks any NEW one from starting until this
        // function returns. See `capture_gate`'s own doc comment for why
        // `alloc_mode` alone cannot make `Measuring`/`Arena` mode safe
        // under concurrency without this.
        let _gate = self.capture_gate.write().unwrap();
        *self.capturing_thread.lock().unwrap() = Some(std::thread::current().id());
        let _capturing_thread_guard = CapturingThreadGuard(&self.capturing_thread);

        self.measured_sizes.lock().unwrap().clear();
        self.pinned_scratch.lock().unwrap().clear();
        self.pinned_scratch_cursor.store(0, Ordering::Relaxed);
        self.alloc_mode.store(AllocMode::Measuring as u8, Ordering::Relaxed);
        let dry_run_result = dry_run();
        self.alloc_mode.store(AllocMode::Normal as u8, Ordering::Relaxed);
        if dry_run_result.is_err() {
            return Ok(dry_run_result);
        }

        let sizes = std::mem::take(&mut *self.measured_sizes.lock().unwrap());
        if std::env::var("CEREBRA_ARENA_DEBUG").is_ok() {
            eprintln!("ARENA_DEBUG dry_run sizes ({}): {:?}", sizes.len(), sizes);
        }
        self.install_capture_arena(&sizes)?;
        // Leave the measured sizes retrievable after this call
        // ([`Self::capture_arena_measured_sizes`]) so a caller can cache
        // them and skip the whole dry-run pass on a later, shape-identical
        // capture via [`Self::with_capture_arena_presized`]. `record_measured`
        // only appends in `Measuring` mode, so re-storing them here cannot
        // be corrupted by ordinary allocations, and the next capture's own
        // `clear()` above resets them.
        *self.measured_sizes.lock().unwrap() = sizes;

        let capture_result = capture();

        self.alloc_mode.store(AllocMode::Normal as u8, Ordering::Relaxed);
        // Deliberately NOT tearing down `capture_arena` here -- caught
        // live 2026-08-27: the captured graph's kernels, and any tensor
        // `capture` itself returned, reference pointers INTO this
        // arena's backing memory. Freeing it as soon as this function
        // returns (the original design) left every subsequent
        // `graph.launch()` replay, and any attempt to read `capture`'s
        // own returned tensor, dereferencing already-freed memory --
        // manifested as `CUDA_ERROR_ILLEGAL_ADDRESS` on the very first
        // post-capture read. The arena must outlive `with_capture_arena`
        // itself, for as long as the caller keeps using the graph it
        // captured -- calling `with_capture_arena` again later
        // legitimately replaces it (a fresh `Some(CaptureArena {...})`
        // assignment above drops the previous one), matching "at most
        // one captured graph's arena alive at a time," never "torn down
        // the instant the capturing call returns."

        Ok(capture_result)
    }
}

impl BackendDevice for CudaDevice {
    type Storage = CudaStorage;

    fn new(ordinal: usize) -> Result<Self> {
        let context = cudarc::driver::CudaContext::new(ordinal).w()?;
        let stream = context.default_stream();
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
            stream_override: Arc::new(RwLock::new(None)),
            side_branch: Arc::new(Mutex::new(None)),
            blas: Arc::new(blas),
            cublas_one_zero: Arc::new(cublas_one_zero),
            curand: Arc::new(Mutex::new(CudaRng(curand))),
            modules: Arc::new(std::sync::RwLock::new(module_store)),
            custom_modules: Arc::new(std::sync::RwLock::new(HashMap::new())),
            seed_value: Arc::new(RwLock::new(299792458)),
            alloc_mode: Arc::new(std::sync::atomic::AtomicU8::new(AllocMode::Normal as u8)),
            measured_sizes: Arc::new(Mutex::new(Vec::new())),
            capture_arena: Arc::new(Mutex::new(None)),
            cublas_workspace: Arc::new(Mutex::new(None)),
            capture_gate: Arc::new(RwLock::new(())),
            capturing_thread: Arc::new(Mutex::new(None)),
            pinned_scratch: Arc::new(Mutex::new(Vec::new())),
            pinned_scratch_cursor: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
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

#[cfg(test)]
mod arena_tests {
    use super::arena_align;

    /// [Unit] Rounds up to the next 256-byte boundary; an already-aligned
    /// size is left unchanged.
    #[test]
    fn unit_arena_align_rounds_up_to_256() {
        assert_eq!(arena_align(0), 0);
        assert_eq!(arena_align(1), 256);
        assert_eq!(arena_align(255), 256);
        assert_eq!(arena_align(256), 256);
        assert_eq!(arena_align(257), 512);
        assert_eq!(arena_align(4096), 4096);
    }
}
