#![allow(unused)]
use super::GgmlDType;
use crate::{CudaDevice, CudaStorage, Error, Result, Tensor};

pub struct QCudaStorage {
    dtype: GgmlDType,
    device: CudaDevice,
}

impl QCudaStorage {
    pub fn zeros(_: &CudaDevice, _: usize, _: GgmlDType) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    pub fn dequantize(&self, _elem_count: usize) -> Result<CudaStorage> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn dequantize_f16(&self, _elem_count: usize) -> Result<CudaStorage> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn quantize(&mut self, _src: &CudaStorage) -> Result<()> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn quantize_imatrix(
        &mut self,
        _src: &CudaStorage,
        _imatrix_weights: &[f32],
        _n_per_row: usize,
    ) -> Result<()> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn quantize_imatrix_onto(
        &mut self,
        _src: &crate::CpuStorage,
        _imatrix_weights: &[f32],
        _n_per_row: usize,
    ) -> Result<()> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn quantize_onto(&mut self, _src: &crate::CpuStorage) -> Result<()> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        0
    }

    pub fn fwd(
        &self,
        _self_shape: &crate::Shape,
        _storage: &CudaStorage,
        _layout: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    /// Stub kept in sync with the real (`cuda.rs`) implementation's
    /// signature (2026-09-01, Goal-2500 item-3b gate fix): the real
    /// implementation returns a `Tensor` view into a persistent
    /// workspace, not a fresh `(CudaStorage, Shape)`, since the
    /// 2026-09-01 alias-bug fix -- this stub's return type must match
    /// or every non-CUDA build (Mac dev machines, the `cargo nt` gate)
    /// fails to compile regardless of whether the caller ever runs it.
    pub fn indexed_moe_forward(
        &self,
        _: &crate::Shape,
        _: &CudaStorage,
        _: &crate::Layout,
        _: &CudaStorage,
        _: &crate::Layout,
    ) -> Result<Tensor> {
        Err(Error::NotCompiledWithCudaSupport)
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    _device: &CudaDevice,
    _data: &[T],
) -> Result<super::QStorage> {
    Err(Error::NotCompiledWithCudaSupport)
}
