//! The two fused epilogues the BERT encoder uses.
//!
//! `add_layernorm` folds the projection bias, the residual add and the LayerNorm into one
//! dispatch; `bias_gelu` folds the MLP's up-projection bias and the GELU. Both keep the
//! same math as the unfused sequence (the CPU implementation is that sequence, which is
//! also what the parity test compares against), but remove five Metal dispatches per layer.

use candle_core::{CpuStorage, Layout, Result, Shape, Tensor, WithDType};
use candle_nn::{LayerNorm, Module};

/// `layer_norm(x + residual + bias, weight, ln_bias)`, where `wb` stacks the three rows of
/// length `cols`: the projection bias, the LayerNorm weight and the LayerNorm bias.
pub(crate) fn add_layernorm(
    x: &Tensor,
    residual: &Tensor,
    wb: &Tensor,
    eps: f64,
) -> Result<Tensor> {
    if x.device().is_metal() && x.dtype() == residual.dtype() && x.dtype() == wb.dtype() {
        let x = x.contiguous()?;
        let residual = residual.contiguous()?;
        let wb = wb.contiguous()?;
        return x.apply_op3_no_bwd(&residual, &wb, &AddLayerNorm { eps });
    }
    let cols = x.dims2()?.1;
    let bias = wb.narrow(0, 0, 1)?.reshape(cols)?;
    let weight = wb.narrow(0, 1, 1)?.reshape(cols)?;
    let ln_bias = wb.narrow(0, 2, 1)?.reshape(cols)?;
    let sum = (x + residual)?.broadcast_add(&bias)?;
    LayerNorm::new(weight, ln_bias, eps).forward(&sum)
}

/// `gelu(x + bias)` with the bias broadcast over the rows. Exact (erf) GELU, matching the
/// checkpoint's `hidden_act`.
pub(crate) fn bias_gelu(x: &Tensor, bias: &Tensor) -> Result<Tensor> {
    if x.device().is_metal() && x.dtype() == bias.dtype() {
        let x = x.contiguous()?;
        let bias = bias.contiguous()?;
        return x.apply_op2_no_bwd(&bias, &BiasGelu);
    }
    x.broadcast_add(bias)?.gelu_erf()
}

pub(crate) struct AddLayerNorm {
    eps: f64,
}

fn check(l_x: &Layout, l_residual: &Layout, l_wb: &Layout) -> Result<(usize, usize)> {
    let (rows, cols) = l_x.shape().dims2()?;
    if l_residual.shape().dims() != [rows, cols] {
        candle_core::bail!(
            "add-layernorm residual is {:?}, expected [{rows}, {cols}]",
            l_residual.shape().dims()
        )
    }
    match l_wb.shape().dims() {
        [3, c] if *c == cols => {}
        d => candle_core::bail!("add-layernorm wb is {d:?}, expected [3, {cols}]"),
    }
    Ok((rows, cols))
}

impl candle_core::CustomOp3 for AddLayerNorm {
    fn name(&self) -> &'static str {
        "add-layernorm"
    }

    fn cpu_fwd(
        &self,
        s_x: &CpuStorage,
        l_x: &Layout,
        s_r: &CpuStorage,
        l_r: &Layout,
        s_wb: &CpuStorage,
        l_wb: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (rows, cols) = check(l_x, l_r, l_wb)?;
        fn f<T: WithDType>(
            x: &[T],
            r: &[T],
            wb: &[T],
            lx: &Layout,
            lr: &Layout,
            lwb: &Layout,
            rows: usize,
            cols: usize,
            eps: f64,
        ) -> Vec<T> {
            let xs = &x[lx.start_offset()..lx.start_offset() + rows * cols];
            let rs = &r[lr.start_offset()..lr.start_offset() + rows * cols];
            let wbs = &wb[lwb.start_offset()..lwb.start_offset() + 3 * cols];
            let (bias, weight, ln_bias) = (&wbs[..cols], &wbs[cols..2 * cols], &wbs[2 * cols..]);
            let mut out = Vec::with_capacity(rows * cols);
            for row in 0..rows {
                let base = row * cols;
                let mut sum = 0f64;
                let mut sq = 0f64;
                for i in 0..cols {
                    let v = xs[base + i].to_f64() + rs[base + i].to_f64() + bias[i].to_f64();
                    sum += v;
                    sq += v * v;
                }
                let mean = sum / cols as f64;
                let var = sq / cols as f64 - mean * mean;
                let inv = 1.0 / (var + eps).sqrt();
                for i in 0..cols {
                    let v = xs[base + i].to_f64() + rs[base + i].to_f64() + bias[i].to_f64();
                    let n = (v - mean) * inv * weight[i].to_f64() + ln_bias[i].to_f64();
                    out.push(T::from_f64(n));
                }
            }
            out
        }
        let out = match (s_x, s_r, s_wb) {
            (CpuStorage::F32(x), CpuStorage::F32(r), CpuStorage::F32(wb)) => {
                CpuStorage::F32(f(x, r, wb, l_x, l_r, l_wb, rows, cols, self.eps))
            }
            (CpuStorage::F16(x), CpuStorage::F16(r), CpuStorage::F16(wb)) => {
                CpuStorage::F16(f(x, r, wb, l_x, l_r, l_wb, rows, cols, self.eps))
            }
            (CpuStorage::BF16(x), CpuStorage::BF16(r), CpuStorage::BF16(wb)) => {
                CpuStorage::BF16(f(x, r, wb, l_x, l_r, l_wb, rows, cols, self.eps))
            }
            _ => candle_core::bail!("add-layernorm needs one float dtype for all inputs"),
        };
        Ok((out, l_x.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        x: &candle_core::MetalStorage,
        l_x: &Layout,
        residual: &candle_core::MetalStorage,
        l_r: &Layout,
        wb: &candle_core::MetalStorage,
        l_wb: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        metal::add_layernorm(self.eps, x, l_x, residual, l_r, wb, l_wb)
    }
}

pub(crate) struct BiasGelu;

impl candle_core::CustomOp2 for BiasGelu {
    fn name(&self) -> &'static str {
        "bias-gelu"
    }

    fn cpu_fwd(
        &self,
        s_x: &CpuStorage,
        l_x: &Layout,
        s_bias: &CpuStorage,
        l_bias: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (rows, cols) = l_x.shape().dims2()?;
        if l_bias.shape().dims() != [cols] {
            candle_core::bail!(
                "bias-gelu bias is {:?}, expected [{cols}]",
                l_bias.shape().dims()
            )
        }
        fn f<T: WithDType>(
            x: &[T],
            bias: &[T],
            lx: &Layout,
            lb: &Layout,
            rows: usize,
            cols: usize,
        ) -> Vec<T> {
            let xs = &x[lx.start_offset()..lx.start_offset() + rows * cols];
            let bs = &bias[lb.start_offset()..lb.start_offset() + cols];
            let mut out = Vec::with_capacity(rows * cols);
            for row in 0..rows {
                for i in 0..cols {
                    let v = xs[row * cols + i].to_f64() + bs[i].to_f64();
                    // 0.5 * v * (1 + erf(v / sqrt(2)))
                    let c = 0.7978845608028654 * 0.5f64.sqrt() * v; // 1/sqrt(2)
                    out.push(T::from_f64(0.5 * v * (1.0 + erf(c))));
                }
            }
            out
        }
        // f64 erf via the standard polynomial is overkill; use the platform libm through
        // f32 because candle's own gelu_erf also computes in f32.
        fn erf(x: f64) -> f64 {
            // Abramowitz & Stegun 7.1.26, max abs error 1.5e-7.
            let sign = if x < 0.0 { -1.0 } else { 1.0 };
            let x = x.abs();
            let t = 1.0 / (1.0 + 0.3275911 * x);
            let y = 1.0
                - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
                    + 0.254829592)
                    * t
                    * (-x * x).exp();
            sign * y
        }
        let out = match (s_x, s_bias) {
            (CpuStorage::F32(x), CpuStorage::F32(b)) => {
                CpuStorage::F32(f(x, b, l_x, l_bias, rows, cols))
            }
            (CpuStorage::F16(x), CpuStorage::F16(b)) => {
                CpuStorage::F16(f(x, b, l_x, l_bias, rows, cols))
            }
            (CpuStorage::BF16(x), CpuStorage::BF16(b)) => {
                CpuStorage::BF16(f(x, b, l_x, l_bias, rows, cols))
            }
            _ => candle_core::bail!("bias-gelu needs one float dtype for both inputs"),
        };
        Ok((out, l_x.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        x: &candle_core::MetalStorage,
        l_x: &Layout,
        bias: &candle_core::MetalStorage,
        l_bias: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        metal::bias_gelu(x, l_x, bias, l_bias)
    }
}

#[cfg(feature = "metal")]
mod metal {
    use candle_core::backend::BackendStorage;
    use candle_core::metal_backend::DeviceId;
    use candle_core::{DType, Layout, MetalDevice, MetalStorage, Result, Shape};
    use candle_metal_kernels::metal::{ComputeCommandEncoder, ComputePipeline};
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    const SOURCE: &str = include_str!("bert_fused_ops.metal");

    fn pipeline(device: &MetalDevice, dtype: DType, kernel: &str) -> Result<ComputePipeline> {
        static PIPELINES: OnceLock<Mutex<HashMap<(DeviceId, DType, String), ComputePipeline>>> =
            OnceLock::new();
        let suffix = match dtype {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            dtype => candle_core::bail!("bert-fused is not implemented for {dtype:?}"),
        };
        let name = format!("{kernel}_{suffix}");
        let key = (device.id(), dtype, name.clone());
        let mut pipelines = PIPELINES
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .map_err(|e| candle_core::Error::Msg(format!("bert-fused pipeline cache: {e}")))?;
        if let Some(pipeline) = pipelines.get(&key) {
            return Ok(pipeline.clone());
        }
        let metal = device.metal_device();
        let library = metal
            .new_library_with_source(SOURCE, None)
            .map_err(candle_core::Error::wrap)?;
        let function = library
            .get_function(&name, None)
            .map_err(candle_core::Error::wrap)?;
        let pipeline = metal
            .new_compute_pipeline_state_with_function(&function)
            .map_err(candle_core::Error::wrap)?;
        pipelines.insert(key, pipeline.clone());
        Ok(pipeline)
    }

    pub(super) fn add_layernorm(
        eps: f64,
        x: &MetalStorage,
        l_x: &Layout,
        residual: &MetalStorage,
        l_r: &Layout,
        wb: &MetalStorage,
        l_wb: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let (rows, cols) = super::check(l_x, l_r, l_wb)?;
        let dtype = x.dtype();
        if residual.dtype() != dtype || wb.dtype() != dtype {
            candle_core::bail!("add-layernorm dtypes differ")
        }
        let cols_u32 = u32::try_from(cols).map_err(candle_core::Error::wrap)?;
        let eps_f32 = eps as f32;
        let device = x.device();
        let pipeline = pipeline(device, dtype, "add_layernorm")?;
        let el = rows * cols;
        let output = device
            .new_buffer_builder()
            .with_size_for(el, dtype)
            .with_label("add_layernorm")
            .build()?;
        let size = dtype.size_in_bytes();
        let guard = device.command_encoder()?;
        guard.set_label("add_layernorm");
        let encoder: &ComputeCommandEncoder = guard.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_input_buffer(0, Some(x.buffer()), l_x.start_offset() * size);
        encoder.set_input_buffer(1, Some(residual.buffer()), l_r.start_offset() * size);
        encoder.set_input_buffer(2, Some(wb.buffer()), l_wb.start_offset() * size);
        encoder.set_output_buffer(3, Some(&output), 0);
        encoder.set_bytes(4, &cols_u32);
        encoder.set_bytes(5, &eps_f32);
        // One threadgroup per row; 256 threads, each striding the row.
        let mut groups = candle_metal_kernels::utils::get_block_dims(1, 1, 1);
        groups.width = rows;
        let mut threads = groups;
        threads.width = super::TG;
        encoder.dispatch_thread_groups(groups, threads);
        drop(guard);
        Ok((
            MetalStorage::new(output, device.clone(), el, dtype),
            l_x.shape().clone(),
        ))
    }

    pub(super) fn bias_gelu(
        x: &MetalStorage,
        l_x: &Layout,
        bias: &MetalStorage,
        l_bias: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let (rows, cols) = l_x.shape().dims2()?;
        if l_bias.shape().dims() != [cols] {
            candle_core::bail!(
                "bias-gelu bias is {:?}, expected [{cols}]",
                l_bias.shape().dims()
            )
        }
        let dtype = x.dtype();
        if bias.dtype() != dtype {
            candle_core::bail!("bias-gelu dtypes differ")
        }
        let el = rows * cols;
        let n = u32::try_from(el).map_err(candle_core::Error::wrap)?;
        let cols_u32 = u32::try_from(cols).map_err(candle_core::Error::wrap)?;
        let device = x.device();
        let pipeline = pipeline(device, dtype, "bias_gelu")?;
        let output = device
            .new_buffer_builder()
            .with_size_for(el, dtype)
            .with_label("bias_gelu")
            .build()?;
        let size = dtype.size_in_bytes();
        let guard = device.command_encoder()?;
        guard.set_label("bias_gelu");
        let encoder: &ComputeCommandEncoder = guard.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &n);
        encoder.set_input_buffer(1, Some(x.buffer()), l_x.start_offset() * size);
        encoder.set_input_buffer(2, Some(bias.buffer()), l_bias.start_offset() * size);
        encoder.set_output_buffer(3, Some(&output), 0);
        encoder.set_bytes(4, &cols_u32);
        let width = pipeline.max_total_threads_per_threadgroup().min(256);
        let mut groups = candle_metal_kernels::utils::get_block_dims(1, 1, 1);
        groups.width = el.div_ceil(width);
        let mut threads = groups;
        threads.width = width;
        encoder.dispatch_thread_groups(groups, threads);
        drop(guard);
        Ok((
            MetalStorage::new(output, device.clone(), el, dtype),
            l_x.shape().clone(),
        ))
    }
}

/// Threadgroup width shared with the Metal shader (must match `TG` there).
const TG: usize = 256;

#[cfg(test)]
mod tests {
    use super::{add_layernorm, bias_gelu};
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{LayerNorm, Module};

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a.to_dtype(DType::F32).unwrap() - b.to_dtype(DType::F32).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    fn shared(rows: usize, cols: usize) -> (Tensor, Tensor, Tensor) {
        let w = Tensor::randn(0f32, 1f32, (3, cols), &Device::Cpu).unwrap();
        (
            Tensor::randn(0f32, 1f32, (rows, cols), &Device::Cpu).unwrap(),
            Tensor::randn(0f32, 1f32, (rows, cols), &Device::Cpu).unwrap(),
            w,
        )
    }

    fn unfused(x: &Tensor, r: &Tensor, wb: &Tensor, cols: usize, eps: f64) -> Tensor {
        let bias = wb.narrow(0, 0, 1).unwrap().reshape(cols).unwrap();
        let weight = wb.narrow(0, 1, 1).unwrap().reshape(cols).unwrap();
        let ln_bias = wb.narrow(0, 2, 1).unwrap().reshape(cols).unwrap();
        let sum = (x + r).unwrap().broadcast_add(&bias).unwrap();
        LayerNorm::new(weight, ln_bias, eps).forward(&sum).unwrap()
    }

    #[test]
    fn add_layernorm_matches_the_unfused_sequence() {
        let (rows, cols) = (2usize, 301usize);
        let (x, r, wb) = shared(rows, cols);
        let got = add_layernorm(&x, &r, &wb, 1e-12).unwrap();
        let want = unfused(&x, &r, &wb, cols, 1e-12);
        assert!(
            max_abs_diff(&got, &want) < 1e-3,
            "diff {}",
            max_abs_diff(&got, &want)
        );
    }

    #[test]
    fn bias_gelu_matches_gelu_of_the_sum() {
        let (rows, cols) = (2usize, 301usize);
        let (x, _, wb) = shared(rows, cols);
        let bias = wb.narrow(0, 0, 1).unwrap().reshape(cols).unwrap();
        let got = bias_gelu(&x, &bias).unwrap();
        let want = x.broadcast_add(&bias).unwrap().gelu_erf().unwrap();
        assert!(
            max_abs_diff(&got, &want) < 1e-4,
            "diff {}",
            max_abs_diff(&got, &want)
        );
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_fused_ops_match_cpu() {
        let dev = Device::new_metal(0).unwrap();
        let (rows, cols) = (5usize, 1024usize);
        for dtype in [DType::F32, DType::F16] {
            let x = Tensor::randn(0f32, 1f32, (rows, cols), &dev)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let r = Tensor::randn(0f32, 1f32, (rows, cols), &dev)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let wb = Tensor::randn(0f32, 0.3, (3, cols), &dev)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let want = unfused(
                &x.to_dtype(DType::F32).unwrap(),
                &r.to_dtype(DType::F32).unwrap(),
                &wb.to_dtype(DType::F32).unwrap(),
                cols,
                1e-5,
            );
            let got = add_layernorm(&x, &r, &wb, 1e-5).unwrap();
            let diff = max_abs_diff(&got, &want);
            assert!(diff < 0.05, "{dtype:?} add-layernorm diff {diff}");

            let bias = Tensor::randn(0f32, 0.3, cols, &dev)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let want = x
                .to_dtype(DType::F32)
                .unwrap()
                .broadcast_add(&bias.to_dtype(DType::F32).unwrap())
                .unwrap()
                .gelu_erf()
                .unwrap();
            let got = bias_gelu(&x, &bias).unwrap();
            let diff = max_abs_diff(&got, &want);
            assert!(diff < 0.05, "{dtype:?} bias-gelu diff {diff}");
        }
    }
}
