//! Qwen3's MLP activation `silu(gate) * up` as one op.
//!
//! As two kernels the `[T, intermediate]` silu output is written and read back once more than
//! needed (at 8192 tokens in bf16, 48 MiB each way per layer). On Metal [`silu_mul`] runs one
//! kernel (`qwen3_silu_mul.metal`) that reads gate and up once, computes in f32 and rounds
//! once. Elsewhere it runs today's two candle ops.

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, Layout, Result, Shape, Tensor, WithDType};

/// `silu(gate) * up` for two tensors of one shape and dtype. On Metal this is one kernel; on
/// any other device it is `candle_nn::ops::silu(gate) * up`.
pub(crate) fn silu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    if gate.device().is_metal() {
        return gate
            .contiguous()?
            .apply_op2_no_bwd(&up.contiguous()?, &SiluMul);
    }
    candle_nn::ops::silu(gate)? * up
}

/// The fused op: its two inputs are gate and up.
pub(crate) struct SiluMul;

fn check(l_gate: &Layout, l_up: &Layout) -> Result<usize> {
    if l_gate.shape() != l_up.shape() {
        candle_core::bail!(
            "silu-mul gate is {:?} but up is {:?}",
            l_gate.shape(),
            l_up.shape()
        )
    }
    for (name, l) in [("gate", l_gate), ("up", l_up)] {
        if !l.is_contiguous() {
            candle_core::bail!("silu-mul {name} has to be contiguous")
        }
    }
    Ok(l_gate.shape().elem_count())
}

fn cpu_inner<T: WithDType>(gate: &[T], up: &[T]) -> Vec<T> {
    gate.iter()
        .zip(up)
        .map(|(g, u)| {
            let (g, u) = (g.to_f64() as f32, u.to_f64() as f32);
            T::from_f64(f64::from(g / (1.0 + (-g).exp()) * u))
        })
        .collect()
}

impl candle_core::CustomOp2 for SiluMul {
    fn name(&self) -> &'static str {
        "silu-mul"
    }

    fn cpu_fwd(
        &self,
        s_gate: &CpuStorage,
        l_gate: &Layout,
        s_up: &CpuStorage,
        l_up: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let n = check(l_gate, l_up)?;
        fn slice<'a, T>(s: &'a [T], l: &Layout, n: usize) -> &'a [T] {
            &s[l.start_offset()..l.start_offset() + n]
        }
        let out = match (s_gate, s_up) {
            (CpuStorage::F32(g), CpuStorage::F32(u)) => {
                CpuStorage::F32(cpu_inner(slice(g, l_gate, n), slice(u, l_up, n)))
            }
            (CpuStorage::F16(g), CpuStorage::F16(u)) => {
                CpuStorage::F16(cpu_inner(slice(g, l_gate, n), slice(u, l_up, n)))
            }
            (CpuStorage::BF16(g), CpuStorage::BF16(u)) => {
                CpuStorage::BF16(cpu_inner(slice(g, l_gate, n), slice(u, l_up, n)))
            }
            _ => candle_core::bail!(
                "silu-mul takes one float dtype for both inputs, got {:?} and {:?}",
                s_gate.dtype(),
                s_up.dtype()
            ),
        };
        Ok((out, l_gate.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        gate: &candle_core::MetalStorage,
        l_gate: &Layout,
        up: &candle_core::MetalStorage,
        l_up: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        metal::forward(gate, l_gate, up, l_up)
    }
}

#[cfg(feature = "metal")]
mod metal {
    use super::check;
    use candle_core::backend::BackendStorage;
    use candle_core::metal_backend::DeviceId;
    use candle_core::{DType, Layout, MetalDevice, MetalStorage, Result, Shape};
    use candle_metal_kernels::metal::{ComputeCommandEncoder, ComputePipeline};
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    const SOURCE: &str = include_str!("qwen3_silu_mul.metal");

    /// One compiled pipeline per device and dtype, built the first time it is asked for.
    fn pipeline(device: &MetalDevice, dtype: DType) -> Result<ComputePipeline> {
        static PIPELINES: OnceLock<Mutex<HashMap<(DeviceId, DType), ComputePipeline>>> =
            OnceLock::new();
        let name = match dtype {
            DType::F32 => "silu_mul_f32",
            DType::F16 => "silu_mul_f16",
            DType::BF16 => "silu_mul_bf16",
            dtype => candle_core::bail!("silu-mul is not implemented for {dtype:?}"),
        };
        let mut pipelines = PIPELINES
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .map_err(|e| candle_core::Error::Msg(format!("silu-mul pipeline cache: {e}")))?;
        if let Some(pipeline) = pipelines.get(&(device.id(), dtype)) {
            return Ok(pipeline.clone());
        }
        let metal = device.metal_device();
        let library = metal
            .new_library_with_source(SOURCE, None)
            .map_err(candle_core::Error::wrap)?;
        let function = library
            .get_function(name, None)
            .map_err(candle_core::Error::wrap)?;
        let pipeline = metal
            .new_compute_pipeline_state_with_function(&function)
            .map_err(candle_core::Error::wrap)?;
        pipelines.insert((device.id(), dtype), pipeline.clone());
        Ok(pipeline)
    }

    pub(super) fn forward(
        gate: &MetalStorage,
        l_gate: &Layout,
        up: &MetalStorage,
        l_up: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let el = check(l_gate, l_up)?;
        let dtype = gate.dtype();
        if up.dtype() != dtype {
            candle_core::bail!("silu-mul dtypes differ: {dtype:?} {:?}", up.dtype())
        }
        let n = u32::try_from(el).map_err(candle_core::Error::wrap)?;
        let device = gate.device();
        let pipeline = pipeline(device, dtype)?;
        let output = device
            .new_buffer_builder()
            .with_size_for(el, dtype)
            .with_label("silu_mul")
            .build()?;
        let size = dtype.size_in_bytes();
        let guard = device.command_encoder()?;
        guard.set_label("silu_mul");
        let encoder: &ComputeCommandEncoder = guard.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &n);
        encoder.set_input_buffer(1, Some(gate.buffer()), l_gate.start_offset() * size);
        encoder.set_input_buffer(2, Some(up.buffer()), l_up.start_offset() * size);
        encoder.set_output_buffer(3, Some(&output), 0);
        // candle-metal-kernels does not re-export objc2-metal's `MTLSize`, so both sizes start
        // as the one value `get_block_dims` hands back and take their widths here.
        let width = pipeline.max_total_threads_per_threadgroup().min(256);
        let mut groups = candle_metal_kernels::utils::get_block_dims(1, 1, 1);
        groups.width = el.div_ceil(width);
        let mut threads = groups;
        threads.width = width;
        encoder.dispatch_thread_groups(groups, threads);
        drop(guard);
        Ok((
            MetalStorage::new(output, device.clone(), el, dtype),
            l_gate.shape().clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{silu_mul, SiluMul};
    use candle_core::{DType, Device, Tensor};

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

    /// Gate and up as the MLP's projections give them, with an odd element count so the last
    /// threadgroup is part-filled on Metal.
    fn inputs(device: &Device) -> (Tensor, Tensor) {
        let shape = (2usize, 7usize, 301usize);
        (
            Tensor::randn(0f32, 3.0, shape, device).unwrap(),
            Tensor::randn(0f32, 1.0, shape, device).unwrap(),
        )
    }

    fn two_ops(gate: &Tensor, up: &Tensor) -> Tensor {
        (candle_nn::ops::silu(gate).unwrap() * up).unwrap()
    }

    #[test]
    fn cpu_op_matches_silu_then_mul() {
        let (gate, up) = inputs(&Device::Cpu);
        let expected = two_ops(&gate, &up);
        let got = gate.apply_op2_no_bwd(&up, &SiluMul).unwrap();
        assert_eq!(got.dims(), expected.dims());
        assert!(max_abs_diff(&got, &expected) < 1e-5);
        // silu moved the values, so this is not up against up.
        assert!(max_abs_diff(&got, &(&gate * &up).unwrap()) > 1e-1);
        // The CPU path of the dispatcher is the two ops themselves.
        let dispatched = silu_mul(&gate, &up).unwrap();
        assert_eq!(max_abs_diff(&dispatched, &expected), 0.0);
    }

    #[test]
    fn cpu_op_rejects_shapes_that_differ() {
        let (gate, up) = inputs(&Device::Cpu);
        let up = up.narrow(2, 0, 300).unwrap().contiguous().unwrap();
        assert!(gate.apply_op2_no_bwd(&up, &SiluMul).is_err());
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_kernel_matches_silu_then_mul() {
        let device = Device::new_metal(0).unwrap();
        let (gate, up) = inputs(&device);
        let reference = two_ops(&gate, &up);
        let got = silu_mul(&gate, &up).unwrap();
        assert_eq!(got.dims(), reference.dims());
        assert!(max_abs_diff(&got, &reference) < 1e-5, "f32 differs");

        // bf16 and f16, each with its own kernel: one rounding against the two ops' two, each
        // measured against the f32 result of the same half-precision inputs. `mantissa` is the
        // dtype's stored mantissa bits, so one step is |x| * 2^-mantissa.
        for (dtype, mantissa) in [(DType::BF16, 7), (DType::F16, 10)] {
            let half = |x: &Tensor| x.to_dtype(dtype).unwrap();
            let f32 = |x: &Tensor| x.to_dtype(DType::F32).unwrap();
            let (gate, up) = (half(&gate), half(&up));
            let reference = two_ops(&f32(&gate), &f32(&up));
            let mean_err = |x: &Tensor| {
                (f32(x) - &reference)
                    .unwrap()
                    .abs()
                    .unwrap()
                    .mean_all()
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap()
            };
            let got = silu_mul(&gate, &up).unwrap();
            assert_eq!(got.dtype(), dtype);
            let (one, two) = (mean_err(&got), mean_err(&two_ops(&gate, &up)));
            assert!(one <= two, "{dtype:?}: mean error {one}, the two ops {two}");
            // And the CPU op on the same inputs rounds the same way, up to one step where Metal's
            // fast exp lands on the other side of a rounding boundary.
            let to_cpu = |x: &Tensor| x.to_device(&Device::Cpu).unwrap();
            let cpu = |x: &Tensor| f32(&to_cpu(x));
            let on_cpu = to_cpu(&gate)
                .apply_op2_no_bwd(&to_cpu(&up), &SiluMul)
                .unwrap();
            assert_eq!(on_cpu.dtype(), dtype);
            let on_cpu = f32(&on_cpu);
            let step = (on_cpu.abs().unwrap() * 2f64.powi(-mantissa)).unwrap();
            let over = ((cpu(&got) - &on_cpu).unwrap().abs().unwrap() - step)
                .unwrap()
                .flatten_all()
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(
                over <= 1e-6,
                "{dtype:?}: Metal and CPU differ by more than one step"
            );
        }
    }
}
