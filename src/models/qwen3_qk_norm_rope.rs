//! Qwen3's q_norm / k_norm and the plain half-split rope as one op.
//!
//! The attention runs an RMSNorm over each head's `[D]` slice of q and k and then rotates it by
//! the `[T, D/2]` cos and sin tables. As two kernels that is two reads and two writes of q and of
//! k per layer; on Metal [`rms_norm_rope_thd`] runs them as one kernel (`qwen3_qk_norm_rope.metal`)
//! that reads each row once and writes it once. Elsewhere it runs today's two candle ops.

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, Layout, Result, Shape, Tensor, WithDType};

/// `rope_thd(rms_norm(xs, weight, eps), cos, sin)`: `xs` is `[B, T, H, D]`, `weight` `[D]`, and
/// `cos` and `sin` `[T, D/2]` (positions `0..T` in every row). On Metal this is one kernel; on any
/// other device it is `candle_nn::ops::rms_norm` then `candle_nn::rotary_emb::rope_thd`.
pub(crate) fn rms_norm_rope_thd(
    xs: &Tensor,
    weight: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let xs = xs.contiguous()?;
    // A no-op when the weight is in the input's dtype, which is how the model runs.
    let weight = weight.to_dtype(xs.dtype())?;
    if xs.device().is_metal() {
        return xs.apply_op3_no_bwd(cos, sin, &QkNormRope { weight, eps });
    }
    let normed = candle_nn::ops::rms_norm(&xs, &weight, eps)?;
    candle_nn::rotary_emb::rope_thd(&normed, cos, sin)
}

/// The fused op. Its three inputs are `xs`, `cos` and `sin`; the norm weight rides in the op,
/// because a custom op takes at most three tensors.
pub(crate) struct QkNormRope {
    pub(crate) weight: Tensor,
    pub(crate) eps: f32,
}

/// The shapes the op runs over, checked once against every input's layout.
struct Dims {
    heads: usize,
    seq_len: usize,
    half_d: usize,
    rows: usize,
}

fn check(l_src: &Layout, l_cos: &Layout, l_sin: &Layout, l_weight: &Layout) -> Result<Dims> {
    let (b, t, h, d) = l_src.shape().dims4()?;
    if !d.is_multiple_of(2) {
        candle_core::bail!("qk-norm-rope needs an even head dim, got {d}")
    }
    let half_d = d / 2;
    for (name, l) in [("cos", l_cos), ("sin", l_sin)] {
        let (cs_t, cs_d) = l.shape().dims2()?;
        if cs_d != half_d || cs_t < t {
            candle_core::bail!(
                "qk-norm-rope {name} is {:?}, it needs [>= {t}, {half_d}]",
                l.shape()
            )
        }
    }
    if l_weight.shape().dims1()? != d {
        candle_core::bail!(
            "qk-norm-rope weight is {:?}, it needs [{d}]",
            l_weight.shape()
        )
    }
    for (name, l) in [
        ("xs", l_src),
        ("cos", l_cos),
        ("sin", l_sin),
        ("weight", l_weight),
    ] {
        if !l.is_contiguous() {
            candle_core::bail!("qk-norm-rope {name} has to be contiguous")
        }
    }
    Ok(Dims {
        heads: h,
        seq_len: t,
        half_d,
        rows: b * t * h,
    })
}

fn cpu_inner<T: WithDType>(
    src: &[T],
    cos: &[T],
    sin: &[T],
    weight: &[T],
    dims: &Dims,
    eps: f32,
) -> Vec<T> {
    let (half_d, d) = (dims.half_d, 2 * dims.half_d);
    let mut dst = vec![T::zero(); dims.rows * d];
    for row in 0..dims.rows {
        let x = &src[row * d..(row + 1) * d];
        let sum_sq: f32 = x
            .iter()
            .map(|v| {
                let v = v.to_f64() as f32;
                v * v
            })
            .sum();
        let total = 1.0 / (sum_sq / d as f32 + eps).sqrt();
        let cs = ((row / dims.heads) % dims.seq_len) * half_d;
        let out = &mut dst[row * d..(row + 1) * d];
        // In f32 throughout, rounded to `T` once at the store, as the Metal kernel does.
        let f = |v: T| v.to_f64() as f32;
        for i in 0..half_d {
            let n1 = f(x[i]) * total * f(weight[i]);
            let n2 = f(x[i + half_d]) * total * f(weight[i + half_d]);
            let (c, s) = (f(cos[cs + i]), f(sin[cs + i]));
            out[i] = T::from_f64(f64::from(n1 * c - n2 * s));
            out[i + half_d] = T::from_f64(f64::from(n1 * s + n2 * c));
        }
    }
    dst
}

impl candle_core::CustomOp3 for QkNormRope {
    fn name(&self) -> &'static str {
        "qk-norm-rope"
    }

    fn cpu_fwd(
        &self,
        s_src: &CpuStorage,
        l_src: &Layout,
        s_cos: &CpuStorage,
        l_cos: &Layout,
        s_sin: &CpuStorage,
        l_sin: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (weight, l_weight) = self.weight.storage_and_layout();
        let candle_core::Storage::Cpu(weight) = &*weight else {
            candle_core::bail!("qk-norm-rope weight is not on the cpu")
        };
        let dims = check(l_src, l_cos, l_sin, l_weight)?;
        fn slice<'a, T>(s: &'a [T], l: &Layout) -> &'a [T] {
            &s[l.start_offset()..l.start_offset() + l.shape().elem_count()]
        }
        let eps = self.eps;
        let out = match (s_src, s_cos, s_sin, weight) {
            (CpuStorage::F32(x), CpuStorage::F32(c), CpuStorage::F32(s), CpuStorage::F32(w)) => {
                CpuStorage::F32(cpu_inner(
                    slice(x, l_src),
                    slice(c, l_cos),
                    slice(s, l_sin),
                    slice(w, l_weight),
                    &dims,
                    eps,
                ))
            }
            (CpuStorage::F16(x), CpuStorage::F16(c), CpuStorage::F16(s), CpuStorage::F16(w)) => {
                CpuStorage::F16(cpu_inner(
                    slice(x, l_src),
                    slice(c, l_cos),
                    slice(s, l_sin),
                    slice(w, l_weight),
                    &dims,
                    eps,
                ))
            }
            (
                CpuStorage::BF16(x),
                CpuStorage::BF16(c),
                CpuStorage::BF16(s),
                CpuStorage::BF16(w),
            ) => CpuStorage::BF16(cpu_inner(
                slice(x, l_src),
                slice(c, l_cos),
                slice(s, l_sin),
                slice(w, l_weight),
                &dims,
                eps,
            )),
            _ => candle_core::bail!(
                "qk-norm-rope takes one float dtype for every input, got {:?}",
                s_src.dtype()
            ),
        };
        Ok((out, l_src.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        src: &candle_core::MetalStorage,
        l_src: &Layout,
        cos: &candle_core::MetalStorage,
        l_cos: &Layout,
        sin: &candle_core::MetalStorage,
        l_sin: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        metal::forward(self, src, l_src, cos, l_cos, sin, l_sin)
    }
}

#[cfg(feature = "metal")]
mod metal {
    use super::{check, QkNormRope};
    use candle_core::backend::BackendStorage;
    use candle_core::metal_backend::DeviceId;
    use candle_core::{DType, Layout, MetalDevice, MetalStorage, Result, Shape};
    use candle_metal_kernels::metal::{ComputeCommandEncoder, ComputePipeline};
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    const SOURCE: &str = include_str!("qwen3_qk_norm_rope.metal");

    /// One compiled pipeline per device and dtype, built the first time it is asked for.
    fn pipeline(device: &MetalDevice, dtype: DType) -> Result<ComputePipeline> {
        static PIPELINES: OnceLock<Mutex<HashMap<(DeviceId, DType), ComputePipeline>>> =
            OnceLock::new();
        let name = match dtype {
            DType::F32 => "qk_norm_rope_f32",
            DType::F16 => "qk_norm_rope_f16",
            DType::BF16 => "qk_norm_rope_bf16",
            dtype => candle_core::bail!("qk-norm-rope is not implemented for {dtype:?}"),
        };
        let mut pipelines = PIPELINES
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .map_err(|e| candle_core::Error::Msg(format!("qk-norm-rope pipeline cache: {e}")))?;
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
        op: &QkNormRope,
        src: &MetalStorage,
        l_src: &Layout,
        cos: &MetalStorage,
        l_cos: &Layout,
        sin: &MetalStorage,
        l_sin: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let (weight, l_weight) = op.weight.storage_and_layout();
        let candle_core::Storage::Metal(weight) = &*weight else {
            candle_core::bail!("qk-norm-rope weight is not on the metal device")
        };
        let dims = check(l_src, l_cos, l_sin, l_weight)?;
        let dtype = src.dtype();
        if cos.dtype() != dtype || sin.dtype() != dtype || weight.dtype() != dtype {
            candle_core::bail!(
                "qk-norm-rope dtypes differ: {dtype:?} {:?} {:?} {:?}",
                cos.dtype(),
                sin.dtype(),
                weight.dtype()
            )
        }
        let device = src.device();
        let pipeline = pipeline(device, dtype)?;
        if dims.half_d > pipeline.max_total_threads_per_threadgroup() || dims.half_d > 32 * 32 {
            candle_core::bail!("qk-norm-rope head dim {} is too wide", 2 * dims.half_d)
        }
        let (heads, seq_len, half_d) = (
            u32::try_from(dims.heads).map_err(candle_core::Error::wrap)?,
            u32::try_from(dims.seq_len).map_err(candle_core::Error::wrap)?,
            u32::try_from(dims.half_d).map_err(candle_core::Error::wrap)?,
        );

        let el = l_src.shape().elem_count();
        let output = device
            .new_buffer_builder()
            .with_size_for(el, dtype)
            .with_label("qk_norm_rope")
            .build()?;
        let size = dtype.size_in_bytes();
        let guard = device.command_encoder()?;
        guard.set_label("qk_norm_rope");
        let encoder: &ComputeCommandEncoder = guard.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &heads);
        encoder.set_bytes(1, &seq_len);
        encoder.set_bytes(2, &half_d);
        encoder.set_bytes(3, &op.eps);
        encoder.set_input_buffer(4, Some(src.buffer()), l_src.start_offset() * size);
        encoder.set_input_buffer(5, Some(cos.buffer()), l_cos.start_offset() * size);
        encoder.set_input_buffer(6, Some(sin.buffer()), l_sin.start_offset() * size);
        encoder.set_input_buffer(7, Some(weight.buffer()), l_weight.start_offset() * size);
        encoder.set_output_buffer(8, Some(&output), 0);
        // candle-metal-kernels does not re-export objc2-metal's `MTLSize`, so both sizes start
        // as the one value `get_block_dims` hands back and take their widths here.
        let mut groups = candle_metal_kernels::utils::get_block_dims(1, 1, 1);
        groups.width = dims.rows;
        let mut threads = groups;
        threads.width = dims.half_d;
        encoder.dispatch_thread_groups(groups, threads);
        drop(guard);
        Ok((
            MetalStorage::new(output, device.clone(), el, dtype),
            l_src.shape().clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{rms_norm_rope_thd, QkNormRope};
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

    /// q as the attention holds it, a norm weight near 1, and `[T, D/2]` tables for positions
    /// `0..T` with frequencies spread the way Qwen3's are, so late rows are really rotated.
    fn inputs(device: &Device, dtype: DType, d: usize) -> (Tensor, Tensor, Tensor, Tensor) {
        let (b, t, h) = (2usize, 9usize, 3usize);
        let xs = Tensor::randn(0f32, 2.0, (b, t, h, d), device).unwrap();
        let weight = Tensor::randn(1f32, 0.2, (d,), device).unwrap();
        let inv: Vec<f32> = (0..d / 2)
            .map(|i| 10000f32.powf(-((2 * i) as f32) / d as f32))
            .collect();
        let freqs: Vec<f32> = (0..t)
            .flat_map(|p| inv.iter().map(move |f| p as f32 * f))
            .collect();
        let freqs = Tensor::from_vec(freqs, (t, d / 2), device).unwrap();
        let cast = |x: Tensor| x.to_dtype(dtype).unwrap();
        (
            cast(xs),
            cast(weight),
            cast(freqs.cos().unwrap()),
            cast(freqs.sin().unwrap()),
        )
    }

    fn two_ops(xs: &Tensor, weight: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
        let normed = candle_nn::ops::rms_norm(xs, weight, 1e-6).unwrap();
        candle_nn::rotary_emb::rope_thd(&normed, cos, sin).unwrap()
    }

    fn fused(xs: &Tensor, weight: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
        let op = QkNormRope {
            weight: weight.clone(),
            eps: 1e-6,
        };
        xs.apply_op3_no_bwd(cos, sin, &op).unwrap()
    }

    #[test]
    fn cpu_op_matches_rms_norm_then_rope() {
        let device = Device::Cpu;
        for d in [16usize, 128] {
            let (xs, weight, cos, sin) = inputs(&device, DType::F32, d);
            let expected = two_ops(&xs, &weight, &cos, &sin);
            let got = fused(&xs, &weight, &cos, &sin);
            assert_eq!(got.dims(), expected.dims());
            assert!(max_abs_diff(&got, &expected) < 1e-5, "d={d} differs");
            // The rotation and the norm both moved the values, so this is not identity vs identity.
            assert!(max_abs_diff(&got, &xs) > 1e-1);
            // The CPU path of the dispatcher is the two ops themselves.
            let dispatched = rms_norm_rope_thd(&xs, &weight, &cos, &sin, 1e-6).unwrap();
            assert_eq!(max_abs_diff(&dispatched, &expected), 0.0);

            // In bf16 and f16 the op works in f32 and rounds once, as the Metal kernel does.
            for dtype in [DType::BF16, DType::F16] {
                let cast = |x: &Tensor| x.to_dtype(dtype).unwrap();
                let f32 = |x: &Tensor| x.to_dtype(DType::F32).unwrap();
                let (xs, weight, cos, sin) = (cast(&xs), cast(&weight), cast(&cos), cast(&sin));
                let once = cast(&two_ops(&f32(&xs), &f32(&weight), &f32(&cos), &f32(&sin)));
                let got = fused(&xs, &weight, &cos, &sin);
                assert_eq!(got.dtype(), dtype);
                let not_once = f32(&got)
                    .ne(&f32(&once))
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap()
                    .mean_all()
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap();
                assert!(
                    not_once <= 0.01,
                    "{dtype:?} d={d}: {not_once} not rounded once"
                );
            }
        }
    }

    #[test]
    fn cpu_op_rejects_tables_shorter_than_the_sequence() {
        let device = Device::Cpu;
        let (xs, weight, cos, sin) = inputs(&device, DType::F32, 16);
        let short = |x: &Tensor| x.narrow(0, 0, 4).unwrap();
        let op = QkNormRope { weight, eps: 1e-6 };
        let err = xs.apply_op3_no_bwd(&short(&cos), &short(&sin), &op);
        assert!(err.is_err(), "a [4, D/2] table was taken for 9 positions");
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_kernel_matches_rms_norm_then_rope() {
        let device = Device::new_metal(0).unwrap();
        for d in [16usize, 128] {
            // f32: the same math, so only the f32 sum order and fast-math rounding separate them.
            let (xs, weight, cos, sin) = inputs(&device, DType::F32, d);
            let reference = two_ops(&xs, &weight, &cos, &sin);
            let got = rms_norm_rope_thd(&xs, &weight, &cos, &sin, 1e-6).unwrap();
            assert_eq!(got.dims(), reference.dims());
            assert!(max_abs_diff(&got, &reference) < 1e-5, "f32 d={d} differs");
            assert!(max_abs_diff(&got, &xs) > 1e-1);

            // bf16 and f16: the kernel works in f32 and rounds once at the store, so it gives the
            // f32 result of the same rounded inputs rounded once, where the two kernels round
            // three times. Measured against that f32 reference.
            for dtype in [DType::BF16, DType::F16] {
                let cast = |x: &Tensor| x.to_dtype(dtype).unwrap();
                let f32 = |x: &Tensor| x.to_dtype(DType::F32).unwrap();
                let (xs, weight, cos, sin) = (cast(&xs), cast(&weight), cast(&cos), cast(&sin));
                let reference = two_ops(&f32(&xs), &f32(&weight), &f32(&cos), &f32(&sin));
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
                let got = rms_norm_rope_thd(&xs, &weight, &cos, &sin, 1e-6).unwrap();
                assert_eq!(got.dtype(), dtype);
                let rounded = cast(&reference);
                let two_ops_t = two_ops(&xs, &weight, &cos, &sin);
                // The share of values that are not the f32 result rounded once.
                let not_rounded = f32(&got)
                    .ne(&f32(&rounded))
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap()
                    .mean_all()
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap();
                let (one, two, once) = (mean_err(&got), mean_err(&two_ops_t), mean_err(&rounded));
                let (max_one, max_once) = (
                    max_abs_diff(&got, &reference),
                    max_abs_diff(&rounded, &reference),
                );
                eprintln!(
                    "{dtype:?} d={d}: not rounded once {not_rounded}, mean error {one} (rounded \
                     once {once}, two ops {two}), max error {max_one} (rounded once {max_once})"
                );
                // Measured: every value is the f32 result rounded once, and the mean error is
                // about half the two kernels'. A fast-math f32 can still land one value on the
                // other side of a rounding tie, hence 1% rather than none.
                assert!(
                    not_rounded <= 0.01,
                    "{dtype:?} d={d}: {not_rounded} of the values are not rounded once"
                );
                assert!(
                    one <= 0.75 * two,
                    "{dtype:?} d={d}: mean error {one}, the two ops {two}"
                );
                assert!(
                    max_one <= 1.01 * max_once,
                    "{dtype:?} d={d}: max error {max_one}, rounded once {max_once}"
                );
            }
        }
    }
}
