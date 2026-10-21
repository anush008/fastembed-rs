//! candle's fused attention, writing its output as `[B, T, H, D]`.
//!
//! `candle_nn::ops::sdpa` hands its kernel the strides of a contiguous `[B, H, T, D]` output, so
//! the attention transposes it back and copies it before `o_proj`. The kernel itself
//! (`candle_metal_kernels::call_sdpa_full`) writes through whatever output strides it is given;
//! [`sdpa_bthd`] gives it those of `[B, T, H, D]`, and the copy goes away.

use candle_core::{Result, Tensor};

/// `sdpa(q, k, v, mask, do_causal, scale, 1.0).transpose(1, 2)`, contiguous: q `[B, H, T, D]`,
/// k and v `[B, Hkv, S, D]` (any strides with the head dim contiguous), the mask
/// `[B, H, T, S]`. On Metal with more than one query position this is candle's full attention
/// kernel writing `[B, T, H, D]` directly; otherwise it is `candle_nn::ops::sdpa` and a copy.
pub(crate) fn sdpa_bthd(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    do_causal: bool,
    scale: f32,
) -> Result<Tensor> {
    #[cfg(feature = "metal")]
    if q.device().is_metal() && q.dim(2)? > 1 {
        let op = metal::SdpaBthd {
            scale,
            mask: mask.cloned(),
            do_causal,
        };
        return q.apply_op3_no_bwd(k, v, &op);
    }
    candle_nn::ops::sdpa(q, k, v, mask, do_causal, scale, 1.0)?
        .transpose(1, 2)?
        .contiguous()
}

#[cfg(feature = "metal")]
mod metal {
    use candle_core::backend::BackendStorage;
    use candle_core::{CpuStorage, DType, Layout, MetalStorage, Result, Shape, Tensor};
    use candle_metal_kernels::SdpaDType;

    pub(super) struct SdpaBthd {
        pub(super) scale: f32,
        pub(super) mask: Option<Tensor>,
        pub(super) do_causal: bool,
    }

    fn sdpa_dtype(dtype: DType) -> Result<SdpaDType> {
        match dtype {
            DType::BF16 => Ok(SdpaDType::BF16),
            DType::F16 => Ok(SdpaDType::F16),
            DType::F32 => Ok(SdpaDType::F32),
            other => candle_core::bail!("sdpa-bthd does not take {other:?}"),
        }
    }

    impl candle_core::CustomOp3 for SdpaBthd {
        fn name(&self) -> &'static str {
            "sdpa-bthd"
        }

        fn cpu_fwd(
            &self,
            _: &CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            candle_core::bail!(
                "sdpa-bthd runs on Metal only; sdpa_bthd takes candle's sdpa elsewhere"
            )
        }

        fn metal_fwd(
            &self,
            q: &MetalStorage,
            q_l: &Layout,
            k: &MetalStorage,
            k_l: &Layout,
            v: &MetalStorage,
            v_l: &Layout,
        ) -> Result<(MetalStorage, Shape)> {
            let (b, h, t, d) = q_l.shape().dims4()?;
            let (_, h_kv, s, k_d) = k_l.shape().dims4()?;
            let (_, v_h, v_s, v_d) = v_l.shape().dims4()?;
            if k_d != d || v_d != d || v_h != h_kv || v_s != s || !h.is_multiple_of(h_kv) {
                candle_core::bail!(
                    "sdpa-bthd shapes do not fit: q {:?} k {:?} v {:?}",
                    q_l.dims(),
                    k_l.dims(),
                    v_l.dims()
                )
            }
            if t < 2 || (self.mask.is_some() && t > s) {
                candle_core::bail!("sdpa-bthd runs the full kernel only: q {t}, k {s}")
            }
            for l in [q_l, k_l, v_l] {
                if l.stride()[3] != 1 {
                    candle_core::bail!("sdpa-bthd needs the head dim contiguous")
                }
            }
            let dtype = q.dtype();
            if k.dtype() != dtype || v.dtype() != dtype {
                candle_core::bail!("sdpa-bthd q, k and v dtypes differ")
            }
            let itype = sdpa_dtype(dtype)?;

            let device = q.device();
            let el = b * t * h * d;
            let output = device
                .new_buffer_builder()
                .with_size_for(el, dtype)
                .with_label("sdpa_bthd_o")
                .build()?;
            // The kernel's (batch, head, sequence) output strides for a `[B, T, H, D]` buffer.
            let o_strides = [t * h * d, d, h * d];

            let mask = self.mask.as_ref().map(|m| m.storage_and_layout());
            let (mask_type, mask_buffer, mask_strides) = match (&self.mask, &mask) {
                (Some(m), Some((storage, layout))) => {
                    let candle_core::Storage::Metal(storage) = &**storage else {
                        candle_core::bail!("sdpa-bthd mask is not on the metal device")
                    };
                    if layout.dims() != [b, h, t, s] {
                        candle_core::bail!(
                            "sdpa-bthd mask is {:?}, it needs {:?}",
                            layout.dims(),
                            [b, h, t, s]
                        )
                    }
                    let mask_type = sdpa_dtype(m.dtype())?;
                    if mask_type != itype {
                        candle_core::bail!("sdpa-bthd mask dtype differs from q's")
                    }
                    (
                        Some(mask_type),
                        Some(storage.buffer()),
                        Some(layout.stride().to_vec()),
                    )
                }
                _ => (None, None, None),
            };

            let size = dtype.size_in_bytes();
            let encoder = device.command_encoder()?;
            encoder.set_label("sdpa_bthd");
            candle_metal_kernels::call_sdpa_full(
                device.device(),
                &encoder,
                device.kernels(),
                q_l.start_offset() * size,
                q_l.dims(),
                q_l.stride(),
                q.buffer(),
                k_l.start_offset() * size,
                k_l.dims(),
                k_l.stride(),
                k.buffer(),
                v_l.start_offset() * size,
                v.buffer(),
                v_l.stride(),
                mask_type,
                mask_buffer,
                mask_strides.as_deref(),
                &output,
                &o_strides,
                self.scale,
                self.do_causal,
                itype,
            )
            .map_err(candle_core::Error::wrap)?;
            drop(encoder);
            Ok((
                MetalStorage::new(output, device.clone(), el, dtype),
                Shape::from_dims(&[b, t, h, d]),
            ))
        }
    }
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::sdpa_bthd;
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

    /// The `[B, T, H, D]` output is candle's `[B, H, T, D]` output transposed, to the bit,
    /// causal and with a padding mask, with q, k and v as the transposed views the model passes.
    #[test]
    fn metal_output_is_candles_output_transposed() {
        let device = Device::new_metal(0).unwrap();
        let (b, t, h, h_kv, d) = (2usize, 37usize, 4usize, 2usize, 128usize);
        for dtype in [DType::F32, DType::BF16] {
            let view = |heads: usize| {
                Tensor::randn(0f32, 1.0, (b, t, heads, d), &device)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap()
                    .transpose(1, 2)
                    .unwrap()
            };
            let (q, k, v) = (view(h), view(h_kv), view(h_kv));
            let scale = (d as f32).powf(-0.5);
            // Left padding on the first row: its first five keys are masked.
            let mut mask = vec![0f32; b * t * t];
            for i in 0..t {
                for j in 0..t {
                    if j > i || j < 5 {
                        mask[i * t + j] = -1e4;
                    }
                }
            }
            let mask = Tensor::from_vec(mask, (b, 1, t, t), &device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap()
                .broadcast_as((b, h, t, t))
                .unwrap();
            for (mask, causal) in [(None, true), (Some(&mask), false)] {
                let expected = candle_nn::ops::sdpa(&q, &k, &v, mask, causal, scale, 1.0)
                    .unwrap()
                    .transpose(1, 2)
                    .unwrap();
                let got = sdpa_bthd(&q, &k, &v, mask, causal, scale).unwrap();
                assert_eq!(got.dims(), &[b, t, h, d]);
                assert!(got.is_contiguous());
                assert_eq!(
                    max_abs_diff(&got, &expected),
                    0.0,
                    "{dtype:?} causal={causal}"
                );
                assert!(max_abs_diff(&got, &got.zeros_like().unwrap()) > 1e-2);
            }
        }
    }
}
