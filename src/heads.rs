//! Small classifier heads over embedding vectors, many heads in one multiply.
//!
//! A head reads one text's embedding vector and returns a probability for each of its words (for
//! example `yes` and `no`). Each head is a pair of files:
//!
//! - `<name>.json`: `name`, `words`, `head_type` (`linear` or `two-layer`), `input_dim`, and
//!   optionally `input` (only `one text's vector` is supported). Other fields are ignored.
//! - `<name>.safetensors`, f32: `scale.mean` and `scale.inv` (`[input_dim]`), then for a linear
//!   head `out.weight` (`[words, input_dim]`) and `out.bias`, or for a two-layer head
//!   `hidden.weight` (`[hidden, input_dim]`), `hidden.bias`, `out.weight` (`[words, hidden]`) and
//!   `out.bias`.
//!
//! The forward of one head: `x = (v - scale.mean) * scale.inv`, then `out(x)` (linear) or
//! `out(relu(hidden(x)))` (two-layer), each layer `y = x W^T + b`, then a softmax over the words.
//!
//! [`EmbeddingHeads`] folds each head's scale into its first layer and stacks every head's first
//! layer into one matrix, so all heads on a batch of vectors cost one multiply, split afterwards.

use std::path::{Path, PathBuf};

use ndarray::{s, Array1, Array2, ArrayView2, Axis};
use serde::Deserialize;

use crate::common::{Error, Result};

#[derive(Deserialize)]
struct HeadFile {
    name: String,
    words: Vec<String>,
    head_type: String,
    input_dim: usize,
    #[serde(default)]
    input: Option<String>,
}

/// One head's place in the stacked first layer, and its second layer if it has one.
struct Head {
    name: String,
    words: Vec<String>,
    /// The head's columns in the stacked first layer: `offset..offset + width`.
    offset: usize,
    width: usize,
    /// A two-layer head's output layer: `W` `[words, hidden]` and `b` `[words]`.
    out: Option<(Array2<f32>, Array1<f32>)>,
}

/// A set of heads over vectors of one length, run together. See the module docs for the files.
pub struct EmbeddingHeads {
    heads: Vec<Head>,
    /// Every head's first layer, scale folded in, as `[input_dim, total width]`.
    stacked: Array2<f32>,
    bias: Array1<f32>,
    input_dim: usize,
}

fn tensor(
    st: &safetensors::SafeTensors,
    name: &str,
    path: &Path,
) -> Result<(Vec<usize>, Vec<f32>)> {
    let view = st
        .tensor(name)
        .map_err(|e| Error::Other(format!("{}: `{name}`: {e}", path.display())))?;
    if view.dtype() != safetensors::Dtype::F32 {
        return Err(Error::Other(format!(
            "{}: `{name}` is {:?}, not f32",
            path.display(),
            view.dtype()
        )));
    }
    let (chunks, rest) = view.data().as_chunks::<4>();
    if !rest.is_empty() {
        return Err(Error::Other(format!(
            "{}: `{name}` is not a whole number of f32s",
            path.display()
        )));
    }
    let data = chunks.iter().map(|b| f32::from_le_bytes(*b)).collect();
    Ok((view.shape().to_vec(), data))
}

fn matrix(st: &safetensors::SafeTensors, name: &str, path: &Path) -> Result<Array2<f32>> {
    let (shape, data) = tensor(st, name, path)?;
    let [rows, cols] = shape[..] else {
        return Err(Error::InvalidShape(format!(
            "{}: `{name}` has shape {shape:?}, not two dimensions",
            path.display()
        )));
    };
    Array2::from_shape_vec((rows, cols), data).map_err(|e| Error::InvalidShape(e.to_string()))
}

fn vector(
    st: &safetensors::SafeTensors,
    name: &str,
    len: usize,
    path: &Path,
) -> Result<Array1<f32>> {
    let (shape, data) = tensor(st, name, path)?;
    if shape != [len] {
        return Err(Error::InvalidShape(format!(
            "{}: `{name}` has shape {shape:?}, not [{len}]",
            path.display()
        )));
    }
    Ok(Array1::from(data))
}

impl EmbeddingHeads {
    /// Load the heads of these `(json, safetensors)` file pairs, in this order. Every head must
    /// read vectors of the same length.
    pub fn from_files(files: &[(PathBuf, PathBuf)]) -> Result<Self> {
        let mut heads = Vec::with_capacity(files.len());
        let mut firsts: Vec<(Array2<f32>, Array1<f32>)> = Vec::with_capacity(files.len());
        let mut input_dim = None;
        let mut offset = 0;
        for (json, weights) in files {
            let file: HeadFile = serde_json::from_slice(&std::fs::read(json)?)
                .map_err(|e| Error::Other(format!("{}: {e}", json.display())))?;
            if let Some(input) = &file.input {
                if input != "one text's vector" {
                    return Err(Error::InvalidArgument(format!(
                        "head `{}` reads `{input}`; only `one text's vector` is supported",
                        file.name
                    )));
                }
            }
            if *input_dim.get_or_insert(file.input_dim) != file.input_dim {
                return Err(Error::InvalidArgument(format!(
                    "head `{}` reads vectors of {} numbers, the others {}",
                    file.name,
                    file.input_dim,
                    input_dim.unwrap_or_default()
                )));
            }
            let bytes = std::fs::read(weights)?;
            let st = safetensors::SafeTensors::deserialize(&bytes)
                .map_err(|e| Error::Other(format!("{}: {e}", weights.display())))?;
            let mean = vector(&st, "scale.mean", file.input_dim, weights)?;
            let inv = vector(&st, "scale.inv", file.input_dim, weights)?;
            let (first, out) = match file.head_type.as_str() {
                "linear" => ("out", None),
                "two-layer" => {
                    let w = matrix(&st, "out.weight", weights)?;
                    let b = vector(&st, "out.bias", w.nrows(), weights)?;
                    ("hidden", Some((w, b)))
                }
                other => {
                    return Err(Error::InvalidArgument(format!(
                        "head `{}` has head_type `{other}`, not `linear` or `two-layer`",
                        file.name
                    )))
                }
            };
            let w = matrix(&st, &format!("{first}.weight"), weights)?;
            let b = vector(&st, &format!("{first}.bias"), w.nrows(), weights)?;
            if w.ncols() != file.input_dim {
                return Err(Error::InvalidShape(format!(
                    "{}: `{first}.weight` reads {} numbers, not input_dim {}",
                    weights.display(),
                    w.ncols(),
                    file.input_dim
                )));
            }
            let words = out.as_ref().map_or(w.nrows(), |(w, _)| w.nrows());
            if words != file.words.len() {
                return Err(Error::InvalidShape(format!(
                    "{}: {words} outputs for {} words",
                    weights.display(),
                    file.words.len()
                )));
            }
            if let Some((w2, _)) = &out {
                if w2.ncols() != w.nrows() {
                    return Err(Error::InvalidShape(format!(
                        "{}: `out.weight` reads {} numbers, the hidden layer gives {}",
                        weights.display(),
                        w2.ncols(),
                        w.nrows()
                    )));
                }
            }
            // (v - mean) * inv, then x W^T + b, is v (W * inv)^T + (b - W (mean * inv)).
            let folded_w = &w * &inv.view().insert_axis(Axis(0));
            let folded_b = &b - &w.dot(&(&mean * &inv));
            let width = w.nrows();
            heads.push(Head {
                name: file.name,
                words: file.words,
                offset,
                width,
                out,
            });
            offset += width;
            firsts.push((folded_w, folded_b));
        }
        let input_dim = input_dim.unwrap_or_default();
        let mut stacked = Array2::zeros((input_dim, offset));
        let mut bias = Array1::zeros(offset);
        for (head, (w, b)) in heads.iter().zip(&firsts) {
            let cols = head.offset..head.offset + head.width;
            stacked.slice_mut(s![.., cols.clone()]).assign(&w.t());
            bias.slice_mut(s![cols]).assign(b);
        }
        Ok(Self {
            heads,
            stacked,
            bias,
            input_dim,
        })
    }

    /// Load every head in `dir` (each `<name>.json` with its `<name>.safetensors`), by name.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let mut names: Vec<String> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name().into_string().ok()?;
                let stem = name.strip_suffix(".json")?;
                (!stem.ends_with(".check")).then(|| stem.to_string())
            })
            .filter(|stem| dir.join(format!("{stem}.safetensors")).exists())
            .collect();
        names.sort();
        let files: Vec<(PathBuf, PathBuf)> = names
            .iter()
            .map(|n| {
                (
                    dir.join(format!("{n}.json")),
                    dir.join(format!("{n}.safetensors")),
                )
            })
            .collect();
        Self::from_files(&files)
    }

    /// The heads' names, in the order `probs` returns them.
    pub fn names(&self) -> Vec<&str> {
        self.heads.iter().map(|h| h.name.as_str()).collect()
    }

    /// The words of head `index`, in the order of its probabilities.
    pub fn words(&self, index: usize) -> Option<&[String]> {
        self.heads.get(index).map(|h| h.words.as_slice())
    }

    /// Each head's word probabilities for each vector: `[head][vector][word]`. All heads' first
    /// layers run as one multiply.
    pub fn probs(&self, vectors: &[Vec<f32>]) -> Result<Vec<Vec<Vec<f32>>>> {
        let mut flat = Vec::with_capacity(vectors.len() * self.input_dim);
        for v in vectors {
            if v.len() != self.input_dim {
                return Err(Error::InvalidArgument(format!(
                    "a vector of {} numbers for heads that read {}",
                    v.len(),
                    self.input_dim
                )));
            }
            flat.extend_from_slice(v);
        }
        let x = ArrayView2::from_shape((vectors.len(), self.input_dim), &flat)
            .map_err(|e| Error::InvalidShape(e.to_string()))?;
        let first = x.dot(&self.stacked) + &self.bias;
        Ok(self
            .heads
            .iter()
            .map(|head| {
                let mine = first.slice(s![.., head.offset..head.offset + head.width]);
                let logits = match &head.out {
                    None => mine.to_owned(),
                    Some((w, b)) => mine.mapv(|v| v.max(0.0)).dot(&w.t()) + b,
                };
                logits
                    .rows()
                    .into_iter()
                    .map(|r| softmax(&r.to_vec()))
                    .collect()
            })
            .collect())
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::EmbeddingHeads;
    use std::path::{Path, PathBuf};

    /// Numbers in [-1, 1) from a fixed seed, so each test writes the same heads.
    fn numbers(seed: u64, n: usize) -> Vec<f32> {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// A head's weights by name: `(name, shape, values)`.
    type Weights = Vec<(String, Vec<usize>, Vec<f32>)>;

    /// A head with random weights: linear when `hidden` is `None`.
    fn random_head(seed: u64, dim: usize, words: usize, hidden: Option<usize>) -> Weights {
        let mut w: Weights = vec![
            ("scale.mean".into(), vec![dim], numbers(seed, dim)),
            (
                "scale.inv".into(),
                vec![dim],
                numbers(seed + 1, dim).iter().map(|x| 1.5 + x).collect(),
            ),
        ];
        let first = hidden.unwrap_or(words);
        let name = if hidden.is_some() { "hidden" } else { "out" };
        w.push((
            format!("{name}.weight"),
            vec![first, dim],
            numbers(seed + 2, first * dim),
        ));
        w.push((
            format!("{name}.bias"),
            vec![first],
            numbers(seed + 3, first),
        ));
        if let Some(h) = hidden {
            w.push((
                "out.weight".into(),
                vec![words, h],
                numbers(seed + 4, words * h),
            ));
            w.push(("out.bias".into(), vec![words], numbers(seed + 5, words)));
        }
        w
    }

    fn write_head(
        dir: &Path,
        name: &str,
        words: &[&str],
        dim: usize,
        weights: &Weights,
    ) -> (PathBuf, PathBuf) {
        let two_layer = weights.iter().any(|(n, _, _)| n == "hidden.weight");
        let json = dir.join(format!("{name}.json"));
        let file = serde_json::json!({
            "name": name,
            "words": words,
            "head_type": if two_layer { "two-layer" } else { "linear" },
            "input_dim": dim,
            "input": "one text's vector",
        });
        std::fs::write(&json, file.to_string()).unwrap();
        let bytes: Vec<(String, Vec<usize>, Vec<u8>)> = weights
            .iter()
            .map(|(n, s, v)| {
                (
                    n.clone(),
                    s.clone(),
                    v.iter().flat_map(|x| x.to_le_bytes()).collect(),
                )
            })
            .collect();
        let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = bytes
            .iter()
            .map(|(n, s, b)| {
                (
                    n.clone(),
                    safetensors::tensor::TensorView::new(safetensors::Dtype::F32, s.clone(), b)
                        .unwrap(),
                )
            })
            .collect();
        let st = dir.join(format!("{name}.safetensors"));
        std::fs::write(&st, safetensors::tensor::serialize(views, None).unwrap()).unwrap();
        (json, st)
    }

    /// The head's forward exactly as its file describes it, one vector at a time.
    fn by_the_formula(weights: &Weights, v: &[f32]) -> Vec<f32> {
        let get = |n: &str| weights.iter().find(|(m, _, _)| m == n).unwrap();
        let layer = |x: &[f32], n: &str| -> Vec<f32> {
            let (_, shape, w) = get(&format!("{n}.weight"));
            let (_, _, b) = get(&format!("{n}.bias"));
            (0..shape[0])
                .map(|o| {
                    b[o] + (0..shape[1])
                        .map(|i| x[i] * w[o * shape[1] + i])
                        .sum::<f32>()
                })
                .collect()
        };
        let (mean, inv) = (&get("scale.mean").2, &get("scale.inv").2);
        let x: Vec<f32> = v
            .iter()
            .zip(mean)
            .zip(inv)
            .map(|((v, m), s)| (v - m) * s)
            .collect();
        let logits = if weights.iter().any(|(n, _, _)| n == "hidden.weight") {
            let h: Vec<f32> = layer(&x, "hidden")
                .into_iter()
                .map(|y| y.max(0.0))
                .collect();
            layer(&h, "out")
        } else {
            layer(&x, "out")
        };
        super::softmax(&logits)
    }

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("fastembed-heads-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn stacked_heads_give_each_head_its_own_answers() {
        let dir = scratch("stacked");
        let dim = 24;
        let a = random_head(10, dim, 2, Some(8));
        let b = random_head(20, dim, 3, None);
        let c = random_head(30, dim, 2, Some(5));
        let files = vec![
            write_head(&dir, "a", &["yes", "no"], dim, &a),
            write_head(&dir, "b", &["x", "y", "z"], dim, &b),
            write_head(&dir, "c", &["open", "close"], dim, &c),
        ];
        let vectors: Vec<Vec<f32>> = (0..4).map(|i| numbers(100 + i, dim)).collect();

        let all = EmbeddingHeads::from_files(&files).unwrap();
        assert_eq!(all.names(), ["a", "b", "c"]);
        assert_eq!(all.words(1).unwrap(), ["x", "y", "z"]);
        let together = all.probs(&vectors).unwrap();

        for (h, weights) in [&a, &b, &c].iter().enumerate() {
            let alone = EmbeddingHeads::from_files(&files[h..h + 1])
                .unwrap()
                .probs(&vectors)
                .unwrap();
            for (t, v) in vectors.iter().enumerate() {
                let want = by_the_formula(weights, v);
                for (w, (got, one)) in want.iter().zip(together[h][t].iter().zip(&alone[0][t])) {
                    assert!(
                        (got - w).abs() < 1e-5,
                        "head {h} text {t}: {got} against {w}"
                    );
                    assert!(
                        (got - one).abs() < 1e-6,
                        "head {h} text {t}: stacked {got}, alone {one}"
                    );
                }
            }
        }
        // The heads disagree, so answers taken from the wrong columns would show.
        assert_ne!(together[0][0], together[2][0]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn heads_refuse_what_they_cannot_read() {
        let dir = scratch("refuse");
        let a = write_head(&dir, "a", &["yes", "no"], 8, &random_head(1, 8, 2, None));
        let b = write_head(&dir, "b", &["yes", "no"], 6, &random_head(2, 6, 2, None));
        let err = EmbeddingHeads::from_files(&[a.clone(), b])
            .err()
            .expect("mixed lengths load");
        assert!(err.to_string().contains("vectors of 6 numbers"), "{err}");

        let heads = EmbeddingHeads::from_files(std::slice::from_ref(&a)).unwrap();
        assert!(heads.probs(&[vec![0.0; 7]]).is_err());

        let mut file: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&a.0).unwrap()).unwrap();
        file["input"] =
            "two texts: the product of their vectors, then the absolute difference".into();
        std::fs::write(&a.0, file.to_string()).unwrap();
        assert!(EmbeddingHeads::from_files(&[a]).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn from_dir_loads_every_head_by_name_and_skips_check_files() {
        let dir = scratch("dir");
        write_head(&dir, "zeta", &["yes", "no"], 8, &random_head(3, 8, 2, None));
        write_head(
            &dir,
            "alpha",
            &["yes", "no"],
            8,
            &random_head(4, 8, 2, Some(4)),
        );
        std::fs::write(dir.join("alpha.check.json"), "[]").unwrap();
        let heads = EmbeddingHeads::from_dir(&dir).unwrap();
        assert_eq!(heads.names(), ["alpha", "zeta"]);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
