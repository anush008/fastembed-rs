//! Similarity helpers for comparing embeddings.
//!
//! ```
//! use fastembed::similarity::{cosine_similarity, top_k};
//!
//! let query = [1.0, 0.0, 0.0];
//! let corpus = [vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.7, 0.7, 0.0]];
//!
//! assert_eq!(cosine_similarity(&query, &corpus[0]), 1.0);
//! assert_eq!(top_k(&query, &corpus, 2)[0].0, 0); // closest is index 0
//! ```

/// Dot product. Stops at the shorter slice.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn dot_f64(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum()
}

/// Cosine similarity in `[-1.0, 1.0]`. `0.0` if either vector is all-zeros.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let na = dot_f64(a, a).sqrt();
    let nb = dot_f64(b, b).sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot_f64(a, b) / (na * nb)).clamp(-1.0, 1.0) as f32
}

/// The `k` closest vectors in `corpus` to `query` as `(index, score)`, best first.
///
/// `corpus` is any slice whose items deref to `&[f32]` (`&[Vec<f32>]`, `&[&[f32]]`, ...).
pub fn top_k<V: AsRef<[f32]>>(query: &[f32], corpus: &[V], k: usize) -> Vec<(usize, f32)> {
    let mut scored: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i, cosine_similarity(query, v.as_ref())))
        .collect();
    scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    scored.truncate(k);
    scored
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_handles_large_finite_vectors() {
        assert_eq!(cosine_similarity(&[1e30], &[1e30]), 1.0);
        assert_eq!(cosine_similarity(&[1e30], &[-1e30]), -1.0);
    }

    #[test]
    fn cosine_handles_small_finite_vectors() {
        assert_eq!(cosine_similarity(&[1e-24], &[1e-24]), 1.0);
        assert_eq!(cosine_similarity(&[1e-24], &[0.0]), 0.0);
    }

    #[test]
    fn top_k_ranks_finite_vectors_of_different_magnitudes() {
        let corpus = [[-1e30, 0.0], [0.0, 1e-24], [1e30, 0.0]];
        let hits = top_k(&[1e-24, 0.0], &corpus, 3);
        assert_eq!(hits, vec![(2, 1.0), (1, 0.0), (0, -1.0)]);
    }

    #[test]
    fn similarity_and_ranking() {
        let q = [1.0, 0.0, 0.0];
        assert_eq!(dot(&q, &[2.0, 3.0, 4.0]), 2.0);
        assert_eq!(cosine_similarity(&q, &[1.0, 0.0, 0.0]), 1.0);
        assert_eq!(cosine_similarity(&q, &[-1.0, 0.0, 0.0]), -1.0);
        assert_eq!(cosine_similarity(&q, &[0.0, 1.0, 0.0]), 0.0);
        // all-zeros guard, no NaN
        assert_eq!(cosine_similarity(&q, &[0.0, 0.0, 0.0]), 0.0);

        let corpus = vec![
            vec![0.0, 1.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.7, 0.7, 0.0],
        ];
        let hits = top_k(&q, &corpus, 2);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].0, 1); // exact match ranks first
        assert_eq!(hits[1].0, 2); // partial overlap second
        assert!(hits[0].1 > hits[1].1);

        // same call compiles over &[&[f32]] and &[[f32; N]]
        let refs: Vec<&[f32]> = corpus.iter().map(|v| v.as_slice()).collect();
        assert_eq!(top_k(&q, &refs, 1)[0].0, 1);
        assert_eq!(top_k(&q, &[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], 1)[0].0, 1);
    }
}
