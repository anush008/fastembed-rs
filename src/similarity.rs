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
///
/// The dot product stops at the shorter slice, so compare vectors of one length.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let na = dot_f64(a, a).sqrt();
    let nb = dot_f64(b, b).sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot_f64(a, b) / (na * nb)).clamp(-1.0, 1.0) as f32
}

/// The `k` closest vectors in `corpus` to `query` as `(index, score)`, best first.
/// Equal scores keep their `corpus` order, and a `NaN` score ranks last.
///
/// `corpus` is any slice whose items deref to `&[f32]` (`&[Vec<f32>]`, `&[&[f32]]`, ...).
pub fn top_k<V: AsRef<[f32]>>(query: &[f32], corpus: &[V], k: usize) -> Vec<(usize, f32)> {
    let mut scored: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i, cosine_similarity(query, v.as_ref())))
        .collect();
    scored.sort_by(|a, b| by_score_descending(a.1, b.1));
    scored.truncate(k);
    scored
}

/// Best score first. A `NaN` is no score, so it sorts last whatever its sign bit.
pub(crate) fn by_score_descending(a: f32, b: f32) -> std::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (false, false) => b.total_cmp(&a),
        (a_nan, b_nan) => a_nan.cmp(&b_nan),
    }
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
    fn top_k_ranks_a_nan_score_last() {
        let corpus = [[f32::NAN, 0.0], [0.0, 1.0], [1.0, 0.0]];
        let hits = top_k(&[1.0, 0.0], &corpus, 3);
        assert_eq!(hits[..2], [(2, 1.0), (1, 0.0)]);
        assert!(hits[2].0 == 0 && hits[2].1.is_nan());
    }

    #[test]
    fn by_score_descending_sorts_a_nan_of_either_sign_last() {
        let mut scores = [f32::NAN, -1.0, -f32::NAN, f32::NEG_INFINITY, 2.0];
        scores.sort_by(|a, b| by_score_descending(*a, *b));
        assert_eq!(scores[..3], [2.0, -1.0, f32::NEG_INFINITY]);
        assert!(scores[3].is_nan() && scores[4].is_nan());
    }

    #[test]
    fn top_k_keeps_equal_scores_in_corpus_order() {
        let corpus: Vec<[f32; 2]> = (0..41)
            .map(|i| if i % 3 == 0 { [1.0, 0.0] } else { [0.0, 1.0] })
            .collect();
        let hits = top_k(&[1.0, 0.0], &corpus, 6);
        let indices: Vec<usize> = hits.iter().map(|hit| hit.0).collect();
        assert_eq!(indices, [0, 3, 6, 9, 12, 15]);
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
