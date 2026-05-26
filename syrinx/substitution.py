"""Stage 5 — Empirically derived substitution matrix construction."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist

from .config import Config
from .utils import save_array

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_substitution_matrix(
    cfg: Config,
    vocabulary: dict[str, Any],
    syllables: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build acoustically derived substitution matrices.

    Computes three matrices (at the 50th, 75th, and 95th percentile mismatch
    penalties) and selects the primary (95th percentile) for downstream use.

    Parameters
    ----------
    cfg:
        Pipeline configuration.
    vocabulary:
        Vocabulary dict from :func:`~syrinx.vocabulary.build_vocabulary`.
    syllables:
        Feature-augmented syllable records. Required when
        ``cfg.use_temporal_features`` is True.

    Returns
    -------
    dict
        Keys:
        - ``matrices``: dict mapping percentile → ``(n × n)`` ndarray
        - ``primary_matrix``: the selected primary matrix
        - ``cluster_letters``: label mapping from vocabulary
        - ``match_scores``: per-cluster match scores
        - ``gap_open``: selected gap open penalty
        - ``gap_extend``: selected gap extend penalty
        - ``label_order``: list of cluster integer labels in matrix row/col order
    """
    centroids = vocabulary["centroids"]
    labels_array = vocabulary["labels"]
    cluster_letters = vocabulary["cluster_letters"]
    unique_labels = sorted(cluster_letters.keys())
    n_clusters = len(unique_labels)

    X = (
        np.vstack([s["features"] for s in syllables]).astype(np.float64)
        if syllables else None
    )

    logger.info("Building substitution matrix for %d clusters", n_clusters)

    # Match scores: within-cluster mean pairwise Euclidean distance, negated
    match_scores = _compute_match_scores(labels_array, unique_labels, X, centroids)

    # Mismatch penalties at each configured percentile
    matrices: dict[int, np.ndarray] = {}
    for pct in cfg.mismatch_percentiles:
        mat = _build_matrix(centroids, match_scores, unique_labels, percentile=pct)
        matrices[pct] = mat
        logger.debug("Built substitution matrix at %dth percentile", pct)

    primary_matrix = matrices[cfg.primary_mismatch_percentile]

    # Gap penalty selection
    gap_open, gap_extend = _select_gap_penalties(
        cfg, vocabulary, syllables, primary_matrix, unique_labels, cluster_letters
    )
    logger.info("Selected gap penalties: open=%.2f, extend=%.2f", gap_open, gap_extend)

    return {
        "matrices": matrices,
        "primary_matrix": primary_matrix,
        "cluster_letters": cluster_letters,
        "match_scores": match_scores,
        "gap_open": gap_open,
        "gap_extend": gap_extend,
        "label_order": unique_labels,
        "n_clusters": n_clusters,
    }


# ---------------------------------------------------------------------------
# Matrix construction
# ---------------------------------------------------------------------------

def _compute_match_scores(
    labels_array: np.ndarray,
    unique_labels: list[int],
    X: np.ndarray | None,
    centroids: np.ndarray,
) -> np.ndarray:
    """Compute per-cluster match scores from mean within-cluster Euclidean distance.

    Parameters
    ----------
    labels_array:
        Full label array from HDBSCAN.
    unique_labels:
        Sorted cluster integer labels.
    X:
        Feature matrix (may be None if only centroids available).
    centroids:
        Centroid array.

    Returns
    -------
    np.ndarray
        Shape ``(n_clusters,)`` — positive match scores (negated mean distance).
    """
    match_scores = np.zeros(len(unique_labels))
    for i, lb in enumerate(unique_labels):
        if X is not None:
            pts = X[labels_array == lb]
            if len(pts) > 1:
                pw = cdist(pts, pts, metric="euclidean")
                mean_dist = pw[np.triu_indices(len(pts), k=1)].mean()
            else:
                mean_dist = 0.0
        else:
            mean_dist = 0.0
        match_scores[i] = -mean_dist  # negate → positive match score
    return match_scores


def _build_matrix(
    centroids: np.ndarray,
    match_scores: np.ndarray,
    unique_labels: list[int],
    percentile: int,
) -> np.ndarray:
    """Construct a single substitution matrix at the given mismatch percentile.

    Parameters
    ----------
    centroids:
        Cluster centroid array, shape ``(n_clusters, n_features)``.
    match_scores:
        Per-cluster match scores, shape ``(n_clusters,)``.
    unique_labels:
        Sorted cluster labels (determines row/column order).
    percentile:
        Percentile of between-cluster distances to use as max mismatch penalty.

    Returns
    -------
    np.ndarray
        Symmetric substitution matrix of shape ``(n_clusters, n_clusters)``.
    """
    n = len(unique_labels)
    pairwise = cdist(centroids, centroids, metric="euclidean")

    # Upper triangle of between-cluster distances
    off_diag = pairwise[np.triu_indices(n, k=1)]
    max_penalty = float(np.percentile(off_diag, percentile))

    # Normalise between-cluster distances to [0, 1]
    max_dist = float(pairwise.max()) if pairwise.max() > 0 else 1.0
    normalised = pairwise / max_dist

    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i, j] = match_scores[i]
            else:
                # Linear scale from 0 to max_penalty
                mat[i, j] = -(normalised[i, j] * max_penalty)

    return mat


# ---------------------------------------------------------------------------
# Gap penalty selection
# ---------------------------------------------------------------------------

def _select_gap_penalties(
    cfg: Config,
    vocabulary: dict[str, Any],
    syllables: list[dict[str, Any]] | None,
    sub_matrix: np.ndarray,
    unique_labels: list[int],
    cluster_letters: dict[int, str],
) -> tuple[float, float]:
    """Grid search for gap penalties that maximise within- vs between-species separation.

    Parameters
    ----------
    cfg:
        Pipeline configuration.
    vocabulary:
        Vocabulary dict.
    syllables:
        Feature-augmented syllable records.
    sub_matrix:
        Primary substitution matrix.
    unique_labels:
        Cluster label order.
    cluster_letters:
        Mapping from cluster int to letter.

    Returns
    -------
    tuple[float, float]
        ``(gap_open, gap_extend)`` penalties.
    """
    if syllables is None or len(syllables) < 10:
        logger.warning("Insufficient syllables for gap penalty selection; using defaults")
        return cfg.gap_open_grid[0], cfg.gap_extend_grid[0]

    # Build song strings for held-out species
    species_strings = _build_song_strings_for_gap_search(
        syllables, vocabulary["labels"], cluster_letters, cfg
    )
    if len(species_strings) < 4:
        return cfg.gap_open_grid[0], cfg.gap_extend_grid[0]

    holdout = dict(list(species_strings.items())[:cfg.gap_penalty_holdout_n_species])

    best_sep = -np.inf
    best_gap_open = cfg.gap_open_grid[0]
    best_gap_extend = cfg.gap_extend_grid[0]

    letter_to_int = {v: k for k, v in cluster_letters.items()}

    for gap_open in cfg.gap_open_grid:
        for gap_extend in cfg.gap_extend_grid:
            sep = _evaluate_gap_penalty(
                holdout, sub_matrix, unique_labels, letter_to_int,
                gap_open, gap_extend
            )
            if sep > best_sep:
                best_sep = sep
                best_gap_open = gap_open
                best_gap_extend = gap_extend

    return best_gap_open, best_gap_extend


def _build_song_strings_for_gap_search(
    syllables: list[dict[str, Any]],
    labels: np.ndarray,
    cluster_letters: dict[int, str],
    cfg: Config,
) -> dict[str, str]:
    """Build per-species song strings for gap penalty evaluation.

    Parameters
    ----------
    syllables:
        Syllable records with labels applied.
    labels:
        Cluster label array.
    cluster_letters:
        Label-to-letter mapping.
    cfg:
        Pipeline configuration.
    """
    species_seqs: dict[str, list[str]] = {}
    for i, syl in enumerate(syllables):
        lb = labels[i]
        if lb == -1:
            continue
        letter = cluster_letters.get(lb, "?")
        sp = syl.get("species", "unknown")
        species_seqs.setdefault(sp, []).append(letter)

    return {sp: "".join(seq) for sp, seq in species_seqs.items() if len(seq) >= 5}


def _evaluate_gap_penalty(
    species_strings: dict[str, str],
    sub_matrix: np.ndarray,
    unique_labels: list[int],
    letter_to_int: dict[str, int],
    gap_open: float,
    gap_extend: float,
) -> float:
    """Compute separation between within- and between-species alignment scores.

    Parameters
    ----------
    species_strings:
        Per-species song strings.
    sub_matrix:
        Substitution matrix.
    unique_labels:
        Label order for matrix indexing.
    letter_to_int:
        Letter-to-cluster-int mapping.
    gap_open:
        Gap open penalty.
    gap_extend:
        Gap extend penalty.
    """
    from Bio.Align import PairwiseAligner

    label_to_idx = {lb: i for i, lb in enumerate(unique_labels)}
    species = sorted(species_strings.keys())
    n = len(species)

    self_scores = []
    cross_scores = []

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.open_gap_score = gap_open
    aligner.extend_gap_score = gap_extend

    for i in range(min(n, 10)):
        s_i = species_strings[species[i]]
        score_ii = _align_score(s_i, s_i, sub_matrix, label_to_idx, letter_to_int, aligner)
        self_scores.append(score_ii)
        for j in range(i + 1, min(n, 10)):
            s_j = species_strings[species[j]]
            score_ij = _align_score(s_i, s_j, sub_matrix, label_to_idx, letter_to_int, aligner)
            cross_scores.append(score_ij)

    if not self_scores or not cross_scores:
        return 0.0

    mean_self = np.mean(self_scores)
    mean_cross = np.mean(cross_scores)
    return float(mean_self - mean_cross)


def _align_score(
    seq1: str,
    seq2: str,
    sub_matrix: np.ndarray,
    label_to_idx: dict[int, int],
    letter_to_int: dict[str, int],
    aligner: Any,
) -> float:
    """Compute alignment score for two letter strings.

    Parameters
    ----------
    seq1, seq2:
        Letter strings.
    sub_matrix:
        Substitution matrix.
    label_to_idx:
        Mapping from cluster int to matrix row/col index.
    letter_to_int:
        Mapping from letter to cluster int.
    aligner:
        BioPython PairwiseAligner.
    """
    try:
        score = float(aligner.score(seq1, seq2))
    except Exception:
        score = 0.0
    return score
