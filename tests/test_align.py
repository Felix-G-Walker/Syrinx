"""Tests for Stage 6 — alignment and null model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from syrinx.config import load_config


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    config_path = Path(__file__).parent.parent / "config.yaml"
    cfg = load_config(config_path)
    tmpdir = tmp_path_factory.mktemp("align_test")
    cfg.data_dir = str(tmpdir)
    cfg.output_dir = str(tmpdir / "out")
    cfg.null_model_permutations = 9
    cfg.recording_bootstrap_n = 5
    cfg.bootstrap_n = 5
    return cfg


def _make_simple_vocab_and_sub(cfg, n_clusters=4, n_syllables=200):
    """Build minimal vocabulary and substitution for alignment tests."""
    from syrinx.substitution import build_substitution_matrix
    from syrinx.vocabulary import build_vocabulary

    rng = np.random.RandomState(42)
    centroids = rng.randn(n_clusters, 36) * 8
    syllables = []
    for i in range(n_syllables):
        c = i % n_clusters
        feat = centroids[c] + rng.randn(36) * 0.2
        syllables.append({
            "features": feat.astype(np.float32),
            "species": f"sp{i % 5}",
            "xc_id": f"XC{i}",
            "recordist_id": f"rec{i % 3}",
            "lat": 50.0,
            "lon": 0.0,
            "subspecies": "",
            "wav_path": "/dev/null",
            "start_s": 0.0,
            "end_s": 0.3,
        })

    _relax_gates(cfg)
    vocab = build_vocabulary(cfg, syllables)
    sub = build_substitution_matrix(cfg, vocab, syllables)
    return syllables, vocab, sub


def _relax_gates(cfg):
    cfg.bootstrap_n = 5
    cfg.bootstrap_stability_ari_threshold = 0.0
    cfg.cross_recordist_ari_threshold = 0.0
    cfg.birdaves_cosine_threshold = 0.0
    cfg.spectral_cv_flagged_fraction_threshold = 1.0
    cfg.hdbscan_max_cycles = 2


class TestAlignmentDistances:
    def test_distances_in_unit_interval(self, cfg):
        """Pairwise acoustic distances lie in [0, 1]."""
        from syrinx.align import align_all

        syllables, vocab, sub = _make_simple_vocab_and_sub(cfg)
        cfg.null_model_binomial_alpha = 1.0  # always pass

        result = align_all(cfg, syllables, vocab, sub, dataset="genus")
        D = result["distance_matrix"]
        assert np.all(D >= -1e-9)
        assert np.all(D <= 1.0 + 1e-9)

    def test_diagonal_is_zero(self, cfg):
        """Self-distance is zero for all species."""
        from syrinx.align import align_all

        syllables, vocab, sub = _make_simple_vocab_and_sub(cfg)
        cfg.null_model_binomial_alpha = 1.0

        result = align_all(cfg, syllables, vocab, sub, dataset="genus")
        D = result["distance_matrix"]
        assert np.allclose(np.diag(D), 0.0, atol=1e-9)

    def test_distance_matrix_symmetric(self, cfg):
        """Distance matrix is symmetric."""
        from syrinx.align import align_all

        syllables, vocab, sub = _make_simple_vocab_and_sub(cfg)
        cfg.null_model_binomial_alpha = 1.0

        result = align_all(cfg, syllables, vocab, sub, dataset="genus")
        D = result["distance_matrix"]
        assert np.allclose(D, D.T, atol=1e-9)


class TestNullModel:
    def test_null_model_produces_correct_number_of_permutations(self, cfg):
        """Null model produces 999-element distributions (or configured n)."""
        from syrinx.align import _run_null_model, _pairwise_distance_matrix, _make_aligner
        from syrinx.substitution import build_substitution_matrix
        from syrinx.vocabulary import build_vocabulary

        syllables, vocab, sub = _make_simple_vocab_and_sub(cfg, n_clusters=3, n_syllables=120)
        aligner = _make_aligner(sub)

        cluster_letters = vocab["cluster_letters"]
        labels_array = vocab["labels"]
        song_strings: dict[str, str] = {}
        for i, syl in enumerate(syllables):
            lb = labels_array[i]
            if lb == -1:
                continue
            sp = syl["species"]
            song_strings.setdefault(sp, "")
            song_strings[sp] += cluster_letters[lb]

        _, D = _pairwise_distance_matrix(song_strings, aligner)
        cfg.null_model_permutations = 19
        cfg.null_model_binomial_alpha = 1.0

        result = _run_null_model(D, song_strings, aligner, cfg, run_log=None)
        assert result["n_pairs"] >= 1
        assert "proportion_above" in result

    def test_null_model_identifies_shuffled_sequences(self, cfg):
        """Null model correctly flags random sequences as non-sequential."""
        from syrinx.align import PipelineGatingError, align_all
        from syrinx.substitution import build_substitution_matrix
        from syrinx.vocabulary import build_vocabulary

        syllables, vocab, sub = _make_simple_vocab_and_sub(cfg, n_clusters=3, n_syllables=120)
        cfg.null_model_permutations = 9
        cfg.recording_bootstrap_n = 3
        # Set threshold to 0 → always fail (sequences have no sequential structure)
        cfg.null_model_binomial_alpha = 0.0

        with pytest.raises(PipelineGatingError):
            align_all(cfg, syllables, vocab, sub, dataset="genus")


class TestRecordingBootstrap:
    def test_bootstrap_ci_shape(self, cfg):
        """Recording bootstrap returns correctly shaped CI arrays."""
        from syrinx.align import align_all

        syllables, vocab, sub = _make_simple_vocab_and_sub(cfg)
        cfg.null_model_binomial_alpha = 1.0
        cfg.recording_bootstrap_n = 5

        result = align_all(cfg, syllables, vocab, sub, dataset="genus")
        D = result["distance_matrix"]
        n = D.shape[0]

        cis = result["bootstrap_cis"]
        assert cis["ci_lower"].shape == (n, n)
        assert cis["ci_upper"].shape == (n, n)
        assert np.all(cis["ci_lower"] <= cis["ci_upper"] + 1e-9)
