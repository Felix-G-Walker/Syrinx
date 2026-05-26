"""Tests for Stage 9 — inferential analyses."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from syrinx.config import load_config
from syrinx.inference import PipelineGatingError


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    config_path = Path(__file__).parent.parent / "config.yaml"
    cfg = load_config(config_path)
    tmpdir = tmp_path_factory.mktemp("inference_test")
    cfg.data_dir = str(tmpdir)
    cfg.output_dir = str(tmpdir / "out")
    cfg.mrm_permutations = 99
    cfg.mantel_permutations = 99
    return cfg


def _make_passed_vocabulary():
    """Build a vocabulary dict with all gates passed."""
    n = 6
    labels = np.array([i % n for i in range(100)])
    centroids = np.eye(n, 36)
    letters = {i: chr(ord("A") + i) for i in range(n)}
    gate_results = {
        f"gate{i}": {"passed": True, "value": 0.9, "threshold": 0.5}
        for i in range(1, 5)
    }
    gate_results["gate1_bootstrap_stability"] = {"passed": True, "value": 0.9, "threshold": 0.5}
    gate_results["gate2_cross_recordist"] = {"passed": True, "value": 0.9, "threshold": 0.5}
    gate_results["gate3_birdaves"] = {"passed": True, "value": 0.9, "threshold": 0.6}
    gate_results["gate4_spectral_homogeneity"] = {"passed": True, "value": 0.05, "threshold": 0.1}
    return {
        "labels": labels,
        "cluster_letters": letters,
        "centroids": centroids,
        "n_clusters": n,
        "noise_fraction": 0.0,
        "gate_results": gate_results,
        "params": {"min_cluster_size": 5, "min_samples": 3},
    }


def _make_failed_vocabulary():
    """Build a vocabulary dict with gate 1 failed."""
    vocab = _make_passed_vocabulary()
    vocab["gate_results"]["gate1_bootstrap_stability"]["passed"] = False
    return vocab


def _make_alignment_result_passed(n_species=10, seed=42):
    """Build an alignment result dict with null model passed."""
    rng = np.random.RandomState(seed)
    names = [f"Phylloscopus_sp{i}" for i in range(n_species)]
    D = rng.rand(n_species, n_species)
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)
    return {
        "distance_matrix": D,
        "species_names": names,
        "alignment_scores": D.copy(),
        "null_model_result": {"passed": True, "binomial_p": 0.001, "n_pairs": 45},
        "bootstrap_cis": {
            "ci_lower": np.zeros((n_species, n_species)),
            "ci_upper": np.ones((n_species, n_species)),
        },
        "noise_floors": {},
        "nominate_only_distance_matrix": None,
        "nominate_entities": [],
    }


def _make_alignment_result_failed_null(n_species=5):
    result = _make_alignment_result_passed(n_species)
    result["null_model_result"]["passed"] = False
    return result


class TestGating:
    def test_raises_when_vocabulary_gates_failed(self, cfg, tmp_path):
        """PipelineGatingError raised if vocabulary gates did not pass."""
        from syrinx.inference import run_inference

        vocab = _make_failed_vocabulary()
        alignment = _make_alignment_result_passed()

        with pytest.raises(PipelineGatingError) as exc_info:
            run_inference(
                cfg,
                alignment_result=alignment,
                vocabulary=vocab,
                species_metadata=[],
                descriptive_result={"region_diversity": {}, "species_stats": {}},
                run_log=None,
                dataset="genus",
            )
        assert "gate" in str(exc_info.value).lower()

    def test_raises_when_null_model_failed(self, cfg):
        """PipelineGatingError raised if null model gate did not pass."""
        from syrinx.inference import run_inference

        vocab = _make_passed_vocabulary()
        alignment = _make_alignment_result_failed_null()

        with pytest.raises(PipelineGatingError) as exc_info:
            run_inference(
                cfg,
                alignment_result=alignment,
                vocabulary=vocab,
                species_metadata=[],
                descriptive_result={"region_diversity": {}, "species_stats": {}},
                run_log=None,
                dataset="genus",
            )
        assert "null model" in str(exc_info.value).lower()


class TestMRMViaPython:
    """MRM Python approximation (used in power simulation) is tested here."""

    def test_mrm_returns_valid_p_value(self):
        from syrinx.power import _run_mrm_python

        rng = np.random.RandomState(42)
        x = rng.randn(45)
        y = x * 0.5 + rng.randn(45) * 0.3

        p = _run_mrm_python(x, y, n_perm=99)
        assert 0.0 <= p <= 1.0

    def test_mrm_detects_strong_correlation(self):
        from syrinx.power import _run_mrm_python

        rng = np.random.RandomState(1)
        x = rng.randn(100)
        y = x + rng.randn(100) * 0.1

        p = _run_mrm_python(x, y, n_perm=199)
        assert p < 0.05, f"Expected significant result, got p={p}"

    def test_mrm_fails_to_reject_null_on_independent(self):
        from syrinx.power import _run_mrm_python

        rng = np.random.RandomState(2)
        x = rng.randn(50)
        y = rng.randn(50)

        # Run 10 times; most should not reject
        p_vals = [_run_mrm_python(x, rng.randn(50), n_perm=99) for _ in range(10)]
        rejection_rate = sum(p < 0.05 for p in p_vals) / 10
        assert rejection_rate <= 0.4, f"Too many false positives: {rejection_rate}"


class TestMantelPython:
    def test_mantel_p_value_in_range(self):
        from syrinx.inference import _mantel_python

        rng = np.random.RandomState(5)
        x = rng.rand(45)
        y = x + rng.randn(45) * 0.5

        result = _mantel_python(x, y, n_perm=99, method="pearson")
        assert "r" in result
        assert "p_value" in result
        assert 0.0 <= result["p_value"] <= 1.0
        assert -1.0 <= result["r"] <= 1.0


class TestH2Spearman:
    def test_h2_with_bbs_data(self, cfg, tmp_path):
        """H2 Spearman runs with BBS data and returns rho and p-value."""
        import json
        from syrinx.inference import _run_h2

        # Write BBS data to temp dir
        bbs = {
            "Scotland": -18.0, "Wales": -31.0, "Northern Ireland": -22.0,
            "England-SE": -55.0, "England-SW": -48.0,
            "England-Midlands": -47.0, "England-N": -35.0,
        }
        bbs_path = Path(cfg.data_dir) / "bbs_regional_trends.json"
        bbs_path.parent.mkdir(parents=True, exist_ok=True)
        with bbs_path.open("w") as fh:
            json.dump(bbs, fh)

        rng = np.random.RandomState(99)
        region_diversity = {
            reg: {
                "composite_diversity": rng.randn(),
                "vocab_size": rng.randint(3, 10),
                "mean_complexity": rng.uniform(0, 2),
                "mean_pairwise_distance": rng.uniform(0.1, 0.9),
            }
            for reg in cfg.bbs_regions
        }

        result = _run_h2(cfg, {"region_diversity": region_diversity}, run_log=None)
        assert "rho" in result
        assert "p_value" in result
        assert -1.0 <= result["rho"] <= 1.0


class TestInterpretResult:
    def test_significant_and_meaningful(self):
        from syrinx.inference import _interpret_result
        s = _interpret_result(True, True, 0.10, 0.25, "semi-partial r²")
        assert "meaningful" in s.lower()

    def test_significant_but_small(self):
        from syrinx.inference import _interpret_result
        s = _interpret_result(True, False, 0.10, 0.05, "semi-partial r²")
        assert "below minimum" in s.lower()

    def test_not_significant(self):
        from syrinx.inference import _interpret_result
        s = _interpret_result(False, False, 0.10, 0.03, "semi-partial r²")
        assert "not statistically significant" in s.lower()
