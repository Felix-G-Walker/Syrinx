"""species_profile.json builder from reference sources."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_species_profiles(
    cfg: Config,
    species_stats: dict[str, dict[str, Any]],
    alignment_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build species_profile.json with descriptive and alignment statistics.

    Parameters
    ----------
    cfg:
        Pipeline configuration.
    species_stats:
        Per-species statistics from :func:`~syrinx.descriptive.compute_descriptives`.
    alignment_result:
        Optional alignment result from Stage 6.

    Returns
    -------
    dict
        Full species profile dict. Also written to ``outputs/species_profile.json``.
    """
    species_names = sorted(species_stats.keys())
    profiles: dict[str, Any] = {}

    for sp in species_names:
        stats = species_stats.get(sp, {})
        profile: dict[str, Any] = {
            "species": sp,
            "n_syllables": stats.get("n_syllables"),
            "entropy": {
                "H1": stats.get("H1"),
                "H2": stats.get("H2"),
                "complexity_C": stats.get("C"),
            },
            "acoustic": {
                "mean_peak_freq_hz": stats.get("mean_peak_freq"),
                "mean_min_freq_hz": stats.get("mean_min_freq"),
                "mean_freq_range_hz": stats.get("mean_freq_range"),
                "mean_peak_amplitude": stats.get("mean_peak_amplitude"),
                "mean_attack_ms": stats.get("mean_attack_ms"),
                "mean_decay_ms": stats.get("mean_decay_ms"),
                "mean_fm_depth": stats.get("mean_fm_depth"),
            },
        }

        if alignment_result:
            names = alignment_result.get("species_names", [])
            if sp in names:
                idx = names.index(sp)
                D = alignment_result.get("distance_matrix")
                if D is not None:
                    row = D[idx, :]
                    profile["mean_acoustic_distance"] = float(row.mean())
                    profile["max_acoustic_distance"] = float(row.max())

        profiles[sp] = profile

    out_path = Path(cfg.output_dir) / "species_profile.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(profiles, fh, indent=2, default=str)
    logger.info("Species profiles written to %s", out_path)
    return profiles
