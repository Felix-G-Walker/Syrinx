"""Shared helpers: manifest I/O, logging, distance normalisation."""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------

def load_manifest(path: Path) -> dict[str, Any]:
    """Load a JSON manifest, returning an empty dict if the file doesn't exist.

    Parameters
    ----------
    path:
        Path to the JSON manifest file.
    """
    if path.exists():
        with path.open() as fh:
            return json.load(fh)
    return {}


def save_manifest(data: dict[str, Any], path: Path) -> None:
    """Atomically write a JSON manifest.

    Parameters
    ----------
    data:
        Data to serialise.
    path:
        Destination path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as fh:
        json.dump(data, fh, indent=2, default=str)
    tmp.replace(path)
    logger.debug("Saved manifest to %s", path)


def append_manifest(record: dict[str, Any], path: Path) -> None:
    """Append a record to a list-valued JSON manifest.

    Parameters
    ----------
    record:
        Record to append.
    path:
        Destination path.
    """
    existing = load_manifest(path)
    if "records" not in existing:
        existing["records"] = []
    existing["records"].append(record)
    save_manifest(existing, path)


# ---------------------------------------------------------------------------
# Pickle serialisation with metadata sidecar
# ---------------------------------------------------------------------------

def save_array(
    array: Any,
    path: Path,
    *,
    config_hash: str,
    random_seed: int,
    n_species: int | None = None,
    n_cells: int | None = None,
    n_syllables_total: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Pickle an array and write a companion JSON metadata sidecar.

    Parameters
    ----------
    array:
        Object to pickle.
    path:
        Destination pickle path.
    config_hash:
        SHA prefix of the config, for provenance.
    random_seed:
        Seed used during this pipeline run.
    n_species:
        Number of species processed (optional).
    n_cells:
        Number of geographic cells (optional).
    n_syllables_total:
        Total syllables across all species/cells (optional).
    extra:
        Additional metadata key/value pairs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(array, fh)

    meta: dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "config_hash": config_hash,
        "random_seed": random_seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if n_species is not None:
        meta["n_species"] = n_species
    if n_cells is not None:
        meta["n_cells"] = n_cells
    if n_syllables_total is not None:
        meta["n_syllables_total"] = n_syllables_total
    if extra:
        meta.update(extra)

    meta_path = path.with_suffix(".json")
    save_manifest(meta, meta_path)
    logger.debug("Saved array to %s", path)


def load_array(path: Path) -> Any:
    """Load a pickled array.

    Parameters
    ----------
    path:
        Path to the pickle file.
    """
    with path.open("rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------------------
# Distance matrix helpers
# ---------------------------------------------------------------------------

def normalise_distance_matrix(D: np.ndarray) -> np.ndarray:
    """Normalise a square distance matrix to [0, 1] by its maximum off-diagonal.

    Parameters
    ----------
    D:
        Square symmetric distance matrix.

    Returns
    -------
    np.ndarray
        Normalised matrix with zero diagonal and off-diagonal values in [0, 1].
    """
    D = D.astype(float).copy()
    np.fill_diagonal(D, 0.0)
    max_val = np.max(D)
    if max_val > 0:
        D /= max_val
    return D


def upper_triangle(D: np.ndarray) -> np.ndarray:
    """Return the upper triangle (excluding diagonal) as a flat vector.

    Parameters
    ----------
    D:
        Square symmetric matrix.
    """
    idx = np.triu_indices(D.shape[0], k=1)
    return D[idx]


def great_circle_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Haversine great-circle distance in kilometres.

    Parameters
    ----------
    lat1, lon1:
        Coordinates of point 1 in decimal degrees.
    lat2, lon2:
        Coordinates of point 2 in decimal degrees.
    """
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def geographic_distance_matrix(lats: list[float], lons: list[float]) -> np.ndarray:
    """Build a pairwise great-circle distance matrix from coordinate lists.

    Parameters
    ----------
    lats:
        List of latitudes in decimal degrees.
    lons:
        List of longitudes in decimal degrees.
    """
    n = len(lats)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = great_circle_distance(lats[i], lons[i], lats[j], lons[j])
            D[i, j] = D[j, i] = d
    return D


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

def configure_logging(level: str = "INFO", log_path: Path | None = None) -> None:
    """Configure root logger with stream and optional file handler.

    Parameters
    ----------
    level:
        Logging level string (DEBUG, INFO, WARNING, ERROR).
    log_path:
        If provided, also write logs to this file.
    """
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(level=getattr(logging, level.upper()), format=fmt, handlers=handlers)


# ---------------------------------------------------------------------------
# Structured pipeline run log
# ---------------------------------------------------------------------------

class PipelineRunLog:
    """Structured JSON log accumulating threshold tests for a pipeline run."""

    def __init__(self, path: Path, config_hash: str, random_seed: int) -> None:
        self.path = path
        self._data: dict[str, Any] = {
            "pipeline_version": PIPELINE_VERSION,
            "config_hash": config_hash,
            "random_seed": random_seed,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "stages": {},
            "threshold_tests": [],
        }

    def record_threshold(
        self,
        name: str,
        value: float,
        threshold: float,
        passed: bool,
        stage: str = "",
    ) -> None:
        """Record a named threshold test result.

        Parameters
        ----------
        name:
            Descriptive name of the test.
        value:
            Observed numerical value.
        threshold:
            Threshold that was tested.
        passed:
            Whether the threshold was met.
        stage:
            Pipeline stage identifier.
        """
        self._data["threshold_tests"].append({
            "stage": stage,
            "name": name,
            "value": value,
            "threshold": threshold,
            "passed": passed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        status = "PASS" if passed else "FAIL"
        logger.info("[%s] %s: %.4f vs threshold %.4f — %s", stage, name, value, threshold, status)

    def record_stage(self, stage: str, summary: dict[str, Any]) -> None:
        """Record summary for a completed stage.

        Parameters
        ----------
        stage:
            Stage identifier string.
        summary:
            Dict of stage-level summary statistics.
        """
        self._data["stages"][stage] = {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            **summary,
        }

    def finalise(self) -> None:
        """Write the completed log to disk."""
        self._data["finished_at"] = datetime.now(timezone.utc).isoformat()
        save_manifest(self._data, self.path)
        logger.info("Pipeline run log written to %s", self.path)


def config_hash_from_path(config_path: Path) -> str:
    """Return a short hash of a config YAML file for provenance.

    Parameters
    ----------
    config_path:
        Path to the YAML file.
    """
    content = config_path.read_bytes()
    return hashlib.sha256(content).hexdigest()[:16]
