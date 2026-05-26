"""Stage 2 — BirdNET segmentation + energy fallback + MAO validation."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .utils import load_manifest, save_manifest

logger = logging.getLogger(__name__)

_BIRDNETLIB_AVAILABLE = False
try:
    from birdnetlib import Recording
    from birdnetlib.analyzer import Analyzer
    _BIRDNETLIB_AVAILABLE = True
except ImportError:
    logger.warning("birdnetlib not installed; energy fallback will be used for all recordings")


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def segment_all(
    cfg: Config,
    recordings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Segment all recordings into syllable clips.

    Parameters
    ----------
    cfg:
        Pipeline configuration.
    recordings:
        Download manifest records (each must have ``wav_path``).

    Returns
    -------
    list[dict]
        Syllable records with ``wav_path``, ``start_s``, ``end_s``,
        ``species``, ``xc_id``, ``segmenter``.
    """
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    manifest_dir = cfg.data_path / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    seg_log_path = manifest_dir / "segmentation_log.json"
    seg_log = load_manifest(seg_log_path)
    seg_log.setdefault("records", [])
    fallback_count = 0

    analyzer = _load_analyzer()

    all_syllables: list[dict[str, Any]] = []
    for rec in recordings:
        wav_path = Path(rec.get("wav_path", ""))
        if not wav_path.exists():
            logger.warning("wav not found: %s; skipping", wav_path)
            continue

        syllables, used_fallback = _segment_recording(
            wav_path, rec, cfg, analyzer
        )
        all_syllables.extend(syllables)

        if used_fallback:
            fallback_count += 1
        seg_log["records"].append({
            "xc_id": rec.get("xc_id"),
            "wav_path": str(wav_path),
            "n_syllables": len(syllables),
            "segmenter": "fallback" if used_fallback else "birdnet",
        })

    save_manifest(seg_log, seg_log_path)
    logger.info(
        "Segmented %d recordings → %d syllables (%d used energy fallback)",
        len(recordings), len(all_syllables), fallback_count,
    )
    return all_syllables


def segment_recording(
    wav_path: Path,
    cfg: Config,
    analyzer: Any = None,
) -> list[dict[str, Any]]:
    """Segment a single recording, returning syllable interval records.

    Parameters
    ----------
    wav_path:
        Path to a 16 kHz mono wav file.
    cfg:
        Pipeline configuration.
    analyzer:
        Optional pre-loaded BirdNET analyzer.

    Returns
    -------
    list[dict]
        Syllable records with ``start_s`` and ``end_s``.
    """
    syllables, _ = _segment_recording(wav_path, {}, cfg, analyzer)
    return syllables


# ---------------------------------------------------------------------------
# MAO validation (run once, results cached)
# ---------------------------------------------------------------------------

def run_mao_validation(cfg: Config) -> dict[str, Any]:
    """Run mean absolute offset validation against reference corpora.

    Downloads Powdermill and Bengalese finch corpora on first call,
    caches results in ``data/manifests/segmentation_validation.json``.

    Parameters
    ----------
    cfg:
        Pipeline configuration.

    Returns
    -------
    dict
        Validation results with keys ``powdermill_mao_ms``,
        ``bengalese_mao_ms``, ``birdnet_primary``.
    """
    manifest_dir = cfg.data_path / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    val_path = manifest_dir / "segmentation_validation.json"

    if val_path.exists():
        cached = load_manifest(val_path)
        if cached.get("completed"):
            logger.info("Using cached MAO validation results")
            return cached

    analyzer = _load_analyzer()
    results: dict[str, Any] = {"completed": False}

    # Primary: Powdermill
    pm_corpus = _ensure_powdermill_corpus(cfg)
    if pm_corpus:
        pm_mao = _compute_mao_corpus(pm_corpus, cfg, analyzer)
        results["powdermill_mao_ms"] = pm_mao
        threshold = cfg.segmentation_mao_threshold_ms
        results["powdermill_threshold_ms"] = threshold
        results["powdermill_passed"] = pm_mao < threshold
        logger.info("Powdermill MAO: %.2f ms (threshold %.1f ms, pass=%s)",
                    pm_mao, threshold, results["powdermill_passed"])
    else:
        logger.warning("Powdermill corpus unavailable; skipping primary MAO check")
        results["powdermill_mao_ms"] = None
        results["powdermill_passed"] = None

    # Secondary: Bengalese finch (non-gating)
    bf_corpus = _ensure_bengalese_corpus(cfg)
    if bf_corpus:
        bf_mao = _compute_mao_corpus(bf_corpus, cfg, analyzer)
        results["bengalese_mao_ms"] = bf_mao
        logger.info("Bengalese finch MAO: %.2f ms (informational only)", bf_mao)
    else:
        results["bengalese_mao_ms"] = None

    # Determine primary segmenter
    pm_passed = results.get("powdermill_passed")
    results["birdnet_primary"] = _BIRDNETLIB_AVAILABLE and (pm_passed is None or pm_passed)
    results["completed"] = True

    save_manifest(results, val_path)
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_analyzer() -> Any:
    if not _BIRDNETLIB_AVAILABLE:
        return None
    try:
        return Analyzer()
    except Exception as exc:
        logger.warning("Could not initialise BirdNET analyzer: %s", exc)
        return None


def _segment_recording(
    wav_path: Path,
    rec_meta: dict[str, Any],
    cfg: Config,
    analyzer: Any,
) -> tuple[list[dict[str, Any]], bool]:
    """Segment a single recording; returns (syllables, used_fallback).

    Parameters
    ----------
    wav_path:
        Path to the wav file.
    rec_meta:
        Recording metadata (may be empty dict for standalone use).
    cfg:
        Pipeline configuration.
    analyzer:
        BirdNET Analyzer instance or None.
    """
    used_fallback = False
    detections = []

    if analyzer is not None and _BIRDNETLIB_AVAILABLE:
        detections = _run_birdnet(wav_path, analyzer)

    if len(detections) < 3:
        detections = _run_energy_fallback(wav_path, cfg)
        used_fallback = True

    pad_s = cfg.boundary_pad_ms / 1000.0
    min_s = cfg.syllable_min_ms / 1000.0
    max_s = cfg.syllable_max_ms / 1000.0

    syllables = []
    for start, end in detections:
        start = max(0.0, start - pad_s)
        end = end + pad_s
        dur = end - start
        if dur < min_s or dur > max_s:
            continue
        syllables.append({
            "wav_path": str(wav_path),
            "start_s": round(start, 4),
            "end_s": round(end, 4),
            "duration_s": round(dur, 4),
            "species": rec_meta.get("species", ""),
            "xc_id": rec_meta.get("xc_id", ""),
            "recordist_id": rec_meta.get("recordist_id", ""),
            "lat": rec_meta.get("lat"),
            "lon": rec_meta.get("lon"),
            "subspecies": rec_meta.get("subspecies", ""),
            "segmenter": "fallback" if used_fallback else "birdnet",
        })

    return syllables, used_fallback


def _run_birdnet(wav_path: Path, analyzer: Any) -> list[tuple[float, float]]:
    """Run BirdNET detection and return onset/offset pairs.

    Parameters
    ----------
    wav_path:
        Path to the wav file.
    analyzer:
        Loaded BirdNET Analyzer instance.
    """
    try:
        recording = Recording(
            analyzer,
            str(wav_path),
            lat=0.0,
            lon=0.0,
            min_conf=0.1,
        )
        recording.analyze()
        detections = []
        for det in recording.detections:
            start = float(det.get("start_time", det.get("start", 0)))
            end = float(det.get("end_time", det.get("end", start + 3.0)))
            detections.append((start, end))
        return detections
    except Exception as exc:
        logger.debug("BirdNET failed on %s: %s", wav_path, exc)
        return []


def _run_energy_fallback(wav_path: Path, cfg: Config) -> list[tuple[float, float]]:
    """Energy onset detection fallback using librosa.

    Parameters
    ----------
    wav_path:
        Path to the wav file.
    cfg:
        Pipeline configuration.
    """
    import librosa

    try:
        y, sr = librosa.load(str(wav_path), sr=None, mono=True)
        onset_frames = librosa.onset.onset_detect(
            y=y, sr=sr, delta=0.07, units="frames"
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        pairs = []
        for i, t in enumerate(onset_times):
            if i + 1 < len(onset_times):
                end = onset_times[i + 1]
            else:
                end = t + 0.5  # default 500 ms for last segment
            pairs.append((float(t), float(end)))
        return pairs
    except Exception as exc:
        logger.warning("Energy fallback failed on %s: %s", wav_path, exc)
        return []


# ---------------------------------------------------------------------------
# Reference corpus helpers
# ---------------------------------------------------------------------------

def _ensure_powdermill_corpus(cfg: Config) -> Path | None:
    """Download and cache the Powdermill reference corpus.

    Parameters
    ----------
    cfg:
        Pipeline configuration.
    """
    import urllib.request
    import zipfile

    dest = cfg.data_path / "reference" / "powdermill"
    dest.mkdir(parents=True, exist_ok=True)
    annotation_file = dest / "annotations.csv"

    if annotation_file.exists():
        return dest

    zenodo_url = "https://zenodo.org/record/4656848/files/powdermill.zip"
    zip_path = dest / "powdermill.zip"
    logger.info("Downloading Powdermill corpus from Zenodo…")
    try:
        urllib.request.urlretrieve(zenodo_url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)
        zip_path.unlink(missing_ok=True)
        return dest
    except Exception as exc:
        logger.warning("Could not download Powdermill corpus: %s", exc)
        return None


def _ensure_bengalese_corpus(cfg: Config) -> Path | None:
    """Download and cache the Bengalese finch reference corpus.

    Parameters
    ----------
    cfg:
        Pipeline configuration.
    """
    import urllib.request
    import zipfile

    dest = cfg.data_path / "reference" / "bengalese"
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.glob("*.wav")):
        return dest

    figshare_url = "https://ndownloader.figshare.com/articles/3470165/versions/1"
    zip_path = dest / "bengalese.zip"
    logger.info("Downloading Bengalese finch corpus from Figshare…")
    try:
        urllib.request.urlretrieve(figshare_url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)
        zip_path.unlink(missing_ok=True)
        return dest
    except Exception as exc:
        logger.warning("Could not download Bengalese finch corpus: %s", exc)
        return None


def _compute_mao_corpus(corpus_dir: Path, cfg: Config, analyzer: Any) -> float:
    """Compute mean absolute offset (ms) between detected and annotated onsets.

    Parameters
    ----------
    corpus_dir:
        Directory containing wav files and annotation CSV.
    cfg:
        Pipeline configuration.
    analyzer:
        BirdNET analyzer (or None to use energy fallback).
    """
    import csv

    annotation_file = corpus_dir / "annotations.csv"
    if not annotation_file.exists():
        # Try to find any csv
        csvs = list(corpus_dir.rglob("*.csv"))
        if not csvs:
            logger.warning("No annotation CSV in %s", corpus_dir)
            return float("nan")
        annotation_file = csvs[0]

    offsets: list[float] = []
    with annotation_file.open() as fh:
        reader = csv.DictReader(fh)
        by_file: dict[str, list[float]] = {}
        for row in reader:
            fname = row.get("filename") or row.get("file") or ""
            try:
                onset = float(row.get("onset_s") or row.get("begin_time") or row.get("onset", 0))
            except ValueError:
                continue
            by_file.setdefault(fname, []).append(onset)

    for fname, expert_onsets in by_file.items():
        wav = corpus_dir / fname
        if not wav.exists():
            wav_candidates = list(corpus_dir.rglob(fname))
            if not wav_candidates:
                continue
            wav = wav_candidates[0]

        if analyzer is not None and _BIRDNETLIB_AVAILABLE:
            detected = _run_birdnet(wav, analyzer)
        else:
            detected = _run_energy_fallback(wav, cfg)

        det_onsets = [s for s, _ in detected]

        for exp_onset in expert_onsets:
            if not det_onsets:
                offsets.append(500.0)  # large penalty for missed detection
                continue
            closest = min(abs(exp_onset - d) for d in det_onsets)
            offsets.append(closest * 1000.0)

    if not offsets:
        return float("nan")
    return float(np.mean(offsets))
