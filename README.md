# Syrinx

**An acoustic phylogenetics pipeline.** Syrinx treats birdsong as sequence data — segmenting recordings into syllables, clustering them into a discrete vocabulary, and using alignment-based distances to reconstruct evolutionary relationships between species. The pipeline tests whether acoustic similarity recovers known molecular phylogenies, and provides tools for visualising vocabularies, distance matrices, and tanglegrams.

The name comes from the *syrinx*, the vocal organ in birds that produces all birdsong.

> **Status:** Research prototype accompanying a preprint (in preparation). Results are exploratory and should be interpreted alongside field data and expert judgement.

---

## What it does

1. **Ingests** birdsong recordings (Xeno-canto, Macaulay Library, or local files).
2. **Segments** recordings into syllables using BirdNET or an energy-based fallback.
3. **Clusters** syllables into a discrete vocabulary (letters A, B, C…) via UMAP + HDBSCAN on learned acoustic embeddings.
4. **Aligns** per-species syllable strings and computes pairwise acoustic distances.
5. **Compares** the resulting acoustic tree against a molecular reference tree (Mantel tests, Robinson-Foulds distance, tanglegrams).
6. **Visualises** vocabularies, UMAP embeddings, complexity distributions, and trees.

---

## Repository layout

```
Syrinx/
├── cli/          # Python pipeline (run_pipeline.py and modules)
├── web/          # Web app — results viewer and public-facing pages
├── docs/         # Paper, methods notes, figure sources
├── tests/        # Unit and integration tests
├── config.yaml   # Pipeline configuration
└── requirements.txt
```

The `cli/` and `web/` trees are independent — the pipeline runs standalone, and the web app reads its outputs.

---

## Quick start (CLI)

```bash
git clone https://github.com/Felix-G-Walker/Syrinx.git
cd Syrinx
pip install -r requirements.txt
python cli/run_pipeline.py --config config.yaml
```

Outputs land in `results/` — distance matrices, trees (Newick), figures (PNG + interactive HTML), and a JSON summary of the run.

### Requirements

- Python 3.10+
- `ffmpeg` (system install)
- `kaleido` for static figure export
- See `requirements.txt` for Python packages

---

## Web app

The web app provides three interfaces:

- **Researcher** — upload recordings, run the pipeline, view results
- **Field** — population-level vocal-diversity diagnostics with traffic-light status
- **Public** — science-communication piece presenting pre-computed results

Run locally:

```bash
cd web
# follow web/README.md for stack-specific instructions
```

---

## Reproducing the paper

The full set of results in the accompanying preprint can be regenerated from a clean checkout:

```bash
python cli/run_pipeline.py --config config.paper.yaml
```

This will pull the species list, fetch recordings, run segmentation and clustering, compute distance matrices, run the Mantel and Robinson-Foulds tests, and render all figures. Expect a multi-hour run depending on hardware and network.

---

## Citing

If you use Syrinx in published work, please cite the preprint (citation to follow on bioRxiv deposit) and link to this repository.

---

## Author

**Felix Walker** — independent researcher, Edinburgh, Scotland.

---

## License

TBD — license file to be added before public release.
