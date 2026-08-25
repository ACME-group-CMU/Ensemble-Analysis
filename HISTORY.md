# Project history — superseded methods

Everything below was deleted from the repo on 2026-08-25 as part of a cleanup. This
is enough to reconstruct any of it if needed; none of it feeds the current pipeline
(`src/` → `data/observables/` → `paper_completion/figures/`).

## Structure similarity / clustering

Used to cluster or subsample the ~3000-structure RSL pool (e.g. before deciding
which structures to relax with DFT, or for diversity sampling). Three fingerprint
backends, each an external package:

- **LIBFP** local fingerprints
- **MBTR** (many-body tensor representation)
- **LEVR** fingerprints

Implementation was `similarity_calculator.py` (per-pair comparison,
`LIBFPSimilarityCalculator` / `LEVRSimilarityCalculator` classes) +
`similarity_analyzer.py` (pairwise similarity matrices, clustering via
`networkx`/DBSCAN). Cached fingerprints and similarity matrices lived in
`data/Stored_info/` (`1k.npy`, `2k.npy`, `3k.npy`, `libfp_3k.npy`, `mbtr_3k.npy`,
`levr_3k.npy`, `clustering_results.pkl`) and `data/unrefined_data/` (similarity
heatmaps). To recreate: re-run the fingerprint calculators over
`data/all_sizes/*.vasp`.

## Louvain community-sampling

`src/ensembles_v2.py` had a family of sampling strategies (`louvain_central_sample`,
`louvain_diverse_sample`, `louvain_cell_tower_sample`, `louvain_max_representation_sample`,
`louvain_random_sample`) built on top of the similarity matrices above, for picking a
representative subset of structures via Louvain community detection on the
similarity graph.

## QUESTS information entropy

`src/entropy.py` computed configurational/information entropy over the ensemble
using **QUESTS** (Schwalbe-Koda et al. 2024, "Model-free quantification of
completeness..."):

```
pip install git+https://github.com/dskoda/quests.git
```

API used: `quests.entropy`, `quests.descriptor`, `quests.matrix`,
`quests.tools.plotting`. Produced `Overview/figures/Energy_vs_Entropy.pdf`. To
recreate: install QUESTS, run its descriptor/entropy pipeline over
`data/all_sizes/*.vasp` (or the DFT energies in `data/observables/*.pkl`).

## Birch-Murnaghan EOS / bulk modulus

**Kept** (`data/bms/`) — not derived data, it's a raw DFT volume scan (7 volumes ×
~250 structures, energy + timing per point) that would cost real compute to redo.
`Birch_Murnaghan.py` / `Birch_Murnaghan2.py` fit the standard 3rd-order B-M EOS
(`scipy.optimize.curve_fit`) to `data/bms/bm_*.txt` to get bulk modulus, compared
against a VASP cross-check in `Overview/figures/QEvsVASP.pdf`.

## External RINGS-code cross-validation

An 18-atom test system (`data/sio2_pbe_18/`) was run through the external **RINGS**
code (Le Roux & Jund, *Comput. Mater. Sci.* 2010 — the same reference CLAUDE.md
cites for the ring-finding criterion) to validate the in-house ring finder
(`src/rings.py`) against an independent implementation. Output was
`data/sio2_pbe_18/RINGS_output/RINGS_SiO2_18_*.csv`. To recreate: install the RINGS
code, convert a subset of `data/all_sizes/*.vasp` to its input format, compare
against `src/rings.py` output for the same structures.

## Pre-DFT / intermediate structure pools

`data/3k_poscar/` (RSL candidate pool before DFT relaxation), `data/zfinal_structures_vasp/`
and `data/zfinal_structures_vasp_failed/` (relaxed/failed results under an older
naming scheme), `data/SiO2_36/`, `data/SiO2_48/`, `data/Atomic_scale_2/`,
`data/SiO2_1200_vasps/` (unrelaxed CONTCARs / per-structure QE working directories) —
all intermediate stages between RSL generation and the final relaxed set now in
`data/all_sizes/`. `data/SiO2_48_237/` was one structure's full QE working directory
(wavefunctions, pseudopotentials, `.scf.in`/`.scf.out`) — debug leftovers, not data.

## Unrelated project: MACE/LAMMPS interface generation

`data/config.sh`, `data/create_lammps_input_v2.py`, `data/generate_interface.py`,
`data/install_mace.sh`, `data/run_interface_md.sh`, `data/final_trajectory.lammpstrj`,
`data/Step_2850000/` — a melt-quench MD / interface-generation pipeline for a
**different** material system (Li-Co-O interfaces via OgreInterface, MACE/CHGNet
potentials), not v-SiO₂. Unrelated to this paper.

`data/config.sh` contained a plaintext Materials Project API key. It was never
committed to git, but was flagged and deleted — rotate the key if it was ever shared
elsewhere.

## Old per-observable caches

`data/rdfs/`, `data/rdfs_from_sq/`, `data/sq/`, `data/counting_functions/`,
`data/bond_angle_distributions/`, `data/structures/`, `data/ring_statistics/`,
`data/qn_distributions/`, `data/energies/`, `data/densities/`, `data/smooth_rdfs/` —
the original `data_management_v2.py` cache layout, one folder per observable, written
by hand-rolled `populate_*` functions with known bugs (see git history / prior
session notes for details — Qⁿ bridging-O definition, ring self-image contamination,
minimum-image convention in S(q)). Fully superseded by the single
`data/observables/*.pkl` cache written by `src/observables.py`, which fixes those
bugs. Regenerate via `python regenerate.py`.

## Old code layout (`outdated/`)

The whole pre-refactor `src/*_v2.py` module set, plus every notebook that drove
figures before they became scripts under `paper_completion/figures/`
(`v2.ipynb`, `36_data_analysis_MD.ipynb`, `glass_characterization_plots.ipynb`,
`18_ring_distribution.ipynb`, `figure_1/2/3.ipynb`, and the `notebooks/` exploratory
set). Superseded in full by `src/{io,bonding,rdf,rings,angles,qn,ensemble,observables,
analysis}.py` — see `CLAUDE.md` for the current layout.
