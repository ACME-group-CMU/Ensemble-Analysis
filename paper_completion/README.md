# paper_completion/

Everything needed to fill in the placeholders, build Fig. 4 + the SI figures,
and drop in the Discussion section for *Random Structure Sampling for Vitreous
Silica: Domain of Validity for Medium-Range Structural Observables*.

## Layout

| File | Purpose |
|------|---------|
| `_load.py` | Shared data loader. Builds Boltzmann-weighted ensembles per cell size from cached observables in `data/`. Every figure script imports from here. |
| `compute_summary.py` | Computes every numerical X/Y/Z value referenced in the paper. Writes `summary_data.json` (machine-readable) and `summary.md` (human-readable, with paper text quoted verbatim and each value substituted). |
| `summary.md` | **Drop-in replacements** for each placeholder, including caveats where the cached data conflicts with the paper text. |
| `discussion.md` | **Drop-in replacement** for the bulleted Discussion outline (5 paragraphs). |
| `figures/fig4_*.py` | Main-text Fig. 4 (energy vs 2MR / Q⁴). |
| `figures/figS1_*.py … figures/figS8_*.py` | SI figures (each self-contained). S8 = raw 0..7 coordination distribution at 2.0 Å, surfacing the over-coordinated Si tail that Q^n hides. |
| `figures/*.pdf`, `*.png` | Rendered figures (PDF for paper, PNG for previewing). |

## Regenerating everything

From the repo root, with the conda env that has numpy / pymatgen / matplotlib:

```bash
PY=/Users/raphaelzstone/miniconda3/envs/ensemble/bin/python
$PY paper_completion/compute_summary.py
for f in paper_completion/figures/fig*.py; do $PY "$f"; done
```

Each figure script writes its own PDF + PNG next to itself. Re-running any
single script updates only that figure.

## Conventions

- **Sizes & colors** — 24-atom (blue), 36-atom (teal/green), 48-atom
  (orange/red). Defined in `_load.py::SIZE_COLORS` so the choice
  propagates to every figure.
- **Boltzmann weighting** — T = 2000 K throughout (`_load.py::T_BOLTZ`). Per-atom
  energies in Ry (the project's energy units), `kB_Ry = 6.33362e-6`.
- **Si–O bond cutoff** — 2.0 Å throughout (rings, BAD, Q^n, coordination),
  matching paper Methods and the first minimum of g_SiO(r). Defaults are
  set in `src/data_management_v2.populate_{bond_angle,qn}_distributions`
  and `src/rings.get_bond_cutoffs`.

## Open items the user should review

1. **2MR and non-Q⁴ trend reverses with size.** Cached data shows these
   *increasing* with cell size, opposite to the current paper claim.
   Verify the direction holds after repopulating at 2.0 Å — the larger
   cutoff catches more borderline bonds and may shift Q^n appreciably
   (see §F: 1.8 Å gave Q⁴ = 83.7 % for 24-atom vs 90.6 % at 2.0 Å, a
   7-point swing). Suggested wording fix is in `summary.md`.
2. **Si–O–Si peak is 131° in the cached data**, not the 137° the paper
   currently quotes. Both are well within the literature range of PBE
   results (Sarnthein 1995: 136–142°). Update the paper number to match
   what the figure actually shows, and revise the "approximately 14° below
   experiment" sentence to "approximately 20° below experiment".
3. **48-atom statistics** are based on N = 181 cached structures (vs the
   500 quoted in Methods). All 48-atom numbers will firm up as the
   remaining DFT relaxations finish.
4. **Energy gap per 2MR** comes from a linear regression of total energy
   against 2MR count rather than a binary cohort difference, because the
   no-2MR cohort becomes too small at 48 atoms to subtract cleanly. The
   regression slope (0.23 eV/2MR for 24-atom) is below Hamann's 1.23 eV/2MR
   for an isolated edge-sharing pair because the 2MR-bearing microstates
   relax other distortions that partially compensate the strain. Both
   numbers are reported in `summary.md`.
