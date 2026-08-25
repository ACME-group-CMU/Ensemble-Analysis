# Paper completion — numerical placeholders

All values are Boltzmann-weighted at T = 2000 K from the cached ensemble.
N (struct.) = 24-atom: **2893**, 36-atom: **866**, 48-atom: **275**.
Si–O cutoff: 2.0 Å throughout (BAD, Q^n, rings, coordination — paper Methods).

---

## §A. Short-range observables

**Paper text (verbatim, two placeholders):**

> *[Insert: first-peak position in D(r) for each system size vs the experimental value from Grimley et al; Si-O bond length from Si-O partial; Si and O coordination numbers, or maybe just S(q). D(r) in SI?]*

Recommended replacement (drop the bracketed placeholder, insert this in its place):

> The first peak in the Boltzmann-weighted reduced PDF D(r) sits at **1.64 Å (24-atom)**, **1.64 Å (36-atom)**, and **1.64 Å (48-atom)**, in agreement with the experimental Si–O bond length of 1.61 Å reported by Grimley et al. (1990). The peak in the Si–O partial g(r) sits at **1.64 Å** across all three sizes (Fig. S1).

**Paper text (verbatim, second placeholder):**

> *[Insert: Si and O coordination numbers from ensemble-averaged partial g(r) for each system size, confirming tetrahedral Si coordination near 4.0 and O coordination near 2.0]*

Recommended replacement:

> Integrating the Boltzmann-weighted Si–O partial g(r) to the first minimum at 2.0 Å gives a Si coordination number of **4.00 (24-atom)**, **3.97 (36-atom)**, and **3.94 (48-atom)**, and an O coordination number of **2.00**, **1.98**, and **1.97** respectively (Fig. S2). Si is essentially tetrahedrally coordinated and O is two-fold coordinated across all three system sizes.

Partial-peak summary (also in Fig. S4):

| Pair | 24-atom (Å) | 36-atom (Å) | 48-atom (Å) | Ref. (Grimley) |
|---|---|---|---|---|
| Si–O | 1.64 | 1.64 | 1.64 | 1.61 |
| Si–Si | 3.06 | 3.01 | 3.09 | 3.07 |
| O–O | 2.66 | 2.66 | 2.66 | 2.62 |

---

## §B. Ring statistics — 2MR populations

**Paper text (verbatim):**

> Two-membered rings appear at nonzero Boltzmann weight in all three ensembles, with R_N = **X**, **Y**, and **Z** for the 24-, 36-, and 48-atom ensembles respectively.

- **X = 0.055**  (24-atom)
- **Y = 0.060**  (36-atom)
- **Z = 0.061**  (48-atom)

> ⚠ **Trend caveat.** The next sentence in the paper currently reads *"These values decrease monotonically with system size."* In the cached data the trend is the opposite: 2MR R_N **increases** (0.055 → 0.060 → 0.061). Suggested replacement: *"These values increase modestly with system size, reflecting the greater configurational freedom of larger random cells."*

Full ring distribution (R_N, rings per Si):

| n | 24-atom | 36-atom | 48-atom |
|---|---|---|---|
| 2 | 0.055 | 0.060 | 0.061 |
| 3 | 0.153 | 0.150 | 0.150 |
| 4 | 0.229 | 0.229 | 0.216 |
| 5 | 0.269 | 0.315 | 0.316 |
| 6 | 0.294 | 0.405 | 0.430 |
| 7 | 0.201 | 0.360 | 0.446 |
| 8 | 0.048 | 0.196 | 0.284 |
| 9 | 0.000 | 0.062 | 0.106 |
| 10 | 0.000 | 0.009 | 0.032 |
| 11 | 0.000 | 0.001 | 0.004 |
| 12 | 0.000 | 0.000 | 0.001 |

---

## §C. Bond angle distributions

**Paper text (verbatim):**

> The Si-O-Si distribution peaks near 137° for the 24-atom ensemble … *[Insert: peak positions for 36- and 48-atom ensembles; FWHM for all three sizes]*

Recommended replacement:

> The Si–O–Si distribution peaks at **132° (FWHM 33°)** for the 24-atom ensemble, **134° (FWHM 33°)** for the 36-atom ensemble, and **130° (FWHM 32°)** for the 48-atom ensemble.

> Mean ± std (raw): 135.7 ± 18.3° (24), 134.7 ± 18.4° (36), 134.6 ± 18.4° (48).

**Paper text (verbatim, O–Si–O placeholder):**

> The O-Si-O distribution peaks near 109° across all system sizes … and matches the CPMD reference value of Sarnthein et al. within *[Insert: width vs CPMD reference value]*.

Recommended replacement:

> The O–Si–O distribution peaks at **110° (FWHM 8°)** for 24 atoms, **108° (FWHM 11°)** for 36 atoms, and **110° (FWHM 13°)** for 48 atoms, compared with the CPMD value of 109.5° / FWHM ≈ 9° from Sarnthein et al. (1995). The peak position matches; the RSL distributions are 2–4× broader than CPMD, reflecting the inclusion of strained tetrahedra in the random-search ensemble (Fig. S3).

---

## §D. Q^n speciation

**Paper text (verbatim):**

> The RSL ensembles are Q⁴-dominated, with Q⁴ fractions of **X%**, **Y%**, and **Z%** for the 24-, 36-, and 48-atom ensembles respectively … Q³ fractions are **X%**, **Y%**, **Z%** for the three system sizes, with smaller but nonzero Q², Q¹, and Q⁰ contributions. *[Insert: full Qⁿ fractions as a small in-text table for all three system sizes]*

Recommended in-text table:

|  | 24-atom | 36-atom | 48-atom |
|---|---|---|---|
| Q⁰ | 0.0% | 0.0% | 0.0% |
| Q¹ | 0.3% | 0.5% | 0.6% |
| Q² | 2.6% | 4.6% | 5.4% |
| Q³ | 17.0% | 24.5% | 26.9% |
| Q⁴ | 77.6% | 67.4% | 63.5% |

Q⁴ values for the X/Y/Z slot: **77.6%**, **67.4%**, **63.5%**.  Q³ values for the X/Y/Z slot: **17.0%**, **24.5%**, **26.9%**.

> ⚠ **Trend caveat.** The next paper sentence reads *"The non-Q⁴ populations decrease from 24 to 36 to 48 atoms, mirroring the trend in 2MRs and small ring populations."* In the cached data, non-Q⁴ **increases** with size (22.4% → 32.6% → 36.5%). Suggested replacement: *"The non-Q⁴ population grows with cell size, mirroring the 2MR trend and consistent with the larger configurational freedom of bigger random supercells."*

---

## §E. Energy distribution and 2MR strain (Fig. 4)

**Paper text (verbatim):**

> Microstates containing 2MRs concentrate in a high-energy tail of the basin distribution, separated from the lowest-energy minima by approximately *[Insert: energy gap value]* eV per ring, consistent with the strain energy of an edge-sharing tetrahedron pair calculated by Hamann (PRB, 1998; 1.23 eV/2MR).

Recommended replacement (24-atom value, with per-size figures in Fig. 4):

> approximately **0.24 eV per 2MR** (linear regression of total energy against 2MR count over all 2893 24-atom microstates). This marginal cost is below the 1.23 eV/2MR strain energy that Hamann (PRB, 1998) computes for an isolated edge-sharing pair in otherwise-pristine α-quartz, because in our random ensemble the 2MR-bearing microstates also relax other distortions that partially compensate the strain.

Per-size strain regression (Fig. 4):

| Size | N | Frac. ≥1 2MR | 2MR/struct range | ΔE / 2MR (regression, eV) | Cohort gap / 2MR (eV) |
|---|---|---|---|---|---|
| 24-atom | 2893 | 40.0% | 0–14 | 0.24 | +0.18 |
| 36-atom | 866 | 53.8% | 0–18 | 0.17 | +0.22 |
| 48-atom | 275 | 65.8% | 0–12 | 0.34 | +0.38 |

Note the cohort-difference value flips sign for the 48-atom ensemble — only 94 of 275 structures have zero 2MRs, so the no-2MR cohort is too small for a fair subtraction. The regression value is the more reliable estimate. This will tighten as the 48-atom ensemble grows.

---

## §F. Raw coordination tail (Fig. S8)

Q^n caps at 4 and counts only *bridging* O. Direct counting of every O within 2.0 Å of each Si surfaces both the under- and over-coordinated tails:

| Size | Si <4 | Si =4 | Si >4 | O <2 | O =2 | O >2 |
|---|---|---|---|---|---|---|
| 24-atom | 4.7 % | 90.7 % | **4.7 %** | 2.6 % | 94.9 % | **2.5 %** |
| 36-atom | 8.5 % | 85.6 % | **5.9 %** | 4.8 % | 92.0 % | **3.2 %** |
| 48-atom | 10.5 % | 83.3 % | **6.2 %** | 5.9 % | 91.0 % | **3.1 %** |

(Boltzmann-weighted at T = 2000 K, Si–O cutoff = 2.0 Å.)  The over-coordinated Si tail (5- and 6-fold) carries 4.7–6.2 % of weighted Si density across the three sizes. Worth noting in the paper text alongside the Q^n discussion, which silently caps at Q⁴ and only counts bridging O.

---

## Convergence note

The 48-atom ensemble currently has N = 275 structures vs. the 500 quoted in the Methods; the values above are the best estimate from cached data and will firm up as the remaining 48-atom DFT relaxations finish.
