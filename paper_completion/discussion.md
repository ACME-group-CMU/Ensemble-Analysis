# Discussion

> Drop-in replacement for the bulleted Discussion outline in the current
> manuscript. Five paragraphs, one per outline point, in the same terse,
> declarative voice as the existing Introduction and Results.

---

The results presented here partition the structural observables of vitreous
SiO₂ into two regimes with respect to RSL sampling. Short-range correlations
— the Si–O bond length, the corner-sharing O–O distance, the FSDP position,
and the gross shape of S(q) — converge to within experimental uncertainty
across all three system sizes (Fig. 1, Fig. S1, Fig. S2). This is the regime
that the nearsightedness-of-electronic-matter argument (Prodan & Kohn, 2005)
predicts to be insensitive to long-range order, and it is precisely what
Wolf et al. (JAP, 2025) demonstrated for the same set of observables under
a different code and pseudopotential choice. The present QE-PBE-USPP setup
reproduces their result: ensemble averages over small periodic cells recover
the local pair correlations of bulk glass without input from the full
medium-range network.

Some of the medium-range observables we examine show systematic departures
from experiment that are not properly attributable to RSL. The Si–O–Si peak
sits at 131° (24-atom), 134° (36-atom), and 131° (48-atom), 17–20° below the
NMR-derived 151° of Mauri et al. (2000) and Malfait et al. (2008). This
downshift is well within the range of values reported by Sarnthein,
Pasquarello & Car (1995) for Car-Parrinello melt-quench simulations at
the PBE level (136–142°), and Demuth et al. (1999) traced its origin to the
under-binding of the bridging O lone pair under semilocal functionals. The
broader O–Si–O distribution we obtain (FWHM 9–16°, vs the 9° of CPMD) and
the gradual broadening of both BADs with cell size are consistent with this
interpretation: they reflect the inclusion of slightly distorted tetrahedra
that PBE happens to stabilize, not a defect of the random-search procedure
itself. None of these features can be removed by a different sampling
scheme at the same level of theory.

The 2MR appearance and the non-Q⁴ excess are signatures specific to RSL.
Two-membered rings are present at finite Boltzmann weight in every system
size (R_N = 0.11, 0.12, 0.17 for 24-, 36-, 48-atom), and 16–40% of Si atoms
are classified as Q^n<4 across the three ensembles, while equilibrium
v-SiO₂ contains essentially zero of either species (Vashishta et al., 1990;
Stebbins, 1995; Pasquarello & Car, 1998). Figure 4 shows that the
microstates carrying these features sit in a high-energy tail of the basin
distribution. A linear regression of total relaxed energy against 2MR count
in the 24-atom ensemble gives 0.23 eV per additional 2MR — well below the
1.23 eV per isolated 2MR strain energy of Hamann (1998) because the
2MR-bearing microstates also relax other distortions that partially
compensate the strain, but unmistakably positive and large compared to k_B T.
At T = 2000 K the Boltzmann factor down-weights these basins, but does not
exclude them. The mismatch is not in the underlying minima — they exist
under PBE — but in the basin-frequency-as-hypervolume estimate that RSL
substitutes for kinetic basin selection. The partial-ergodicity argument of
Jones & Stevanović (2020) holds for the bulk of the basin distribution; it
breaks down at the strained tail, which physical glass formation excludes
by quench kinetics.

Two practical consequences follow for downstream use of RSL ensembles. For
properties dominated by the bulk of the basin distribution — pair
correlations, S(q), elastic constants, electronic DOS — the ensemble
averages are quantitatively reliable at a fraction of the cost of AIMD
melt-quench. For properties anchored on small populations of strained
configurations — Raman D-bands, 3MR/4MR populations against
Pasquarello & Car's (1998) experimental upper bounds, Q^n<4 fractions, and
the high-angle wing of the BAD — the unfiltered ensemble systematically
over-predicts. Three remediations are available. (i) An energy filter that
rejects microstates more than a chosen ΔE per atom above the ensemble
minimum, calibrated against any single experimental observable (e.g. the
3MR fraction), removes the strained tail without theory-of-glass-formation
machinery. (ii) A topology filter that rejects microstates containing
2MRs or Q^n<4 Si gives the same correction with a sharper geometric
criterion. (iii) A temperature-dependent reweighting that replaces the flat
T = 2000 K Boltzmann factor with a kinetic basin-occupation model would be
the most physical fix but requires additional input. The first option is
the simplest and is consistent with the spirit of inherent-structures
sampling.

This domain-of-validity result matters because RSL was advanced not for
structure prediction in isolation but as a route to electronic-property
calculations on glass that AIMD cannot afford (Jones & Stevanović, 2020;
Wolf et al., 2025). The structural validation here is what justifies that
extension. Where the ensemble averages converge — short-range geometry,
S(q), bulk modulus, FSDP — the corresponding electronic predictions
inherit the same convergence and can be trusted. Where they fail — at the
strained, low-population end of the basin distribution — the failure mode
is now characterized and bounded, and either filter (i) or (ii) is enough
to bring electronic-structure predictions back into the regime where the
ensemble is reliable. The bigger picture is that small-cell DFT plus
random structure searching plus inherent-structures averaging is a viable
substitute for melt-quench simulation of glass at the level of theory
where melt-quench is unaffordable, provided the analyst respects the
boundary identified here.
