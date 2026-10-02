# Two-channel duality, fold memory, and arithmetic blindness in polymer contact networks

Paper 4 of the prime-driven topological polymer (PDTP) series.

> R. Chen, *Two-channel duality, fold memory, and arithmetic blindness in polymer contact networks* (2026).
> DOI: [10.5281/zenodo.23100245](https://doi.org/10.5281/zenodo.23100245).
> Repository: https://github.com/Ruqing1963/contact-network-two-channels

## The question

A collapsed chain is a network: beads are vertices, bonds and contacts are edges. Algorithmic class field
theory (*ANCCFT*, DOI 10.5281/zenodo.22883908) organizes graph invariants into two channels:

- the **bulk** (Laplacian) channel, with the sandpile group, divisors and the Hodge decomposition;
- the **boundary** (Perron) channel.

On arithmetic quotients of a tree only the bulk channel sees arithmetic. What do the two channels of the PDTP
contact network see of the prime sequence?

## The answer

The analysis uses 82 stored conformations at N = 40 000: 24 true, 8 S1 shuffles and 40 S6″ (PNT-only)
production globules, plus 5 true and 5 S1 fully relaxed globules.

| Invariant | Result |
|---|---|
| Unweighted: degree, spanning trees, λ₂, spectral radius, exact coarse sandpile group | No arm separation. |
| Bulk channel: divisor energy Dᵀ Δ⁺ D | True vs S1 Z_seq = +11.4 (production and relaxed); True vs S6″ p = 0.76. It reads the PNT envelope. |
| Bulk channel with Legendre modes ≤ K removed (K = 4, 8, 16, 32) | All 16 contrasts null; p_Holm = 1. The pilot's p = 0.043 does not replicate. |
| Boundary channel: Perron correlation | No contrast survives correction. |
| Contact cycle share H_c (Hodge) | Equal across arms. |
| Own-sequence memory: every statistic vs gap shuffles on the same conformation | z ≈ 6–24 in every channel, equal for primes, shuffles and PNT-only surrogates. |
| Albanese energies | Backbone winding = N exactly. The closure flux reduces to the end-to-end resistance. κ(Y*)·H is an integer on 82/82 graphs (exact, PARI/GP). |

The two-channel dichotomy survives, with positional in place of arithmetic content. The folded network
remembers its sequence, but nothing in it distinguishes the primes from a smooth model of their density.

## Layout

```
code/      contact_graph.py   contact graph, incidence, Laplacian solves, conformation loading, gap shuffles
           legendre_filter.py orthogonal removal of smooth backbone modes
           hodge.py           discrete Hodge decomposition of edge 1-forms
           albanese.py        winding energies, end-to-end resistance, exact PARI/GP check
           stats_re.py        sequence-level statistic, Holm and Benjamini-Hochberg corrections
           core.py            exact PDTP stiffness map and sieve (PDTP v4.0, unmodified)
scripts/   run_legendre_full_ensemble.py   bulk channel, envelope removal       (Table I, Fig. 1)
           run_hodge_decomposition.py      contact cycle share
           run_boundary_channel.py         Perron channel
           run_albanese_mass.py            Albanese energies, closure flux      (Fig. 3)
           run_unweighted_and_sandpile.py  unweighted invariants, exact sandpile arithmetic
           plot_figures.py                 Figs. 1-3
data/      CSV outputs of every script (per-globule results and contrast tables)
paper/     main.tex, references.bib, figures/
tests/     test_operators.py, run_all_tests.py
```

## Reproduce

The conformations are not duplicated here. They are in the PDTP v4.0 repository
([prime-driven-topological-polymer](https://github.com/Ruqing1963/prime-driven-topological-polymer), `data/v4/conf`
and `data/v4/relaxed_endpoints`). Point `PDTP_V4_DATA` at that `data/v4` directory:

```bash
pip install -r requirements.txt
export PDTP_V4_DATA=/path/to/prime-driven-topological-polymer/data/v4
python tests/run_all_tests.py                    # 12 tests
python scripts/run_legendre_full_ensemble.py     # ~6 min on 6 cores
python scripts/run_hodge_decomposition.py        # ~5 min
python scripts/run_boundary_channel.py           # ~1 min
python scripts/run_albanese_mass.py              # ~6 min, needs PARI/GP for the exact check
python scripts/run_unweighted_and_sandpile.py    # needs PARI/GP (set GP_EXE)
python scripts/plot_figures.py
```

Gap-shuffle nulls are seeded with `zlib.crc32` of the file name and are reproducible bit for bit.

## Related work

- PDTP v4.0: DOI [10.5281/zenodo.22944122](https://doi.org/10.5281/zenodo.22944122)
- *Collapse, convergence, and the scale of the prime-gap imprint*: DOI [10.5281/zenodo.23076433](https://doi.org/10.5281/zenodo.23076433)
- *What a prime-programmed polymer can read*: DOI [10.5281/zenodo.23094095](https://doi.org/10.5281/zenodo.23094095)
- *Algorithmic Non-commutative Class Field Theory*: DOI [10.5281/zenodo.22883908](https://doi.org/10.5281/zenodo.22883908)

## License

MIT.
