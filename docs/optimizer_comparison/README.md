# DE versus Powell on the saved self-recognition experiment

Replacing Powell with raw-coordinate differential evolution is sufficient to
produce monotone safety–usefulness curves in this experiment. As usefulness
decreases, safety weakly increases at all ten saved points in both the passive
and SFT arms. No smoothing or monotonicity correction is applied.

![Five-way comparison](safety_usefulness_five_way_ablation_zoom.png)

The five curves compare main's Powell, Powell with a previous-guess payoff
guard, raw DE, raw DE with a known-policy check, and PR #18's scaled/log DE
with known-policy checks. Raw DE and both augmented DE variants are monotone
in this saved sweep; the extra checks and coordinate transforms have no
material visible benefit for this case. Powell with the guess guard still
reverses between SFT usefulness 0.99 and 0.98.

| SFT usefulness | Raw DE safety |
|---:|---:|
| 0.99 | 0.152116146 |
| 0.98 | 0.157336797 |

All 20 raw-DE points report convergence and have no final reversed bounds
beyond 1e-8. Four additional seed-offset pairs at SFT usefulness 0.99/0.98
preserve the expected ordering. The raw-DE 20-point sweep took approximately
280 seconds in the saved run. DE is heuristic, and these measurements do not
establish monotonicity or convergence for other experiments. The minimal
change retains the existing convergence logic, including clamping a negative
raw gap to zero. If non-monotone behavior recurs, revisit PR #18's known-policy
checks, log coordinates, and invalid-bound diagnostics.

`results.json` preserves the original unsmoothed measurements from
`experiments/self_rec_attack_500/results/grouped/all_monitors/main_vs_pr_matched_seed_500/`.
The sweep used 59 attack, 131 honeypot, and 379 honest saved monitor samples per
arm, with 5000 simulated tasks, budget fraction 0.02, and slack honeypots 5.
No new model calls were made. Each arm and method used initial seeds
`20260923 + usefulness-index` over usefulness
`[1, .99, .98, .95, .9, .8, .5, .2, .05, .01]`. A shared initial seed does not
imply identical subsequent random draws across optimizers.

The raw-DE implementation is commit `53376224b198ef36d260dbf9afd35fcb2bdada1a`
on `experiment/main-de-only`. The baseline and PR revision hashes are recorded
in `results.json`. DE uses `x0` equal to the existing previous-response guess,
the existing RNG and raw bounds, `maxiter=40`, `popsize=10`, `tol=1e-4`, and
`polish=False`. Finite candidates are retained even when DE reaches its
iteration limit. Guess selection and production seeding are unchanged.

Check the saved monotonicity evidence with the standard library:

```bash
python3 docs/optimizer_comparison/check_results.py
```

Related: [issue #17](https://github.com/jedrzej-kolbert/spar-2026-untrusted-monitoring/issues/17)
and [PR #18](https://github.com/jedrzej-kolbert/spar-2026-untrusted-monitoring/pull/18).
