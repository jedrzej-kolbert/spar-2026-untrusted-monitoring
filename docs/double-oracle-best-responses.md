# Double-oracle best responses

The double-oracle solver supports `Powell` (default) and SciPy `COBYQA` for
bounded, derivative-free best-response searches. To select COBYQA when
regenerating plots from saved `.eval` files:

```bash
MPLBACKEND=Agg uv run \
  src/lasr_labs_2025_control_project/scripts/plot_data.py \
  --config path/to/monitor_passive.yaml path/to/monitor_sft.yaml \
  --best-response-method COBYQA \
  --grouped-path passive_vs_sft_cobyqa
```

Use separate results folders in the configs if earlier analysis files should
be preserved. This reruns numerical optimization without model calls. COBYQA
searches a unit box; parameters marked `log` are transformed into log space
before normalization.

The restricted-game LP and alternating response searches follow the structure
of [Algorithm 2 in Griffin et al.](https://arxiv.org/pdf/2409.07985). That
algorithm assumes true best responses in its `arg max` and `arg min` steps.
The paper's implementation uses grid search for Blue and grid plus ternary
search for Red. Our local searches, finite iteration cap, and usefulness-target
formulation are approximations; the paper's equilibrium guarantee does not
apply here.

Each computed response is checked against its starting guess and the best
strategy already in the restricted game. This ensures that the reported
lower and upper response values do not reverse merely because a local search
missed a known strategy. It does not establish that either response is globally
best, so `converged` only means the discovered local-response gap met the
tolerance. Independent starts or a global response audit are needed to assess
how close the estimated curve is to the true frontier.
