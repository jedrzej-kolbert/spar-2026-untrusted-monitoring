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

In the paper's model, Blue commits to a mixed strategy and Red chooses a
deterministic response after seeing that commitment. The paper wiki describes
Blue's choice as the honeypot/audit allocation and Red's choice as an attack
rate and self-recognition threshold. A draw from Blue's mixture is not revealed
to Red. Let `A[i,j]` be Blue's modelled safety for the `i`th Blue strategy and
`j`th Red strategy already evaluated by the offline solver. The restricted-game
LP returns mixtures `p` and `q`. The intended Red response solves

`min_r sum_i p[i] * safety(blue_i, r)`.

`red_strategies` contains earlier candidates `r_j`, not private observations
from deployment. The check against them computes `min_j (p @ A)[j]` from
payoffs the offline solver already evaluated. It neither changes Red's action
space nor gives the deployed Red model a new signal. The substantive threat
model assumption is that Red can respond to Blue's *committed distribution*
`p`; the check does not add that assumption.

Each computed response is checked against its starting guess and the best
strategy already in the restricted game. This ensures that the reported
lower and upper response values do not reverse merely because a local search
missed a known strategy. It does not establish that either response is globally
best. In particular, the local-search values called `lower_bound` and
`upper_bound` are not certified bounds on the unrestricted game's value;
they are candidate response payoffs. `converged` only means their discovered
gap met the tolerance and neither local search missed a known strategy.
Independent starts or a global response audit are needed to assess how close
the estimated curve is to the true frontier.
