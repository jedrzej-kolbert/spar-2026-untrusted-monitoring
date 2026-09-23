# Optional PRIMA BOBYQA best responses

The double-oracle optimizer can use [PRIMA's BOBYQA](https://github.com/libprima/prima)
for bounded, derivative-free best-response searches. Powell remains the default.
PRIMA is **local**: a small stopping gap is not a certificate of a global Nash
equilibrium. The optimizer compares each returned response with its starting
guess and the best strategy already in the restricted game. It records a local
search miss in the analysis history and does not mark that iteration converged.

The Python binding crashed in the environment used for the original comparison.
This opt-in backend instead uses PRIMA's C interface through a small subprocess.
Building it requires a C compiler, a Fortran compiler, and CMake. The source
revision below is the one used for the comparison:

```bash
git clone https://github.com/libprima/prima.git /tmp/libprima-prima
git -C /tmp/libprima-prima checkout 43863c69ccd6408dedcd782d0dfdcbaad404dcde
bash tools/prima/build_runner.sh /tmp/libprima-prima /tmp/prima_bobyqa_runner
export PRIMA_BOBYQA_RUNNER=/tmp/prima_bobyqa_runner
```

From the repository root, use the existing monitor configs and `.eval` files:

```bash
MPLBACKEND=Agg uv run \
  src/lasr_labs_2025_control_project/scripts/plot_data.py \
  --config path/to/monitor_passive.yaml path/to/monitor_sft.yaml \
  --best-response-method PRIMA-BOBYQA \
  --grouped-path passive_vs_sft_prima
```

Use fresh `results_folder` values in the configs to preserve earlier analysis
files. This command reads saved monitor scores and reruns only the numerical
optimization; it makes no model calls. The C runner and Python adapter exchange
`EVAL` and `RESULT` records over pipes. Blue's honeypot fraction and red's ROC
threshold are optimized linearly on `[0, 1]`; red's attack probability is
normalized in log space.

The original 500-problem comparison used 379 benign solutions, 59 attacks, and
131 honeypots. Two PRIMA runs gave monotone SFT curves, but an independent
red-response search found a better strategy at usefulness 0.90. Treat these
curves as exploratory until final best responses are checked with multistart or
global searches.
