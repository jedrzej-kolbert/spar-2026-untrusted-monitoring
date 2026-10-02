"""Build the test-range task images up front and list the ones that cannot build.

Inspect aborts a whole eval when one task image fails to build, so find them first.
Run with PYTHONPATH=src and the integration Python; prints one line per failure.
"""

import subprocess
import sys
from pathlib import Path

from control_arena import EvalMode
from control_arena.settings.bash_arena.bash_arena_setting import BashArenaSetting

start, stop = int(sys.argv[1]), int(sys.argv[2])
test = [
    s
    for s in BashArenaSetting().get_dataset(EvalMode.ATTACK)
    if int(str(s.id)) % 2 == 0
]
failed = []
for i, sample in enumerate(test[start:stop], start):
    compose = Path(sample.sandbox.config)
    try:
        result = subprocess.run(
            ["docker", "compose", "-f", str(compose), "build", "--quiet"],
            capture_output=True,
            text=True,
            timeout=1200,
        )
        ok, detail = (
            result.returncode == 0,
            result.stderr.strip().splitlines()[-1:] or [""],
        )
    except subprocess.TimeoutExpired:
        ok, detail = False, ["build timed out after 1200 s"]
    if not ok:
        failed.append(str(sample.id))
        print(f"FAILED position {i} task {sample.id}: {detail[0][:200]}", flush=True)
    elif i % 20 == 0:
        print(f"built through position {i}", flush=True)
print("FAILED_IDS", failed, flush=True)
