"""Read-only pipeline watcher; --once is a runnable status check.

Read compact Inspect journal summaries, not full trajectories. Monitor expires
independently of the actual experiment; re-run this watcher to renew the watch.
"""

import json
import re
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "run_logs/remaining_generations_20261001.log"
PID = (ROOT / "run_logs/remaining_generations.pid").read_text().strip()
position = LOG.stat().st_size
previous = None
seen_errors = set()
while True:
    with LOG.open(errors="replace") as file:
        file.seek(position)
        text = file.read()
        position = file.tell()
    for line in text.splitlines():
        if re.match(r"^===|^Traceback|^[A-Za-z]+(?:Error|Exception):", line):
            print(line, flush=True)
    logs = list((ROOT / "logs").glob("*bash-arena*.eval"))
    if logs:
        latest = max(logs, key=lambda path: path.stat().st_mtime)
        try:
            with zipfile.ZipFile(latest) as archive:
                records = [
                    sample
                    for name in archive.namelist()
                    if name.startswith("_journal/summaries/")
                    for sample in json.loads(archive.read(name))
                ]
                count = sum(name.startswith("samples/") for name in archive.namelist())
            errors = [sample for sample in records if sample.get("error")]
            status = (latest.name, count, len(errors))
            initial = previous is None
            if status != previous:
                if not initial or "--once" in sys.argv:
                    print(
                        f"Generation: {count} saved records, {len(errors)} summarized errors in {latest.name}",
                        flush=True,
                    )
                previous = status
            for sample in errors:
                key = (latest.name, str(sample["id"]))
                if key not in seen_errors:
                    if not initial or "--once" in sys.argv:
                        print(
                            f"Sample failure {sample['id']}: {str(sample['error'])[:240]}",
                            flush=True,
                        )
                    seen_errors.add(key)
        except (zipfile.BadZipFile, FileNotFoundError, KeyError, json.JSONDecodeError):
            pass  # The recorder can replace the archive during a read.
    try:
        live = Path(f"/proc/{PID}/stat").read_text().split()[2] != "Z"
    except FileNotFoundError:
        live = False
    if not live:
        print(
            "Expanded pipeline ended; verify completion marker and final outputs.",
            flush=True,
        )
    if not live or sys.argv[1:] == ["--once"]:
        break
    time.sleep(30)
