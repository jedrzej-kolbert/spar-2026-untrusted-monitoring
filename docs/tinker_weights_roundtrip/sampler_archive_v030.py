"""Download a sampler checkpoint with the SDK 0.30 archive endpoint (the route the
Nov 2026 HF move used). Run with the main checkout's venv (tinker 0.30):
    python -I sampler_archive_v030.py tinker://.../sampler_weights/x OUT_DIR
"""

import sys
import tarfile
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv("/home/jkk/spar-2026-untrusted-monitoring/.env")
import tinker  # noqa: E402

path, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
rest = tinker.ServiceClient().create_rest_client()
url = rest.get_checkpoint_archive_url_from_tinker_path(path).result().url
tar_path = out / "archive.tar"
urllib.request.urlretrieve(url, tar_path)
with tarfile.open(tar_path) as tar:
    tar.extractall(out, filter="data")
print(sorted(p.name for p in out.iterdir()))
