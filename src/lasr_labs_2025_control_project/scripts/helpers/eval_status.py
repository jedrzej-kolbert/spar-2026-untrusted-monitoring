from pathlib import Path

from inspect_ai.log import read_eval_log


def finished_successfully(path: Path) -> bool:
    """True if `path` holds a complete eval log whose run succeeded.

    The pipeline scripts copy a failed run's log to save_path too (then exit
    non-zero), so existence alone does not mean the run is done.
    """
    if not path.exists() or not path.stat().st_size:
        return False
    try:
        return read_eval_log(path, header_only=True).status == "success"
    except Exception:
        return False
