"""Google Drive persistence helpers for Colab runs.

On Colab with /content/drive mounted, redirect persistent outputs
(checkpoints, results, data cache) into a single Drive folder so they
survive VM crashes / disconnects. Falls back to local paths otherwise.
"""

import os
from typing import Optional

# Folder under MyDrive that mirrors the repo's persistent directories.
DRIVE_BASE = "AOAPaper"


def drive_dir() -> Optional[str]:
    """Return the Drive mirror root if mounted, else None."""
    drive_mount = "/content/drive/MyDrive"
    if os.path.isdir(drive_mount):
        path = os.path.join(drive_mount, DRIVE_BASE)
        os.makedirs(path, exist_ok=True)
        return path
    return None


def persist_path(local_dir: str, default: str) -> str:
    """Route local_dir to its Drive mirror when Drive is mounted."""
    root = drive_dir()
    if root is None:
        return default
    path = os.path.join(root, os.path.basename(local_dir))
    os.makedirs(path, exist_ok=True)
    return path
