import os
import sys
from pathlib import Path


def get_paths():
    """Return (datafolder, git_folder, python_exe) for the current machine."""
    if os.name == 'nt':  # Windows workstation
        datafolder = os.path.join("P:\\userdata", 'raulh87', 'data')
        git_folder = r"C:\github"
        python_exe = r"C:\ProgramData\anaconda3\python.exe"
    else:  # Linux remote server
        datafolder = os.path.join(
            '/home', 'raulh87', 'mnt', 'a471', 'userdata', 'raulh87', 'data'
        )
        git_folder = os.path.join(
            '/home', 'raulh87', 'mnt', 'a471', 'userdata', 'raulh87', 'github'
        )
        # Launch worker subprocesses with the same interpreter that's running
        # this script: the remote has no bare `python` on PATH (only python3),
        # and sys.executable is guaranteed to exist with the right environment.
        python_exe = sys.executable or 'python3'
    return datafolder, git_folder, python_exe


def get_queue_dir(datafolder):
    """Return path to the job queue directory, creating subdirectories if needed."""
    queue_dir = Path(datafolder) / 'job_queue'
    for subdir in ('pending', 'waiting', 'running', 'completed', 'failed', 'logs'):
        (queue_dir / subdir).mkdir(parents=True, exist_ok=True)
    return queue_dir
