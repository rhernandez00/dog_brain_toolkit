#!/usr/bin/env python
"""Worker: claims and executes pending RSA analysis jobs from the shared queue.

Both the local Windows machine and the remote Linux server can run this script
simultaneously. Each job is claimed atomically so no job runs twice.

Usage:
  python run_jobs.py                      # run one pending job and stop
  python run_jobs.py --max_jobs 0 --loop  # run all jobs, keep polling for more
  python run_jobs.py --loop               # run jobs one at a time, keep polling
"""
import argparse
import subprocess
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from scheduler.paths import get_paths, get_queue_dir
from scheduler.jobs import claim_job, complete_job, fail_job


def build_command(job, git_folder, python_exe, marker_dir):
    searchlight = os.path.join(git_folder, "dog_brain_toolkit", "searchlight.py")
    cmd = [
        python_exe,
        "-u",  # unbuffered stdout/stderr so log files update in real time
        searchlight,
        "--dataset",        job["dataset"],
        "--model",          job["model"],
        "--rsa_model",      job["rsa_model"],
        "--specie",         job["specie"],
        "--rsa_method",     job.get("rsa_method", "kendall"),
        "--dis_method",     job.get("dis_method", "mahalanobis"),
        "--mah_fold",       job.get("mah_fold", "stim-wise"),
        "--steps_to_run",   str(job["step"]),
        "--z_threshold",    str(job["z_threshold"]),
        "--reps",           str(job["reps"]),
        "--reps_group",     str(job["reps_group"]),
        "--job_marker_dir", str(marker_dir),
    ]
    # Fields below are only present on dashboard-scheduled jobs; classic
    # scheduler jobs omit them and fall back to searchlight.py's own defaults.
    if job.get("radius") is not None:
        cmd += ["--radius", str(job["radius"])]
    if job.get("mask_type"):
        cmd += ["--mask_type", str(job["mask_type"])]
    if job.get("participant") is not None:
        # Scope a per-participant job to a single subject.
        cmd += ["--participants_forced", str(job["participant"])]
    if job.get("replace_file"):
        cmd.append("--replace_file")
    if job.get("replace_rnd_files"):
        cmd.append("--replace_rnd_files")
    return cmd


def run_job(job, git_folder, python_exe, log_dir, marker_dir):
    cmd = build_command(job, git_folder, python_exe, marker_dir)
    log_path = Path(log_dir) / f"{job['job_id']}_{int(time.time())}.log"
    print(f"  cmd : {' '.join(cmd)}")
    print(f"  log : {log_path}")
    with open(log_path, "w") as log_f:
        result = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, text=True)
    return result.returncode, str(log_path)


def main():
    parser = argparse.ArgumentParser(
        description="Run pending RSA analysis jobs from the shared queue"
    )
    parser.add_argument("--max_jobs", type=int, default=0,
                        help="Max jobs to run before stopping (0 = unlimited; default: 0)")
    parser.add_argument("--loop", action="store_true",
                        help="Keep polling for new jobs when queue is empty")
    parser.add_argument("--poll_interval", type=int, default=60,
                        help="Seconds between polls when --loop is active (default: 60)")
    args = parser.parse_args()

    datafolder, git_folder, python_exe = get_paths()
    queue_dir = get_queue_dir(datafolder)
    log_dir = queue_dir / "logs"
    markers_root = queue_dir / "markers"

    print(f"Queue : {queue_dir}")
    print(f"Python: {python_exe}")

    jobs_run = 0
    while True:
        path, job = claim_job(queue_dir)

        if job is None:
            if args.loop:
                print(f"No pending jobs. Polling again in {args.poll_interval}s ...")
                time.sleep(args.poll_interval)
                continue
            else:
                print("No pending jobs found.")
                break

        print(f"\n[JOB] {job['job_id']}")
        print(f"  step  : {job['step']} ({job['label']})")
        print(f"  specie: {job['specie']}")

        marker_dir = markers_root / job["job_id"]
        expected_marker = marker_dir / f"{job['step']}.done"

        returncode, log_path = run_job(job, git_folder, python_exe, log_dir, marker_dir)

        if returncode == 0 and expected_marker.exists():
            complete_job(queue_dir, path, job)
            print(f"  [OK] completed successfully")
        elif returncode == 0 and not expected_marker.exists():
            fail_job(queue_dir, path, job,
                     f"exit code 0 but marker not found — step may have silently failed | log: {log_path}")
            print(f"  [FAIL] no marker file — step did not signal success, see {log_path}")
        else:
            fail_job(queue_dir, path, job, f"exit code {returncode} | log: {log_path}")
            print(f"  [FAIL] exit code {returncode}, see {log_path}")

        jobs_run += 1
        if args.max_jobs and jobs_run >= args.max_jobs:
            print(f"\nReached --max_jobs {args.max_jobs}. Stopping.")
            break

    print(f"\nDone. Jobs run this session: {jobs_run}")


if __name__ == "__main__":
    main()
