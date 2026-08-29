#!/usr/bin/env python
"""Worker: claims and executes pending RSA analysis jobs from the shared queue.

Both the local Windows machine and the remote Linux server can run this script
simultaneously. Each job is claimed atomically so no job runs twice.

Pending jobs are claimed by priority: every pending priority-1 job runs before
any priority-2 job, and those before priority-3 (the default, and what jobs
queued before priorities existed count as). Within one priority the order is by
job id, as before. Priority only orders the *pending* pool — a waiting job still
has to have its dependencies complete before it can be claimed at all.

Usage:
  python run_jobs.py                      # run one pending job and stop
  python run_jobs.py --max_jobs 0 --loop  # run all jobs, keep polling for more
  python run_jobs.py --loop               # run jobs one at a time, keep polling
  python run_jobs.py --verbose            # force --verbose on every job it runs
  python run_jobs.py --no_verbose         # force verbose off, whatever the job says
  python run_jobs.py --quiet              # log only, do not echo job output here

Job output is written to the job's log file and mirrored to this terminal as it
is produced (use --quiet to suppress the terminal copy).
"""
import argparse
import subprocess
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from scheduler.paths import get_paths, get_queue_dir
from scheduler.jobs import claim_job, complete_job, fail_job, job_priority


def build_command(job, git_folder, python_exe, marker_dir, verbose_override=None):
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
    if job.get("min_percentage_available") is not None:
        cmd += ["--min_percentage_available", str(job["min_percentage_available"])]
    if job.get("participant") is not None:
        # Scope a per-participant job to a single subject.
        cmd += ["--participants_forced", str(job["participant"])]
    if job.get("replace_file"):
        cmd.append("--replace_file")
    if job.get("replace_rnd_files"):
        cmd.append("--replace_rnd_files")
    if job.get("shuffle_participants"):
        # Set by create_job() when this job duplicates one already in the
        # queue, so a concurrent duplicate walks participants/permutations
        # in a different order rather than racing the other instance
        # file-by-file.
        cmd.append("--shuffle_participants")
    # verbose defaults to True on jobs built via scheduler/dag.py; older queued
    # job files without the field also get --verbose via this default.
    # run_jobs.py --verbose / --no_verbose overrides whatever the job says, so a
    # worker can be made (non-)chatty without re-scheduling the queue.
    verbose = job.get("verbose", True) if verbose_override is None else verbose_override
    if verbose:
        cmd.append("--verbose")
    return cmd


def run_job(job, git_folder, python_exe, log_dir, marker_dir, verbose_override=None,
            echo=True):
    cmd = build_command(job, git_folder, python_exe, marker_dir, verbose_override)
    log_path = Path(log_dir) / f"{job['job_id']}_{int(time.time())}.log"
    print(f"  cmd : {' '.join(cmd)}")
    print(f"  log : {log_path}")

    # Child prints (searchlight.py --verbose) go to the log file and, unless
    # --quiet, to this terminal as well. Force UTF-8 in the child so non-ASCII
    # output does not blow up on the Windows console codepage.
    env = dict(os.environ, PYTHONIOENCODING="utf-8")

    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,          # line buffered, so the tee streams live
            env=env,
        )
        for line in proc.stdout:
            log_f.write(line)
            log_f.flush()
            if echo:
                sys.stdout.write(line)
                sys.stdout.flush()
        proc.stdout.close()
        returncode = proc.wait()

    return returncode, str(log_path)


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
    parser.add_argument("--verbose", dest="verbose", action="store_true", default=None,
                        help="Force --verbose on searchlight.py for every job this worker runs, "
                             "ignoring the job's own 'verbose' field")
    parser.add_argument("--no_verbose", dest="verbose", action="store_false",
                        help="Force verbose OFF for every job this worker runs")
    parser.add_argument("--quiet", action="store_true",
                        help="Do not echo job output to this terminal (it still goes to the log file)")
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
        print(f"  prio  : {job_priority(job)}")

        marker_dir = markers_root / job["job_id"]
        expected_marker = marker_dir / f"{job['step']}.done"

        returncode, log_path = run_job(job, git_folder, python_exe, log_dir, marker_dir,
                                       verbose_override=args.verbose,
                                       echo=not args.quiet)

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
