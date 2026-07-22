#!/usr/bin/env python
"""Display the status of scheduled RSA analysis jobs as a step x species table.

Usage:
  python job_status.py --dataset EmoC --rsa_model test-model
  python job_status.py --dataset EmoC --model basic-block --rsa_model test-model \\
      --z_threshold 3.1 --reps 10 --reps_group 50
"""
import argparse
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from scheduler.paths import get_paths, get_queue_dir
from scheduler.dag import STEP_LABELS, make_job_id
from scheduler.jobs import job_files

STATE_LABEL = {
    "pending":   "READY ",
    "waiting":   "WAIT  ",
    "running":   "RUN   ",
    "completed": " OK   ",
    "failed":    " FAIL ",
}


def find_state(queue_dir, job_id):
    """Most-active state this job_id is in. Checks the canonical filename
    plus any ``__dup{N}`` re-run/duplicate instances (see create_job) so a
    fresh in-flight duplicate outranks a stale completed/failed record."""
    queue_dir = Path(queue_dir)
    for state in ("running", "pending", "waiting", "completed", "failed"):
        if job_files(queue_dir, job_id, state):
            return state
    return None


def main():
    parser = argparse.ArgumentParser(description="Show RSA job status")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="basic-block")
    parser.add_argument("--rsa_model", required=True)
    parser.add_argument("--method", default="mahalanobis")
    parser.add_argument("--z_threshold", type=float, default=3.1)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--reps_group", type=int, default=1000)
    args = parser.parse_args()

    datafolder, _, _ = get_paths()
    queue_dir = get_queue_dir(datafolder)

    print()
    print(f"Dataset   : {args.dataset}")
    print(f"Model     : {args.model}")
    print(f"RSA model : {args.rsa_model}")
    print(f"method    : {args.method}")
    print(f"z_thresh  : {args.z_threshold}  reps: {args.reps}  reps_group: {args.reps_group}")
    print()

    col_w = 10
    header = f"{'Step':<5} {'Label':<28} {'Dog (D)':>{col_w}} {'Human (H)':>{col_w}}"
    print(header)
    print("-" * len(header))

    for step in range(11):
        label = STEP_LABELS.get(step, f"Step {step}")
        row = f"{step:<5} {label:<28}"
        for specie in ("D", "H"):
            job_id = make_job_id(
                args.dataset, args.model, args.rsa_model, specie, step,
                args.z_threshold, args.reps, args.reps_group, args.method
            )
            state = find_state(queue_dir, job_id)
            cell = STATE_LABEL.get(state, "  --  ")
            row += f" {cell:>{col_w}}"
        print(row)
    print()


if __name__ == "__main__":
    main()
