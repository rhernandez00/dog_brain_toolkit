#!/usr/bin/env python
"""Schedule all analyses required to produce a thresholded, cluster-corrected
z-map and result tables for a given RSA model in dogs (D), humans (H), or both.

Usage:
  python schedule_rsa.py --dataset EmoC --model basic-block --rsa_model test-model
  python schedule_rsa.py --dataset EmoC --model basic-block --rsa_model test-model \\
      --specie D --reps 10 --reps_group 50 --z_threshold 3.1
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from scheduler.paths import get_paths, get_queue_dir
from scheduler.dag import build_job_graph
from scheduler.jobs import create_job


def main():
    parser = argparse.ArgumentParser(
        description="Schedule a full RSA analysis pipeline for a given model"
    )
    parser.add_argument("--dataset", required=True,
                        help="Dataset name (e.g. EmoC)")
    parser.add_argument("--model", default="basic-block",
                        help="GLM model name (default: basic-block)")
    parser.add_argument("--rsa_model", required=True,
                        help="RSA model name (CSV filename without .csv extension)")
    parser.add_argument("--specie", default="both", choices=["H", "D", "both"],
                        help="Species to schedule: H, D, or both (default: both)")
    parser.add_argument("--target_step", type=int, default=10,
                        help="Final pipeline step to schedule (default: 10)")
    parser.add_argument("--z_threshold", type=float, default=3.1,
                        help="Z-score threshold for cluster definition (default: 3.1)")
    parser.add_argument("--reps", type=int, default=100,
                        help="Number of permutations for step 4 (default: 100; use fewer for testing)")
    parser.add_argument("--reps_group", type=int, default=1000,
                        help="Number of group permutations for step 5 (default: 1000; use fewer for testing)")
    args = parser.parse_args()

    datafolder, _, _ = get_paths()
    queue_dir = get_queue_dir(datafolder)

    species = ["H", "D"] if args.specie == "both" else [args.specie]

    total_created = 0
    for specie in species:
        print(f"\n--- Scheduling {specie} (target: step {args.target_step}) ---")
        jobs = build_job_graph(
            dataset=args.dataset,
            model=args.model,
            rsa_model=args.rsa_model,
            specie=specie,
            target_step=args.target_step,
            z_threshold=args.z_threshold,
            reps=args.reps,
            reps_group=args.reps_group,
        )
        for job in jobs:
            created = create_job(queue_dir, job)
            marker = "+" if created else "="
            if created:
                total_created += 1
            print(
                f"  [{marker}] step {job['step']:02d} ({job['label']:<26}) -> {job['status']}"
            )

    print(f"\nCreated {total_created} new job(s) in {queue_dir}")
    print("Run 'python run_jobs.py' to execute pending jobs.")
    print("Run 'python job_status.py --dataset ... --rsa_model ...' to check progress.")


if __name__ == "__main__":
    main()
