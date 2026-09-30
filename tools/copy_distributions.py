"""Reuse searchlight steps 6-8 null distributions between RSA models.

Example (adjust dataset, GLM model and radius to the source analysis):
    python copy_distributions.py --dataset EmoC --model basic --radius 3 \
        --source_model LcSSG --target_models RcSG RmSSG --specie H --dry_run

Remove --dry_run to copy. Then run searchlight.py separately for each target,
using --rsa_model RcSG (or RmSSG) --steps_to_run 75 9 and the same analysis
settings. The targets must already have their own step-3 real group maps.

Only reuse a null distribution when scientifically appropriate for the target
analysis (same cohort, grid, mask, radius, methods and permutation design).
Step-6 filenames do not encode radius or methods, so these cannot be verified
from their names. Step 9 must use a z threshold present in the copied step-8
dictionary. This script copies that dictionary intact, including all thresholds.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil


def path_component(value):
    """Keep model names and CLI selectors within their expected directories."""
    if not value or value in (".", "..") or any(c in value for c in '/\\:*?"<>|'):
        raise argparse.ArgumentTypeError(f"Expected a single path component: {value!r}")
    return value


def build_plan(args):
    """Return source/destination pairs and provenance paths; validate first."""
    if args.source_model in args.target_models:
        raise ValueError("The source model cannot also be a target model.")
    if len(set(args.target_models)) != len(args.target_models):
        raise ValueError("Target models must be unique.")
    root = args.datafolder / args.dataset / "results"
    rnd = root / "RSA_rnd" / args.model
    rsa = root / "RSA" / args.model
    stem = f"{args.specie}-r-{args.radius}_{args.dis_method}_{args.rsa_method}"
    source = args.source_model
    required = [rnd / f"{args.specie}-{source}_{stat}.nii.gz" for stat in ("mean", "std")]
    cluster = rsa / source / "dist" / f"{stem}_dist.npy"
    required.append(cluster)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required source distribution not found: {path}")

    z_maps = []
    if args.include_z_maps:
        z_maps = sorted(p for p in (rnd / source / "mean").glob(f"{stem}_z_*.nii.gz")
                        if re.fullmatch(re.escape(stem) + r"_z_\d+\.nii\.gz", p.name))
        if not z_maps:
            raise FileNotFoundError(f"No step-7 permutation z maps found for {stem}")

    plan, receipts = [], []
    for target in args.target_models:
        pairs = [(required[i], rnd / f"{args.specie}-{target}_{stat}.nii.gz")
                 for i, stat in enumerate(("mean", "std"))]
        pairs.append((cluster, rsa / target / "dist" / cluster.name))
        # Preserve original log contents, including their source model paths.
        for src, dst in list(pairs):
            extension = ".nii.gz" if src.name.endswith(".nii.gz") else ".npy"
            src_log = src.with_name(src.name.removesuffix(extension) + "_log.txt")
            dst_log = dst.with_name(dst.name.removesuffix(extension) + "_log.txt")
            if src_log.is_file():
                pairs.append((src_log, dst_log))
            elif dst_log.exists():
                raise FileExistsError(f"Target has a log but the source does not: {dst_log}")
        pairs.extend((p, rnd / target / "mean" / p.name) for p in z_maps)
        receipt = rsa / target / "dist" / f"{stem}_copied_distributions.json"
        for src, dst in pairs:
            if src.resolve() == dst.resolve():
                raise ValueError(f"Source and target resolve to the same file: {dst}")
        plan.extend(pairs)
        receipts.append((target, receipt, pairs))

    for dst in [dst for _, dst in plan] + [p for _, p, _ in receipts]:
        if dst.exists() and (not args.overwrite or not dst.is_file()):
            raise FileExistsError(f"Destination exists: {dst}. Use --overwrite to replace files.")
    return plan, receipts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_data = (r"P:\userdata\raulh87\data" if os.name == "nt" else
                    "/home/raulh87/mnt/a471/userdata/raulh87/data")
    parser.add_argument("--datafolder", type=Path, default=Path(default_data))
    parser.add_argument("--dataset", type=path_component, default="EmoC")
    parser.add_argument("--model", type=path_component, default="basic", help="GLM model, as in searchlight.py")
    parser.add_argument("--source_model", type=path_component, default="LcSSG", help="Source RSA model")
    parser.add_argument("--target_models", type=path_component, nargs="+", default=["RcSG", "RmSSG"], help="Target RSA models")
    parser.add_argument("--specie", choices=["H", "D"], default="H")
    parser.add_argument("--radius", type=int, required=True)
    parser.add_argument("--dis_method", type=path_component, default="mahalanobis")
    parser.add_argument("--rsa_method", type=path_component, default="kendall")
    parser.add_argument("--include_z_maps", action="store_true", help="Also copy step-7 permutation z maps; unnecessary for steps 75 and 9")
    parser.add_argument("--dry_run", action="store_true", help="Validate and show copies without writing")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing destination files")
    args = parser.parse_args(argv)
    if args.radius <= 0:
        parser.error("--radius must be positive")
    try:
        plan, receipts = build_plan(args)
        for src, dst in plan:
            print(f"{src}\n  -> {dst}")
        if args.dry_run:
            print(f"Dry run: {len(plan)} files and {len(receipts)} provenance records; nothing written.")
            return 0
        for src, dst in plan:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        for target, path, pairs in receipts:
            path.write_text(json.dumps({
                "copied_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_rsa_model": args.source_model, "target_rsa_model": target,
                "dataset": args.dataset, "glm_model": args.model, "specie": args.specie,
                "radius": args.radius, "dis_method": args.dis_method, "rsa_method": args.rsa_method,
                "files": [{"source": str(src.resolve()), "destination": str(dst.resolve())}
                          for src, dst in pairs],
            }, indent=2) + "\n", encoding="utf-8")
        print(f"Copied {len(plan)} files. Targets can now use searchlight.py --steps_to_run 75 9.")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
