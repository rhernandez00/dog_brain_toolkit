"""Complete missing real step-15 fits in published step-15.4 participant ZIPs.

The original runner's cache signature includes input mtimes. Re-synced packages
therefore cannot safely be run in-place: a signature miss can discard the saved
permutations. This module computes step 15 in a separate local folder, merges
those products with each original ZIP, validates the merged ZIP, and replaces
the original through a Drive .part file. Unchanged archives are skipped.
"""

import json
import tempfile
import zipfile
from pathlib import Path

import run_colab_regression as runner


def _paths(manifest, regression, target, entry, reps):
    real = runner.gpu.run_folder(manifest, regression, target, entry, False)
    rnd = runner.gpu.run_folder(manifest, regression, target, entry, True)
    stem, _ = runner.gpu.map_stem(manifest)
    real_files = [(real / f"{stem}_{name}_map.nii.gz").as_posix()
                  for name in ("beta", "t", "p")]
    real_files.append((real / f"{stem}_regression.json").as_posix())
    null_files = []
    for index in range(reps):
        stem, suffix = runner.gpu.map_stem(manifest, index)
        null_files.extend((rnd / f"{stem}_{name}_map{suffix}.nii.gz").as_posix()
                          for name in ("beta", "t", "p"))
        null_files.append((rnd / f"{stem}_regression{suffix}.json").as_posix())
    return real_files, null_files, (real / "colab_run_receipt.json").as_posix()


def audit_archives(packages_dir, results_dir, support_zip, *, specie="H",
                   dataset="EmoB", model="basic-block", regression="visual_3",
                   dis_method="correlation", models=None, reps=100, seed=42):
    """Require complete saved permutations and identify missing real products."""
    with zipfile.ZipFile(support_zip) as archive:
        support = json.loads(archive.read("regression_manifest.json"))
    for key, expected in dict(dataset=dataset, model=model, regression_model=regression,
                              dis_method=dis_method).items():
        if support[key] != expected:
            raise ValueError(f"Support ZIP {key}={support[key]!r}, expected {expected!r}")
    participants = support["participants_by_species"][specie]
    packages = runner.discover_packages(packages_dir, specie, dataset,
                                         participants=participants, model=model,
                                         dis_method=dis_method)
    selected = list(models) if models is not None else support["models"]
    if not selected or set(selected) - set(support["models"]) or set(selected) & set(support["controls"]):
        raise ValueError("Invalid target model selection")
    pending = {}
    complete = 0
    inspected = 0
    total = len(packages) * len(selected)
    for sub, (_, manifest) in packages.items():
        for target in selected:
            path = Path(results_dir) / f"result_regression_{target}_{specie}-sub-{sub:02d}.zip"
            if not path.is_file():
                raise FileNotFoundError(f"Step-15.4 participant ZIP is missing: {path}")
            with zipfile.ZipFile(path) as archive:
                infos = archive.infolist()
                names = {info.filename for info in infos}
                if len(names) != len(infos):
                    raise ValueError(f"Duplicate ZIP members: {path}")
                receipt = runner._receipt_relative(manifest, regression, target).as_posix()
                if receipt not in names:
                    raise ValueError(f"Missing participant receipt: {path}")
                steps = json.loads(archive.read(receipt)).get("steps", [])
                if 15.4 not in steps:
                    raise ValueError(f"Step 15.4 is incomplete in {path}")
                real_missing = []
                for entry in manifest["runs"]:
                    real, null, _ = _paths(manifest, regression, target, entry, reps)
                    absent_null = [name for name in null if name not in names]
                    if absent_null:
                        raise ValueError(f"{path.name}: {len(absent_null)} step-15.4 files missing; first: {absent_null[0]}")
                    real_missing.extend(name for name in real if name not in names)
            if real_missing or 15 not in steps:
                pending.setdefault(sub, {})[target] = len(real_missing)
            else:
                complete += 1
            inspected += 1
            if inspected % 40 == 0 or inspected == total:
                runner.log(f"Audited {inspected}/{total} participant ZIPs")
    return support, packages, selected, pending, complete


def _merge(original_path, delta_path, output_path, manifest, regression, target, reps):
    receipt = runner._receipt_relative(manifest, regression, target).as_posix()
    real_files, run_receipts = set(), set()
    for entry in manifest["runs"]:
        real, _, run_receipt = _paths(manifest, regression, target, entry, reps)
        real_files.update(real)
        run_receipts.add(run_receipt)
    with zipfile.ZipFile(original_path) as original:
        old_names = set(original.namelist())
        old_receipt = json.loads(original.read(receipt))
        if delta_path is not None:
            delta = zipfile.ZipFile(delta_path)
            delta_names = set(delta.namelist())
            missing = real_files - delta_names
            if missing:
                delta.close()
                raise ValueError(f"Step-15 delta missing {len(missing)} files; first: {sorted(missing)[0]}")
            delta_receipt = json.loads(delta.read(receipt))
        else:
            delta, delta_names, delta_receipt = None, set(), None
            if real_files - old_names:
                raise ValueError("Receipt-only repair requested for incomplete maps")
        try:
            with zipfile.ZipFile(output_path, "w", allowZip64=True) as merged:
                for info in original.infolist():
                    name = info.filename
                    if name == receipt or name in run_receipts or (delta and name in real_files):
                        continue
                    merged.writestr(info, original.read(info))
                if delta:
                    for info in delta.infolist():
                        if info.filename in real_files:
                            merged.writestr(info, delta.read(info))
                for name in sorted(run_receipts):
                    old_run = json.loads(original.read(name)) if name in old_names else {}
                    new_run = json.loads(delta.read(name)) if delta and name in delta_names else {}
                    record = dict(old_run or new_run)
                    record["steps"] = sorted(set(old_run.get("steps", [])) | set(new_run.get("steps", [])) | {15, 15.4})
                    if new_run:
                        record["step15_signature"] = new_run.get("signature")
                    merged.writestr(name, json.dumps(record, indent=2))
                old_receipt["steps"] = sorted(set(old_receipt["steps"]) | {15})
                if delta_receipt:
                    old_receipt["step15_signature"] = delta_receipt.get("signature")
                    old_receipt["step15_source"] = "saved 15.4 ZIP plus newly computed real fits"
                merged.writestr(receipt, json.dumps(old_receipt, indent=2))
        finally:
            if delta:
                delta.close()
    with zipfile.ZipFile(output_path) as check:
        infos = check.infolist()
        if len(infos) != len({info.filename for info in infos}):
            raise ValueError(f"Duplicate members in merged ZIP: {output_path}")
        if check.testzip() is not None:
            raise ValueError(f"Corrupt merged ZIP: {output_path}")
        names = set(check.namelist())
        if real_files - names:
            raise ValueError(f"Real step-15 maps absent in merged ZIP: {output_path}")
        if 15.4 not in json.loads(check.read(receipt))["steps"]:
            raise ValueError(f"Step-15.4 receipt lost in merged ZIP: {output_path}")


def complete_saved_step15(packages_dir, results_dir, support_zip, *,
                          specie="H", dataset="EmoB", model="basic-block",
                          regression="visual_3", dis_method="correlation", models=None,
                          reps=100, reps_group=1000, seed=42, device="cuda",
                          voxel_batch=8192, step1_batch=256,
                          step1_results_dir=None, work_root="/content/regression_work"):
    support, packages, selected, pending, complete = audit_archives(
        packages_dir, results_dir, support_zip, specie=specie, dataset=dataset,
        model=model, regression=regression, dis_method=dis_method,
        models=models, reps=reps, seed=seed)
    total = len(packages) * len(selected)
    runner.log(f"Step-15 audit: {complete}/{total} complete ZIPs; "
               f"{sum(map(len, pending.values()))} need real fits or receipt repair")
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    delta_dir = Path(results_dir) / "step15_deltas"
    written = []
    for sub, targets in pending.items():
        manifest = packages[sub][1]
        compute = [target for target, missing in targets.items() if missing]
        runner.log(f"{specie}-sub-{sub:02d}: {len(compute)} targets need real maps")
        with tempfile.TemporaryDirectory(prefix=f"complete-15-{sub:02d}-", dir=work_root) as temp:
            root = Path(temp)
            if compute:
                runner.run_regression(
                    packages_dir, delta_dir, support_zip, specie=specie,
                    dataset=dataset, model=model, dis_method=dis_method,
                    models=compute, participants=[sub], regression_model=regression,
                    steps=[15], step1_results_dir=step1_results_dir, reps=reps,
                    reps_group=reps_group, seed=seed, work_root=root / "work",
                    device=device, voxel_batch=voxel_batch, step1_batch=step1_batch)
            for target in targets:
                original = Path(results_dir) / f"result_regression_{target}_{specie}-sub-{sub:02d}.zip"
                delta = delta_dir / original.name if targets[target] else None
                if delta and not delta.is_file():
                    raise FileNotFoundError(f"Step-15 delta absent: {delta}")
                local_original = root / "original.zip"
                local_merged = root / "merged.zip"
                runner.copy_with_progress(original, local_original)
                _merge(local_original, delta, local_merged, manifest, regression, target, reps)
                partial = original.with_suffix(".zip.part")
                runner.copy_with_progress(local_merged, partial)
                if partial.stat().st_size != local_merged.stat().st_size:
                    raise IOError(f"Incomplete Drive copy: {partial}")
                partial.replace(original)
                written.append(str(original))
                runner.log(f"Completed step 15 while preserving 15.4: {original.name}")
                local_original.unlink()
                local_merged.unlink()
    return written
