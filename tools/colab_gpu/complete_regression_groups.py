"""Add step 15.3 real means to saved step 15.5 group ZIPs.

Reads only real beta maps from the published participant ZIPs. Original group
archives remain untouched; combined archives are written to a separate folder.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

import run_colab_regression as runner


def numpy_group_moments(bank, selections, device="cpu", voxel_batch=8192, group_batch=16):
    """One-draw CPU equivalent of the Torch population mean/std reducer."""
    if device != "cpu" or len(selections) != 1:
        raise ValueError("NumPy reducer is only for one real CPU group draw")
    for v in range(0, bank.shape[1], voxel_batch):
        values = np.asarray(bank[selections[0], v:v + voxel_batch], dtype=np.float64)
        yield 0, v, values.mean(axis=0, keepdims=True), values.std(axis=0, keepdims=True)


def isolated_save_vector(path, data, mask, dtype=np.float32, outside=0):
    """Save via a Torch-free process to avoid the local Conda OpenMP collision."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    script = (
        "import sys,numpy as np,nibabel as nib; "
        "mask=nib.load(sys.argv[2]); "
        "volume=np.full(mask.shape,float(sys.argv[4]),dtype=np.dtype(sys.argv[5])); "
        "volume[mask.get_fdata()>0]=np.load(sys.argv[1]); "
        "nib.save(nib.Nifti1Image(volume,mask.affine),sys.argv[3])"
    )
    with tempfile.TemporaryDirectory(prefix="save-vector-") as temp:
        vector = Path(temp) / "vector.npy"
        np.save(vector, data)
        subprocess.run([sys.executable, "-c", script, str(vector), mask.get_filename(),
                        str(path), str(outside), np.dtype(dtype).name], check=True)


def complete_groups(base, *, specie="D", targets=None, device="cpu", output_dir=None,
                    group_source_dir=None):
    base = Path(base)
    dataset, model, regression = "EmoB", "basic-block", "visual_3"
    source_dir = base / f"results_regression_{dataset}"
    group_source_dir = (Path(group_source_dir) if group_source_dir else
                        base / (f"results_regression_group_{dataset}" if specie == "H"
                                else f"results_regression_{dataset}"))
    output_dir = Path(output_dir) if output_dir else base / f"results_regression_{dataset}_inference_ready"
    support_zip = base / f"regression_support_{dataset}_{regression}.zip"
    with zipfile.ZipFile(support_zip) as archive:
        support = json.loads(archive.read("regression_manifest.json"))
    packages = runner.discover_packages(
        base / f"pkg_{dataset}", specie, dataset,
        participants=support["participants_by_species"][specie],
        model=model, dis_method="correlation",
    )
    if set(packages) != set(support["participants_by_species"][specie]):
        raise ValueError("Participant packages do not match the support manifest")
    reference = next(iter(packages.values()))[1]
    candidates = sorted(group_source_dir.glob(f"result_regression_group_*_{specie}.zip"))
    if targets is not None:
        candidates = [p for p in candidates if p.name[len("result_regression_group_"):-len(f"_{specie}.zip")] in targets]
    if not candidates:
        raise FileNotFoundError(f"No step 15.5 group ZIPs for {specie} in {group_source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for source in candidates:
        target = source.name[len("result_regression_group_"):-len(f"_{specie}.zip")]
        output = output_dir / source.name
        if output.resolve() == source.resolve():
            raise ValueError("Output directory must differ from the group source directory")
        if output.exists():
            runner.log(f"Already exists, skipping {output}")
            continue
        runner.log(f"Step 15.3 from saved real betas: {target}")
        with tempfile.TemporaryDirectory(prefix="real-group-") as temp:
            root = Path(temp)
            runner.extract_safe(support_zip, root, prefix="data/")
            mask, _ = runner.gpu.gpu_rsa.load_reference_mask(str(root / "data"), reference)
            real_members = []
            for sub, (_, manifest) in packages.items():
                participant = source_dir / f"result_regression_{target}_{specie}-sub-{sub:02d}.zip"
                if not participant.is_file():
                    raise FileNotFoundError(participant)
                receipt = runner._receipt_relative(manifest, regression, target).as_posix()
                with zipfile.ZipFile(participant) as archive:
                    names = set(archive.namelist())
                    if receipt not in names or 15 not in json.loads(archive.read(receipt)).get("steps", []):
                        raise ValueError(f"Missing completed step 15 in {participant}")
                    for entry in manifest["runs"]:
                        folder = runner.gpu.run_folder(manifest, regression, target, entry, False)
                        stem, suffix = runner.gpu.map_stem(manifest, None)
                        member = (folder / f"{stem}_beta_map{suffix}.nii.gz").as_posix()
                        if member not in names:
                            raise FileNotFoundError(f"{participant}: {member}")
                        destination = root / "data" / member
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(member) as src, destination.open("wb") as dst:
                            shutil.copyfileobj(src, dst)
                        real_members.append((participant.name, member))
            original_reducer = runner.gpu.group_moments
            original_saver = runner.gpu.save_vector
            try:
                if device == "cpu":
                    runner.gpu.group_moments = numpy_group_moments
                    runner.gpu.save_vector = isolated_save_vector
                runner._group_target(root, packages, target, regression, mask, {15.3},
                                     100, 1000, 42, device, 8192, 16)
            finally:
                runner.gpu.group_moments = original_reducer
                runner.gpu.save_vector = original_saver
            relative = Path(dataset) / "results/RSA_regression" / model / regression / target / "mean"
            products = sorted((root / "data" / relative).glob("*"))
            if len(products) != 3:
                raise ValueError(f"Expected step 15.3 mean/std/JSON for {target}; got {products}")
            group_receipt = runner._receipt_relative(reference, regression, target, True).as_posix()
            temporary = output.with_suffix(".zip.tmp")
            try:
                with zipfile.ZipFile(source) as original, zipfile.ZipFile(temporary, "w", allowZip64=True) as combined:
                    if group_receipt not in original.namelist():
                        raise ValueError(f"No group receipt in {source}")
                    receipt_record = json.loads(original.read(group_receipt))
                    if 15.5 not in receipt_record.get("steps", []):
                        raise ValueError(f"Step 15.5 not complete in {source}")
                    for info in original.infolist():
                        if info.filename != group_receipt:
                            combined.writestr(info, original.read(info))
                    for product in products:
                        combined.write(product, product.relative_to(root / "data").as_posix(),
                                       compress_type=zipfile.ZIP_STORED)
                    receipt_record["steps"] = [15.3, 15.5]
                    receipt_record["real_group_source"] = "saved step-15 participant beta maps"
                    receipt_record["real_group_input_maps"] = len(real_members)
                    combined.writestr(group_receipt, json.dumps(receipt_record, indent=2))
                with zipfile.ZipFile(temporary) as check:
                    if check.testzip() is not None:
                        raise ValueError(f"Corrupt combined ZIP: {temporary}")
                temporary.replace(output)
            finally:
                if temporary.exists():
                    temporary.unlink()
        runner.log(f"Ready: {output}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", help="rsa_colab Drive folder")
    parser.add_argument("--specie", choices=["D", "H"], default="D")
    parser.add_argument("--target", action="append", dest="targets")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--out", help="Separate folder for combined group ZIPs")
    parser.add_argument("--group-source", help="Folder containing saved step-15.5 group ZIPs")
    args = parser.parse_args()
    print(complete_groups(args.base, specie=args.specie, targets=args.targets,
                          device=args.device, output_dir=args.out,
                          group_source_dir=args.group_source))
