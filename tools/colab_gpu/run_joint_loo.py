"""Run one extracted cohort and checkpoint portable result archives on Drive."""
import argparse
from pathlib import Path
import zipfile
from loo_joint import Analysis, digest, publish, read_json, signature, write_json


def checkpoint(path, root, files):
    def write(tmp):
        with zipfile.ZipFile(tmp, 'w', allowZip64=True) as z:
            for p in map(Path, files):
                z.write(p, p.relative_to(root).as_posix())
        with zipfile.ZipFile(tmp) as z:
            if z.testzip() is not None:
                raise IOError(f'Archive verification failed: {path}')
    publish(path, write)
    write_json(path.with_suffix('.json'), dict(sha256=digest(path)))


def completed(path):
    receipt = path.with_suffix('.json')
    if not path.is_file() or not receipt.is_file():
        return False
    try:
        expected = read_json(receipt)['sha256']
    except (ValueError, KeyError):
        return False
    return digest(path) == expected


def run(package, output, reps, seed, device, batch=20):
    pkg = Path(package)
    meta = read_json(pkg / 'manifest.json')
    for name, expected in meta['hashes'].items():
        if digest(pkg / name) != expected:
            raise ValueError(f'Package fingerprint mismatch: {name}')
    if batch < 1 or reps < 2:
        raise ValueError('Need positive checkpoint batch and at least two draws.')
    root = pkg / 'data'
    # Isolate imports across package, code, seed and draw count changes.
    result = Path(output) / (meta['specie'] + '_' + signature([meta, reps, seed, batch])[:20])
    for cls in meta['classes']:
        a = Analysis(root, meta['dataset'], {}, meta['participants'], meta['specie'],
                     root / meta['dataset'] / 'ROI' / meta['specie'] / (meta['mask_type']+'.nii.gz'),
                     meta['model'], meta['task'], meta['radius'], cls,
                     rsa_method=meta['rsa_method'], mask_type=meta['mask_type'])
        maps = a.mappings(reps, seed, prepare=True, replace=True)
        base = f'result_loo_{meta["specie"]}_{cls}'
        path = result / (base + '_observed.zip')
        if not completed(path):
            a.observed(device)
            files = list(a.paths.values()) + [str(a.group)+suffix for suffix in
                                              ('_mean.nii.gz','_std.nii.gz','_mean.json')]
            checkpoint(path, root, files + [a.plan_path])
        for start in range(0, reps, batch):
            path = result / f'{base}_{start:05d}.zip'
            if completed(path):
                continue
            files = []
            for i in range(start, min(reps, start+batch)):
                a.draw(i, maps[i], device, replace=True)
                files.extend(a.draw_paths(i))
            checkpoint(path, root, files)
            for p in files:
                p.unlink()
            print(f'{cls}: {min(reps,start+batch)}/{reps} draws archived', flush=True)
        del a
    print(f'VERIFIED_RESULT_DIR={result.name}', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package')
    parser.add_argument('output')
    parser.add_argument('--reps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda', choices=['cuda','cpu'])
    a = parser.parse_args()
    run(a.package, a.output, a.reps, a.seed, a.device)

