"""Bootstrap the packaged joint LOO runner from mounted Google Drive."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

def main(drive='/content/drive/MyDrive/rsa_colab', device='cuda', reps=1000, seed=0):
    import torch
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Connect a GPU runtime, or explicitly select CPU.')
    drive = Path(drive)
    pkgs = [drive / f'loo_group_{s}.zip' for s in ('D','H')]
    for p in pkgs:
        if not p.is_file():
            raise FileNotFoundError(p)
    missing = [p for p in ('scipy','nibabel') if importlib.util.find_spec(p) is None]
    if missing:
        subprocess.run([sys.executable,'-m','pip','install',*missing],check=True)
    for p in pkgs:
        with tempfile.TemporaryDirectory(prefix='loo_') as scratch, zipfile.ZipFile(p) as z:
            expected_species = p.stem.rsplit('_', 1)[-1]
            if json.loads(z.read('manifest.json'))['specie'] != expected_species:
                raise ValueError(f'Wrong participant species in {p}')
            z.extractall(scratch)
            script = Path(scratch) / 'code/run_joint_loo.py'
            command = [sys.executable, '-u', str(script), scratch, str(drive / 'loo_results'),
                       '--device', device, '--reps', str(reps), '--seed', str(seed)]
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True) as worker:
                for line in worker.stdout:
                    print(line, end='', flush=True)
                if worker.wait():
                    raise subprocess.CalledProcessError(worker.returncode, command)

if __name__ == '__main__':
    main(**globals().get('LOO_JOB', {}))

