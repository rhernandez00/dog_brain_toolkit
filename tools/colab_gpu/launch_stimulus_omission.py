"""Start monitored omission analysis using the packages already on Drive.

Usable in a Colab notebook or with colab exec -s rsa-omission -f FILE.
Google sign-in/mounting remains an interactive user action.
"""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

FILES = ('gpu_rsa.py', 'gpu_stimulus_omission.py', 'run_colab_stimulus_omission.py',
         'stimulus_contribution.py', 'supervise_stimulus_omission.py', 'launch_stimulus_omission.py')


def launch(drive_root='/content/drive/MyDrive/rsa_colab',
           output_root='/content/drive/MyDrive/rsa_colab/omission_analysis',
           work='/content/omission_supervisor', maximum_workers=2,
           step1_batch=128, voxel_batch=256, code_override=None, allowed_missing_runs=(), adopt_workers=None):
    missing = [name for name in ('numpy', 'scipy', 'nibabel', 'psutil')
               if importlib.util.find_spec(name) is None]
    if missing:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *missing])
    import torch
    import psutil
    if not torch.cuda.is_available():
        raise RuntimeError('This run requires a GPU runtime. Connect a GPU before launching.')
    drive_root, output_root, work = Path(drive_root), Path(output_root), Path(work)
    if not drive_root.is_dir():
        raise RuntimeError('Google Drive is not mounted. Complete drive.mount("/content/drive") in Colab, '
                           'or colab drivemount -s rsa-omission in your terminal, then rerun.')
    for specie in ('D', 'H'):
        folder = drive_root / f'omission_{specie}'
        if not folder.is_dir():
            raise FileNotFoundError(f'Uploaded package folder not found: {folder}')
    work.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    pid_file = output_root / 'supervisor_pid.json'
    if pid_file.is_file():
        try:
            previous = json.loads(pid_file.read_text())
            process = psutil.Process(previous['pid'])
            if (process.is_running() and abs(process.create_time() - previous['created']) < 1
                    and any('supervise_stimulus_omission' in part for part in process.cmdline())):
                print(f'Existing supervisor is active: PID {process.pid}', flush=True)
                return previous
        except (OSError, ValueError, KeyError, psutil.Error):
            pass
    source = Path(code_override) if code_override else drive_root / 'omission_code'
    if not all((source / name).is_file() for name in FILES):
        source = Path(__file__).resolve().parent
    if not all((source / name).is_file() for name in FILES):
        raise FileNotFoundError('Upload the current omission_code support folder; the data ZIPs do not need rebuilding.')
    runtime = work / 'code'
    runtime.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copyfile(source / name, runtime / name)
    command = [sys.executable, '-u', str(runtime / 'supervise_stimulus_omission.py'),
               '--drive-root', str(drive_root), '--output-root', str(output_root),
               '--work', str(work / 'workers'), '--maximum-workers', str(maximum_workers),
               '--step1-batch', str(step1_batch), '--voxel-batch', str(voxel_batch)]
    for item in allowed_missing_runs:
        command.extend(['--allow-missing-run', item])
    if adopt_workers:
        command.extend(['--adopt-workers', str(adopt_workers)])
    env = dict(os.environ)
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = '2'
    with (output_root / 'supervisor.log').open('a', buffering=1) as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                   start_new_session=(os.name != 'nt'))
    receipt = dict(pid=process.pid, created=psutil.Process(process.pid).create_time(),
                   output_root=str(output_root), status=str(output_root / 'status.json'),
                   resources=str(output_root / 'resources.jsonl'), log=str(output_root / 'supervisor.log'))
    temporary = Path(str(pid_file) + '.part')
    temporary.write_text(json.dumps(receipt, indent=2))
    os.replace(temporary, pid_file)
    print(json.dumps(receipt, indent=2), flush=True)
    return receipt


if __name__ == '__main__':
    launch(**globals().get('OMISSION_JOB', {}))
