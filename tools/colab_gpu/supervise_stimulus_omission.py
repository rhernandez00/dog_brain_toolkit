"""Bounded, resource-aware OS workers on one Colab VM; no extra VMs."""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil


def snapshot(work):
    ram = psutil.virtual_memory()
    result = dict(time=time.time(), cpus=os.cpu_count(), cpu_percent=psutil.cpu_percent(),
                  ram_available=ram.available, ram_total=ram.total,
                  disk_free=shutil.disk_usage(work).free, gpu_utilization=None,
                  vram_free=None, vram_total=None)
    try:
        text = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.free,memory.total',
            '--format=csv,noheader,nounits'], text=True, timeout=8)
        utilization, free, total = map(float, text.strip().splitlines()[0].split(','))
        result.update(gpu_utilization=utilization, vram_free=int(free * 2**20), vram_total=int(total * 2**20))
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return result


def can_admit(history, active, maximum, job, oldest_age=0):
    """Require sustained headroom; low utilization alone is insufficient."""
    if active >= maximum or not history:
        return False
    current = history[-1]
    if current['ram_available'] < job['ram'] + 4 * 2**30 or current['disk_free'] < job['disk'] + 20 * 2**30:
        return False
    if current['vram_free'] is not None and current['vram_free'] < job['vram']:
        return False
    if job['vram'] and (current['vram_free'] is None or current['gpu_utilization'] is None):
        return False
    if not active:
        return True
    if len(history) < 3 or oldest_age < 45:
        return False
    for item in list(history)[-3:]:
        if item['cpu_percent'] > 70 or item['ram_available'] < job['ram'] + 4 * 2**30:
            return False
        if job['vram'] and item['gpu_utilization'] is not None and item['gpu_utilization'] > 65:
            return False
        if item['vram_free'] is not None and item['vram_free'] < job['vram'] * 1.25:
            return False
    return True


def next_admissible(pending, history, active, maximum, oldest_age=0):
    """A queued GPU job must not block a runnable CPU job behind it."""
    return next((job for job in pending
                 if can_admit(history, active, maximum, job, oldest_age)), None)


class AdoptedWorker:
    """Monitor an existing worker without restarting its in-flight computation."""
    def __init__(self, pid, started, verify_completed):
        self.process = psutil.Process(pid)
        if abs(self.process.create_time() - started) > 1:
            raise ValueError(f'Worker PID {pid} has been reused.')
        self.pid = pid
        self.verify_completed = verify_completed
        self.code = None

    def poll(self):
        if self.code is not None:
            return self.code
        try:
            if self.process.is_running() and self.process.status() != psutil.STATUS_ZOMBIE:
                return None
        except psutil.NoSuchProcess:
            pass
        # A non-parent cannot waitpid() for the exit code. Require all exact
        # expected checkpoint receipts before declaring the job successful.
        self.code = 0 if self.verify_completed() else 1
        return self.code

    def terminate(self):
        try:
            self.process.terminate()
        except psutil.NoSuchProcess:
            pass

    def kill(self):
        try:
            self.process.kill()
        except psutil.NoSuchProcess:
            pass

    def wait(self, timeout=None):
        started = time.monotonic()
        while self.poll() is None:
            if timeout is not None and time.monotonic() - started >= timeout:
                raise subprocess.TimeoutExpired(str(self.pid), timeout)
            time.sleep(0.1)
        return self.code


def cohort_coverage(packages, specie, allowed_missing_runs=()):
    """Authorize only the exact absent participant/run pairs supplied by the user."""
    runs = sorted({int(e['run_N']) for package in packages.values()
                   for e in package['manifest']['runs']})
    approved = set()
    for item in allowed_missing_runs:
        code, participant, run = item.split(':')
        if code == specie:
            approved.add((int(participant), int(run)))
    actual = {(sub, run) for sub, package in packages.items() for run in runs
              if run not in {int(e['run_N']) for e in package['manifest']['runs']}}
    if actual != approved:
        raise ValueError(f'{specie}: absent participant/run pairs {sorted(actual)} differ from '
                         f'approved pairs {sorted(approved)}. No additional participants will be excluded.')
    included = {str(run): sorted(sub for sub in packages if (sub, run) not in actual) for run in runs}
    if any(len(subs) < 2 for subs in included.values()):
        raise ValueError('Every group comparison needs at least two participants.')
    return included, min(len(subs) / len(packages) for subs in included.values())


def estimate(manifest, voxels, reference_shape, batch):
    units, conditions, pairs = len(manifest['runs']), len(manifest['categories']), len(manifest['pairs'])
    # A bounding cube intentionally overestimates the spherical neighborhood.
    features = min(voxels, (2 * int(manifest['radius']) + 1) ** 3)
    beta = units * len(manifest['stim_types']) * voxels * 8
    category = units * conditions * voxels * 8
    vram = batch * (12 * features**2 + 4 * units * conditions * features
                    + 3 * units * pairs * features) * 8 + 2 * category + 2**30
    ram = beta + 5 * category + voxels * features * 12 + 3 * 2**30
    disk = beta + units * len(manifest['stim_types']) * int(__import__('math').prod(reference_shape)) * 8
    return dict(ram=ram, disk=disk, vram=vram)


def write_status(path, value):
    temporary = Path(str(path) + '.part')
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, path)


def run(cohorts, output_root, work='/content/omission_supervisor', maximum_workers=2,
        sample_seconds=15, step1_batch=128, voxel_batch=256, comparison_batch=4096,
        min_percentage_available=1.0, device='cuda', allowed_missing_runs=(), adopt_workers=None):
    import run_colab_stimulus_omission as runner
    if not 1 <= maximum_workers <= 4 or sample_seconds < 1:
        raise ValueError('maximum_workers must be 1..4; sample_seconds must be positive.')
    output_root, work = Path(output_root), Path(work)
    output_root.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    jobs, groups, required = [], [], {}
    policy = dict(allowed_missing_runs=list(allowed_missing_runs), participants_by_run={})
    runner_file = Path(runner.__file__).resolve()
    for specie, packages_dir in cohorts.items():
        packages, reference, mask = runner.discover_packages(packages_dir, specie, 'EmoC', 'basic-block')
        included_by_run, coverage_threshold = cohort_coverage(packages, specie, allowed_missing_runs)
        policy['participants_by_run'][specie] = included_by_run
        out = output_root / f'results_omission_{specie}'
        out.mkdir(parents=True, exist_ok=True)
        required[specie] = set()
        for sub, package in packages.items():
            name = f'{specie}-sub-{sub:02d}'
            required[specie].add(name)
            command = [sys.executable, '-u', str(runner_file), '--packages', str(packages_dir),
                       '--out', str(out), '--specie', specie, '--participants', str(sub),
                       '--steps', '1', '--step1-worker']
            # A measured full dog omission is faster on four CPU cores than
            # on the T4; this also leaves the GPU available for the human data.
            worker_device = 'cpu' if specie == 'D' and int(mask.sum()) <= 20000 else device
            requirements = estimate(package['manifest'], int(mask.sum()), reference.shape, step1_batch)
            if worker_device == 'cpu':
                requirements['vram'] = 0
            jobs.append(dict(name=name, specie=specie, command=command, device=worker_device,
                             package=package, output=out,
                             **requirements))
        command = [sys.executable, '-u', str(runner_file), '--packages', str(packages_dir),
                   '--out', str(out), '--specie', specie, '--steps', '11.2']
        group_disk = sum(len(p['manifest']['runs']) for p in packages.values()) * int(mask.sum()) * len(next(iter(packages.values()))['manifest']['pairs']) * 8
        groups.append(dict(name=f'{specie}-step11.2', specie=specie, command=command,
                           device=device, minimum_percentage=min(min_percentage_available, coverage_threshold),
                           ram=4 * 2**30, disk=group_disk, vram=2 * 2**30))
    write_status(output_root / 'analysis_policy.json', policy)
    # Interleave cohorts; each worker has private scratch and a private log.
    jobs.sort(key=lambda job: (int(job['name'].split('-')[-1]), job['specie']))
    pending = deque(jobs)
    active, finished, failed = {}, set(), {}
    history = deque(maxlen=6)
    maximum = maximum_workers
    config = output_root / 'supervisor_config.json'
    if not config.exists():
        write_status(config, dict(maximum_workers=maximum_workers))
    status_path = output_root / 'status.json'
    log_path = output_root / 'resources.jsonl'
    logs = output_root / 'logs'
    logs.mkdir(exist_ok=True)
    if adopt_workers:
        adopted = json.loads(Path(adopt_workers).read_text())
        lookup = {job['name']: job for job in jobs}
        for name, item in adopted.items():
            if name not in lookup:
                raise ValueError(f'Cannot adopt unknown worker {name}.')
            job = lookup[name]
            expected = runner.records_for(job['package'], job['output'])
            def verify(records=expected):
                return all(runner.complete(r['path'], r['receipt'], r['members']) for r in records)
            try:
                process = psutil.Process(item['pid'])
                arguments = process.cmdline()
                if not any('run_colab_stimulus_omission.py' in part for part in arguments):
                    raise ValueError(f'PID {item["pid"]} is not an omission worker.')
                worker = AdoptedWorker(item['pid'], item['started'], verify)
            except psutil.NoSuchProcess:
                if verify():
                    finished.add(name)
                    pending.remove(job)
                    continue
                raise RuntimeError(f'Worker {name} disappeared without completing its checkpoints.')
            job['device'] = item['device']
            active[name] = dict(process=worker, log=(logs / f'{name}.log').open('a'),
                                started=item['started'], spec=job)
            pending.remove(job)
            print(f'Adopted {name} on {job["device"]}: PID {worker.pid}', flush=True)
    print(f'Supervising {len(jobs)} participant jobs and {len(groups)} group jobs.', flush=True)
    try:
        while pending or active or groups:
            for name, job in list(active.items()):
                code = job['process'].poll()
                if code is None:
                    continue
                job['log'].close()
                del active[name]
                if code == 0:
                    finished.add(name)
                    print(f'Completed {name}', flush=True)
                else:
                    tail = (logs / f'{name}.log').read_text(errors='replace')[-12000:].lower()
                    if ('cuda out of memory' in tail or 'cuda error: out of memory' in tail) and not job['spec'].get('retried'):
                        retry = dict(job['spec'], retried=True)
                        pending.appendleft(retry)
                        maximum_workers = maximum = 1
                        step1_batch = max(8, step1_batch // 2)
                        voxel_batch = max(16, voxel_batch // 2)
                        write_status(config, dict(maximum_workers=1))
                        history.clear()
                        print(f'Retrying {name} serially with smaller batches after CUDA memory pressure.', flush=True)
                        continue
                    failed[name] = dict(exit_code=code, log=str(logs / f'{name}.log'))
                    print(f'FAILED {name}; see {logs / (name + ".log")}', flush=True)
            if failed:
                raise RuntimeError(f'Analysis worker failed: {failed}')
            for group in list(groups):
                if required[group['specie']].issubset(finished):
                    pending.appendleft(group)
                    groups.remove(group)
            reading = snapshot(work)
            history.append(reading)
            with log_path.open('a') as handle:
                handle.write(json.dumps(dict(reading, active=list(active))) + '\n')
            try:
                maximum = max(1, min(maximum_workers, int(json.loads(config.read_text()).get('maximum_workers', maximum_workers))))
            except (OSError, ValueError, TypeError):
                pass
            if pending:
                oldest_age = max((time.time() - job['started'] for job in active.values()), default=0)
                candidate = next_admissible(pending, history, len(active), maximum, oldest_age)
                if candidate is not None:
                    job = candidate
                    pending.remove(job)
                    scratch = work / job['name']
                    scratch.mkdir(parents=True, exist_ok=True)
                    command = job['command'] + ['--work', str(scratch), '--device', job['device'],
                        '--step1-batch', str(step1_batch), '--voxel-batch', str(voxel_batch),
                        '--comparison-batch', str(comparison_batch),
                        '--min-percentage-available', str(job.get('minimum_percentage', min_percentage_available))]
                    env = dict(os.environ)
                    for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
                        env[variable] = str(max(1, min(4 if job['device'] == 'cpu' else 2,
                                                      (os.cpu_count() or 2) // maximum_workers)))
                    log = (logs / (job['name'] + '.log')).open('a', buffering=1)
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                    active[job['name']] = dict(process=process, log=log, started=time.time(), spec=job)
                    history.clear()  # Observe the new load before adding another worker.
                    print(f'Started {job["name"]} on {job["device"]}: PID {process.pid}; active={len(active)}', flush=True)
                elif not active:
                    raise RuntimeError(f'Insufficient RAM/VRAM/disk to start pending jobs: {reading}')
            status = dict(state='running', updated=time.time(), active={name:dict(pid=j['process'].pid, started=j['started'],
                                                                              device=j['spec']['device'])
                          for name,j in active.items()}, finished=sorted(finished), failed=failed,
                          pending=[j['name'] for j in pending], waiting_groups=[g['name'] for g in groups], resources=reading)
            write_status(status_path, status)
            print(f'Workers={list(active)} completed={len(finished)} CPU={reading["cpu_percent"]:.0f}% '
                  f'GPU={reading["gpu_utilization"]}% RAM-free={reading["ram_available"]/2**30:.1f}GB', flush=True)
            if pending or active or groups:
                time.sleep(sample_seconds)
        write_status(status_path, dict(state='complete', updated=time.time(), finished=sorted(finished), failed={}))
    except BaseException as error:
        for job in active.values():
            job['process'].terminate()
        for job in active.values():
            try:
                job['process'].wait(timeout=20)
            except subprocess.TimeoutExpired:
                job['process'].kill()
                job['process'].wait()
            job['log'].close()
        write_status(status_path, dict(state='failed', updated=time.time(), error=str(error),
                                      finished=sorted(finished), failed=failed))
        raise
    return sorted(finished)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--drive-root', default='/content/drive/MyDrive/rsa_colab')
    parser.add_argument('--output-root', default='/content/drive/MyDrive/rsa_colab/omission_analysis')
    parser.add_argument('--work', default='/content/omission_supervisor')
    parser.add_argument('--maximum-workers', type=int, default=2)
    parser.add_argument('--step1-batch', type=int, default=128)
    parser.add_argument('--voxel-batch', type=int, default=256)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--allow-missing-run', action='append', default=[], metavar='SPECIE:PARTICIPANT:RUN')
    parser.add_argument('--adopt-workers', help='JSON status of existing workers to preserve during supervisor replacement')
    args = parser.parse_args()
    run({s: str(Path(args.drive_root) / f'omission_{s}') for s in ('D', 'H')}, args.output_root,
        work=args.work, maximum_workers=args.maximum_workers, step1_batch=args.step1_batch,
        voxel_batch=args.voxel_batch, device=args.device, allowed_missing_runs=args.allow_missing_run,
        adopt_workers=args.adopt_workers)
