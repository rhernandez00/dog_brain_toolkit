import json
import socket
from datetime import datetime
from pathlib import Path


def load_job(path):
    # utf-8-sig transparently strips a UTF-8 BOM if one is present (some files
    # get written with a BOM on Windows), and is a no-op otherwise.
    with open(path, 'r', encoding='utf-8-sig') as f:
        return json.load(f)


def save_job(path, job):
    # Write plain UTF-8 (no BOM) so the Linux worker's json.load never chokes.
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(job, f, indent=2)


def list_jobs_in_state(queue_dir, state):
    return sorted(Path(queue_dir).glob(f"{state}/*.json"))


def create_job(queue_dir, job):
    """Write a new job JSON to the appropriate state folder.
    Returns False (and prints a notice) if the job ID already exists anywhere."""
    queue_dir = Path(queue_dir)
    fname = f"{job['job_id']}.json"
    for state in ('pending', 'waiting', 'running', 'completed', 'failed'):
        if (queue_dir / state / fname).exists():
            print(f"  [skip] {job['job_id']} already exists in {state}/")
            return False
    job = dict(job)
    job['created_at'] = datetime.utcnow().isoformat()
    dest = queue_dir / job['status'] / fname
    save_job(dest, job)
    return True


def claim_job(queue_dir):
    """Atomically claim one pending job. Returns (path, job) or (None, None)."""
    queue_dir = Path(queue_dir)
    machine = socket.gethostname()
    for path in sorted((queue_dir / 'pending').glob('*.json')):
        target = queue_dir / 'running' / path.name
        try:
            path.rename(target)
        except (FileNotFoundError, PermissionError):
            continue  # another worker grabbed it first
        job = load_job(target)
        job['status'] = 'running'
        job['started_at'] = datetime.utcnow().isoformat()
        job['machine'] = machine
        save_job(target, job)
        return target, job
    return None, None


def complete_job(queue_dir, running_path, job):
    queue_dir = Path(queue_dir)
    running_path = Path(running_path)
    job = dict(job)
    job['status'] = 'completed'
    job['completed_at'] = datetime.utcnow().isoformat()
    dest = queue_dir / 'completed' / running_path.name
    running_path.rename(dest)
    save_job(dest, job)
    promote_waiting_jobs(queue_dir)


def fail_job(queue_dir, running_path, job, error_msg):
    queue_dir = Path(queue_dir)
    running_path = Path(running_path)
    job = dict(job)
    job['status'] = 'failed'
    job['completed_at'] = datetime.utcnow().isoformat()
    job['error'] = error_msg
    dest = queue_dir / 'failed' / running_path.name
    running_path.rename(dest)
    save_job(dest, job)


def promote_waiting_jobs(queue_dir):
    """Move any waiting job whose dependencies are all completed to pending."""
    queue_dir = Path(queue_dir)
    completed_ids = {p.stem for p in (queue_dir / 'completed').glob('*.json')}
    for path in sorted((queue_dir / 'waiting').glob('*.json')):
        job = load_job(path)
        if all(dep in completed_ids for dep in job.get('deps', [])):
            target = queue_dir / 'pending' / path.name
            path.rename(target)
            job['status'] = 'pending'
            save_job(target, job)
            print(f"  [promoted] {job['job_id']} -> pending")
