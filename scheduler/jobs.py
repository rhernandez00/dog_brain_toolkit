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


ALL_STATES = ('pending', 'waiting', 'running', 'completed', 'failed')

# ---------------------------------------------------------------------------
# Job priority: 1 = run first, 3 = run last (default).
# ---------------------------------------------------------------------------
# Priority is deliberately NOT part of the job id: the same analysis queued at a
# different urgency is still the same analysis, so re-scheduling it at priority 1
# must still be seen as a duplicate by create_job() and must still satisfy the
# same dep ids in promote_waiting_jobs().
MIN_PRIORITY = 1
MAX_PRIORITY = 3
DEFAULT_PRIORITY = 3
PRIORITIES = (1, 2, 3)


def normalize_priority(value, default=DEFAULT_PRIORITY):
    """Coerce anything to a valid priority. Missing/garbage -> ``default``.

    Job files written before priorities existed have no 'priority' field, so
    they read back as the default (3, i.e. last).
    """
    try:
        p = int(value)
    except (TypeError, ValueError):
        return default
    return min(max(p, MIN_PRIORITY), MAX_PRIORITY)


def job_priority(job):
    return normalize_priority(job.get('priority'))


def job_files(queue_dir, job_id, state):
    """Return every queue file for ``job_id`` in ``state``: the canonical
    ``{job_id}.json`` plus any ``__dup{N}`` siblings created when the same
    job was (re)scheduled while another instance already existed."""
    d = Path(queue_dir) / state
    files = []
    canonical = d / f"{job_id}.json"
    if canonical.exists():
        files.append(canonical)
    files.extend(sorted(d.glob(f"{job_id}__dup*.json")))
    return files


def create_job(queue_dir, job):
    """Write a new job JSON to the appropriate state folder.

    Job IDs are not required to be unique in the queue. If a job with the
    same ID already exists in *any* state -- pending/waiting/running (a
    duplicate still in flight) or completed/failed (a re-run request) -- the
    new job is written under a disambiguated ``__dup{N}`` filename instead of
    being refused, and is flagged with ``shuffle_participants=True`` so
    ``searchlight.py`` walks participants/permutations in a different order
    than the other instance. This doesn't guard against write races by
    itself -- rsa_utils.py's own per-output-file temp-lock (e.g.
    ``calculate_cross_participant_similarity``) already makes concurrent
    writes to the same output path safe by having the later writer detect
    the other one's temp/finished file and skip it; shuffling just makes two
    concurrent instances diverge instead of racing file-by-file in lockstep.
    """
    queue_dir = Path(queue_dir)
    job = dict(job)
    job['priority'] = job_priority(job)
    job_id = job['job_id']
    existing = [f for state in ALL_STATES for f in job_files(queue_dir, job_id, state)]
    if existing:
        job['shuffle_participants'] = True
        i = 2
        while any((queue_dir / state / f"{job_id}__dup{i}.json").exists()
                  for state in ALL_STATES):
            i += 1
        fname = f"{job_id}__dup{i}.json"
        print(f"  [dup] {job_id} already in the queue ({len(existing)} record(s)); "
              f"scheduling another instance as {fname} (--shuffle_participants)")
    else:
        fname = f"{job_id}.json"
    job['created_at'] = datetime.utcnow().isoformat()
    dest = queue_dir / job['status'] / fname
    save_job(dest, job)
    return True


def _claim_path(queue_dir, path, machine):
    """Try to move one pending job to running/. Returns (path, job) or None."""
    target = queue_dir / 'running' / path.name
    try:
        path.rename(target)
    except (FileNotFoundError, PermissionError):
        return None  # another worker grabbed it first
    job = load_job(target)
    job['status'] = 'running'
    job['started_at'] = datetime.utcnow().isoformat()
    job['machine'] = machine
    save_job(target, job)
    return target, job


def claim_job(queue_dir):
    """Atomically claim one pending job, highest priority first.

    Jobs are ordered by ``priority`` (1 before 2 before 3; a job file without
    the field counts as 3) and, within a priority, by filename — the same
    deterministic order the queue used before priorities existed.

    Deciding the order means reading the pending job files, which is not free on
    the shared network disk, so the scan stops at the first ``MIN_PRIORITY`` job
    it sees: nothing can outrank it. Only when no priority-1 job exists does the
    whole pending folder get read.

    Returns (path, job) or (None, None).
    """
    queue_dir = Path(queue_dir)
    machine = socket.gethostname()
    candidates = []
    for path in sorted((queue_dir / 'pending').glob('*.json')):
        try:
            prio = job_priority(load_job(path))
        except (OSError, ValueError):
            # Unreadable/half-written file (or one another worker just claimed):
            # treat as lowest priority and let the rename attempt decide.
            prio = MAX_PRIORITY
        if prio <= MIN_PRIORITY:
            claimed = _claim_path(queue_dir, path, machine)
            if claimed:
                return claimed
            continue  # lost the race — keep looking
        candidates.append((prio, path.name, path))

    for _prio, _name, path in sorted(candidates, key=lambda t: (t[0], t[1])):
        claimed = _claim_path(queue_dir, path, machine)
        if claimed:
            return claimed
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
    """Move any waiting job whose dependencies are all completed to pending.

    Completed IDs are read from each file's own ``job_id`` field rather than
    its filename stem, so a dependency counts as satisfied if *any* instance
    of it completed -- including one stored under a ``__dup{N}`` filename
    (see ``create_job``) because the canonical attempt failed and a
    duplicate instance succeeded instead.
    """
    queue_dir = Path(queue_dir)
    completed_ids = set()
    for p in (queue_dir / 'completed').glob('*.json'):
        try:
            completed_ids.add(load_job(p).get('job_id', p.stem))
        except Exception:
            completed_ids.add(p.stem)
    for path in sorted((queue_dir / 'waiting').glob('*.json')):
        job = load_job(path)
        if all(dep in completed_ids for dep in job.get('deps', [])):
            target = queue_dir / 'pending' / path.name
            path.rename(target)
            job['status'] = 'pending'
            save_job(target, job)
            print(f"  [promoted] {job['job_id']} -> pending")
