STEP_LABELS = {
    0: "Beta maps",
    1: "Pairwise similarity",
    2: "Model similarity",
    3: "Group similarity map",
    4: "RND permuted model",
    5: "RND group permutations",
    6: "Voxelwise RND distribution",
    7: "Z-maps",
    8: "Cluster size distribution",
    9: "Cluster correction",
    10: "Create tables",
}

# Generic dependency graph; species-specific adjustments applied in build_job_graph
_STEP_DEPS = {
    0:  [],
    1:  [0],     # for dogs; humans skip step 0 (beta maps pre-exist)
    2:  [1],
    3:  [2],
    4:  [1],     # RND uses pairwise maps, not group map
    5:  [4],
    6:  [5],
    7:  [3, 6],  # z-maps need group map AND voxelwise RND distribution
    8:  [7],
    9:  [7, 8],  # cluster correction needs real z-map AND cluster dist
    10: [9],
}


def make_job_id(dataset, model, rsa_model, specie, step, z_threshold, reps, reps_group, rsa_method="kendall", dis_method="mahalanobis", mah_fold="stim-wise", participant=None):
    job_id = (
        f"{dataset}__{model}__{rsa_model}__{specie}"
        f"__step{step:02d}__zt{z_threshold}__r{reps}__rg{reps_group}"
        f"__rsa{rsa_method}__dis{dis_method}__mah{mah_fold}"
    )
    # Per-participant jobs (scheduled from the dashboard for a single missing map)
    # get a __subNN suffix so they never collide with the whole-step job or with
    # each other. participant=None keeps the classic whole-step id unchanged.
    if participant is not None:
        job_id += f"__sub{int(participant):02d}"
    return job_id


def build_job_graph(dataset, model, rsa_model, specie, target_step=10,
                    start_step=2,
                    z_threshold=3.1, reps=100, reps_group=1000, rsa_method="kendall",
                    dis_method="mahalanobis", 
                    mah_fold="stim-wise",
                    replace_rnd_files=False):
    """
    Return job dicts in topological order (leaf steps first) for running
    target_step for the given specie.  Steps below start_step are never
    scheduled (their outputs are assumed to already exist on disk).
    """
    # Collect all needed steps via backwards walk from target_step
    needed = set()
    queue = [target_step]
    while queue:
        step = queue.pop()
        if step in needed:
            continue
        needed.add(step)
        deps = list(_STEP_DEPS.get(step, []))
        if specie == 'H' and step == 1:
            deps = []  # humans have pre-existing beta maps
        queue.extend(deps)

    # Drop steps below the floor; their dependents become roots (pending)
    needed = {s for s in needed if s >= start_step}

    # Build per-step dep lists restricted to needed steps
    adj = {}
    for step in needed:
        deps = list(_STEP_DEPS.get(step, []))
        if specie == 'H' and step == 1:
            deps = []
        adj[step] = [d for d in deps if d in needed]

    # Kahn's topological sort (deterministic: always pick smallest step)
    in_degree = {s: len(adj[s]) for s in needed}
    ready = sorted(s for s in needed if in_degree[s] == 0)
    topo_order = []
    while ready:
        step = ready.pop(0)
        topo_order.append(step)
        for s in sorted(needed):
            if step in adj[s]:
                in_degree[s] -= 1
                if in_degree[s] == 0:
                    ready.append(s)
                    ready.sort()

    jobs = []
    for step in topo_order:
        dep_steps = adj[step]
        dep_ids = [
            make_job_id(dataset, model, rsa_model, specie, d, z_threshold, reps, reps_group, rsa_method, dis_method, mah_fold)
            for d in dep_steps
        ]
        job_id = make_job_id(dataset, model, rsa_model, specie, step, z_threshold, reps, reps_group, rsa_method, dis_method, mah_fold)
        jobs.append({
            "job_id": job_id,
            "dataset": dataset,
            "model": model,
            "rsa_model": rsa_model,
            "specie": specie,
            "step": step,
            "label": STEP_LABELS.get(step, f"Step {step}"),
            "status": "pending" if not dep_ids else "waiting",
            "deps": dep_ids,
            "rsa_method": rsa_method,
            "dis_method": dis_method,
            "mah_fold": mah_fold,
            "z_threshold": z_threshold,
            "reps": reps,
            "reps_group": reps_group,
            "replace_rnd_files": replace_rnd_files,
            "created_at": None,
            "started_at": None,
            "completed_at": None,
            "machine": None,
            "error": None,
        })
    return jobs


def build_single_job(dataset, model, rsa_model, specie, step,
                     z_threshold=3.1, reps=100, reps_group=1000,
                     rsa_method="kendall", dis_method="mahalanobis",
                     mah_fold="stim-wise", participant=None,
                     radius=None, mask_type=None,
                     replace_file=False, replace_rnd_files=False):
    """Build a single, *independent* job dict (no dependencies, status=pending).

    Used by the dashboard's "schedule missing" / per-map buttons: the user has
    looked at the disk and wants to (re)compute one specific map, so the job is
    created ready-to-run with an empty ``deps`` list. ``participant`` (an int)
    scopes the job to one subject via searchlight's ``--participants_forced``;
    ``participant=None`` schedules the whole step (the single group map for the
    group steps 3/5/6/7/8/9/10).

    The extra ``radius`` / ``mask_type`` / ``replace_file`` fields are honoured
    by ``run_jobs.build_command`` (they are absent from classic scheduler jobs,
    which read them with ``.get()`` and fall back to searchlight's defaults).
    """
    job_id = make_job_id(dataset, model, rsa_model, specie, step, z_threshold,
                         reps, reps_group, rsa_method, dis_method, mah_fold,
                         participant=participant)
    label = STEP_LABELS.get(step, f"Step {step}")
    if participant is not None:
        label = f"{label} (sub-{int(participant):02d})"
    return {
        "job_id": job_id,
        "dataset": dataset,
        "model": model,
        "rsa_model": rsa_model,
        "specie": specie,
        "step": step,
        "label": label,
        "status": "pending",
        "deps": [],
        "rsa_method": rsa_method,
        "dis_method": dis_method,
        "mah_fold": mah_fold,
        "z_threshold": z_threshold,
        "reps": reps,
        "reps_group": reps_group,
        "participant": (int(participant) if participant is not None else None),
        "radius": radius,
        "mask_type": mask_type,
        "replace_file": replace_file,
        "replace_rnd_files": replace_rnd_files,
        "created_at": None,
        "started_at": None,
        "completed_at": None,
        "machine": None,
        "error": None,
    }
