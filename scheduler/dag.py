import hashlib

from .jobs import DEFAULT_PRIORITY, normalize_priority

STEP_LABELS = {
    0: "Beta maps",
    1: "Pairwise similarity",
    2: "Model similarity",
    3: "Group similarity map",
    4: "RND permuted model",
    4.5: "RND participant distribution",
    5: "RND group permutations",
    6: "Voxelwise RND distribution",
    7: "Z-maps",
    7.6: "Participant z-maps",
    8: "Cluster size distribution",
    9: "Cluster correction",
    10: "Create tables",
    15: "Multiple regression RSA",
}

# Generic dependency graph; species-specific adjustments applied in build_job_graph
_STEP_DEPS = {
    0:  [],
    1:  [0],     # for dogs; humans skip step 0 (beta maps pre-exist)
    2:  [1],
    3:  [2],
    4:  [1],     # RND uses pairwise maps, not group map
    4.5: [4],    # per participant mean/std across that unit's permuted maps
    5:  [4],
    6:  [5],
    7:  [3, 6],  # z-maps need group map AND voxelwise RND distribution
    7.6: [2, 4.5],  # participant z-maps need the real map AND its own distribution
    8:  [7],
    9:  [7, 8],  # cluster correction needs real z-map AND cluster dist
    10: [9],
    15: [1],     # regresses the step-1 pairwise maps against the model design
}

# Steps 4.5, 7.6 and 15 hang off the main line rather than feeding it: nothing
# in 5..10 lists them as a dependency, so a classic 2->10 graph is unchanged and
# they are only scheduled when asked for by name.


def step_token(step):
    """Render a step for a job id: 'step04' for 4, 'step04.5' for 4.5.

    Integer steps keep the exact spelling every existing job id, marker folder
    and queue lookup already uses -- a float 4.0 read back from a job JSON also
    renders as 'step04', so a round trip through the queue cannot change an id.
    """
    if float(step) == int(step):
        return f"step{int(step):02d}"
    return f"step{float(step):04.1f}"


def rsa_models_token(rsa_models_list):
    """Render a whole list of RSA models as one job-id token.

    Step 15 fits many target models inside a single job -- the run's pairwise
    maps are loaded once and every target reuses them -- so the id cannot name
    them all. The token is the model count plus a digest of the sorted names:
    the same set in any order gives the same token (so a re-schedule is still
    recognised as a duplicate), and any other set gives a different one (so two
    batteries never collide in the queue). The names themselves live in the job
    JSON's ``rsa_models_list``.
    """
    names = sorted(set(rsa_models_list))
    digest = hashlib.sha1('|'.join(names).encode('utf-8')).hexdigest()[:8]
    return f"list{len(names)}-{digest}"


def make_job_id(dataset, model, rsa_model, specie, step, z_threshold, reps, reps_group, rsa_method="kendall", dis_method="mahalanobis", mah_fold="stim-wise", participant=None, regression_model=None, rsa_models_list=None):
    # A batch job (several target models, one shared load of the pairwise maps)
    # takes a digest token in the rsa_model slot instead of a model name.
    if rsa_models_list:
        if rsa_model is not None:
            raise ValueError(
                "Pass either rsa_model or rsa_models_list, not both: the id can "
                "only name one target set."
            )
        rsa_model = rsa_models_token(rsa_models_list)
    elif rsa_model is None:
        raise ValueError("A job id needs an rsa_model or a non-empty rsa_models_list.")
    job_id = (
        f"{dataset}__{model}__{rsa_model}__{specie}"
        f"__{step_token(step)}__zt{z_threshold}__r{reps}__rg{reps_group}"
        f"__rsa{rsa_method}__dis{dis_method}__mah{mah_fold}"
    )
    if regression_model is not None:
        job_id += f"__reg{regression_model}"
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
                    replace_rnd_files=False,
                    verbose=True,
                    priority=DEFAULT_PRIORITY,
                    min_percentage_available=1.0,
                    regression_model=None):
    """
    Return job dicts in topological order (leaf steps first) for running
    target_step for the given specie.  Steps below start_step are never
    scheduled (their outputs are assumed to already exist on disk).

    ``priority`` (1 = first, 3 = last, the default) is stamped on every job in
    the graph and decides the order ``run_jobs.py`` claims *pending* jobs. It
    does not reorder the graph itself: a step still waits for its deps whatever
    its priority.
    """
    priority = normalize_priority(priority)
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
            make_job_id(dataset, model, rsa_model, specie, d, z_threshold, reps, reps_group, rsa_method, dis_method, mah_fold, regression_model=regression_model)
            for d in dep_steps
        ]
        job_id = make_job_id(dataset, model, rsa_model, specie, step, z_threshold, reps, reps_group, rsa_method, dis_method, mah_fold, regression_model=regression_model)
        jobs.append({
            "job_id": job_id,
            "dataset": dataset,
            "model": model,
            "rsa_model": rsa_model,
            "regression_model": regression_model,
            "specie": specie,
            "step": step,
            "label": STEP_LABELS.get(step, f"Step {step}"),
            "status": "pending" if not dep_ids else "waiting",
            "deps": dep_ids,
            "priority": priority,
            "rsa_method": rsa_method,
            "dis_method": dis_method,
            "mah_fold": mah_fold,
            "min_percentage_available": min_percentage_available,
            "z_threshold": z_threshold,
            "reps": reps,
            "reps_group": reps_group,
            "replace_rnd_files": replace_rnd_files,
            "verbose": verbose,
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
                     replace_file=False, replace_rnd_files=False,
                     verbose=True, priority=DEFAULT_PRIORITY,
                     min_percentage_available=1.0, regression_model=None,
                     rsa_models_list=None):
    """Build a single, *independent* job dict (no dependencies, status=pending).

    Used by the dashboard's "schedule missing" / per-map buttons: the user has
    looked at the disk and wants to (re)compute one specific map, so the job is
    created ready-to-run with an empty ``deps`` list. ``participant`` (an int)
    scopes the job to one subject via searchlight's ``--participants_forced``;
    ``participant=None`` schedules the whole step (the single group map for the
    group steps 3/5/6/7/8/9/10).

    The extra ``radius`` / ``mask_type`` / ``replace_file`` /
    ``min_percentage_available`` fields are honoured by
    ``run_jobs.build_command`` (they are absent from classic scheduler jobs,
    which read them with ``.get()`` and fall back to searchlight's defaults).

    ``priority`` (1 = first, 3 = last, the default) decides the order
    ``run_jobs.py`` claims this job relative to the rest of the pending queue.

    ``rsa_models_list`` builds a *batch* job instead: pass ``rsa_model=None``
    and a list of target models, and the one job runs all of them (step 15 loads
    each run's pairwise maps once and fits every target against that stack, so
    splitting them into one job per model would re-read the same maps N times).
    ``run_jobs.build_command`` forwards the list as ``--rsa_models_list``.
    """
    rsa_models_list = list(rsa_models_list) if rsa_models_list else None
    job_id = make_job_id(dataset, model, rsa_model, specie, step, z_threshold,
                         reps, reps_group, rsa_method, dis_method, mah_fold,
                         participant=participant,
                         regression_model=regression_model,
                         rsa_models_list=rsa_models_list)
    label = STEP_LABELS.get(step, f"Step {step}")
    if rsa_models_list is not None:
        label = f"{label} ({len(rsa_models_list)} models)"
    if participant is not None:
        label = f"{label} (sub-{int(participant):02d})"
    return {
        "job_id": job_id,
        "dataset": dataset,
        "model": model,
        "rsa_model": rsa_model,
        "rsa_models_list": rsa_models_list,
        "regression_model": regression_model,
        "specie": specie,
        "step": step,
        "label": label,
        "status": "pending",
        "deps": [],
        "priority": normalize_priority(priority),
        "rsa_method": rsa_method,
        "dis_method": dis_method,
        "mah_fold": mah_fold,
        "min_percentage_available": min_percentage_available,
        "z_threshold": z_threshold,
        "reps": reps,
        "reps_group": reps_group,
        "participant": (int(participant) if participant is not None else None),
        "radius": radius,
        "mask_type": mask_type,
        "replace_file": replace_file,
        "replace_rnd_files": replace_rnd_files,
        "verbose": verbose,
        "created_at": None,
        "started_at": None,
        "completed_at": None,
        "machine": None,
        "error": None,
    }
