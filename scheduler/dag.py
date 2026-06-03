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


def make_job_id(dataset, model, rsa_model, specie, step, z_threshold, reps, reps_group,
                method="mahalanobis"):
    return (
        f"{dataset}__{model}__{rsa_model}__{specie}"
        f"__step{step:02d}__zt{z_threshold}__r{reps}__rg{reps_group}__m{method}"
    )


def build_job_graph(dataset, model, rsa_model, specie, target_step=10,
                    z_threshold=3.1, reps=100, reps_group=1000,
                    method="mahalanobis"):
    """
    Return job dicts in topological order (leaf steps first) for running
    target_step for the given specie.
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

    if specie == 'H':
        needed.discard(0)  # step 0 not applicable for humans

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
            make_job_id(dataset, model, rsa_model, specie, d, z_threshold, reps, reps_group, method)
            for d in dep_steps
        ]
        job_id = make_job_id(dataset, model, rsa_model, specie, step, z_threshold, reps, reps_group, method)
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
            "method": method,
            "z_threshold": z_threshold,
            "reps": reps,
            "reps_group": reps_group,
            "created_at": None,
            "started_at": None,
            "completed_at": None,
            "machine": None,
            "error": None,
        })
    return jobs
