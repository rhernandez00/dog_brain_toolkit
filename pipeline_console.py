#!/usr/bin/env python
"""
pipeline_console.py — Interactive progress console for the RSA pipeline.

What it does
------------
Reads the ``rsa_models/`` folder of a dataset to discover which RSA models can be
run, then — for a model you select — probes the **actual output files on disk**
for every pipeline step (0-10). For each step it reports whether the expected
files already exist, and for the per-participant steps (0, 1, 2, 4) it tells you
which participants passed and which are missing, so you can see exactly where a
model (or a single participant) got stuck. When the job scheduler was used, it
also surfaces the recorded error ("why") from ``job_queue/failed/``.

This is a *read-only* status tool — it never computes anything. "Done" means the
files that ``searchlight.py`` would have written are present on disk.

Usage
-----
Interactive (menu-driven):
    python pipeline_console.py --dataset EmoC

Non-interactive one-shot report for a specific model:
    python pipeline_console.py --dataset EmoC --rsa_model test-model --specie D --report

Key options (all optional except --dataset):
    --model        GLM model               (default: basic-block)
    --rsa_model    RSA model CSV name       (interactive picker if omitted)
    --specie       D or H                   (default: D)
    --method       pairwise method          (default: mahalanobis)
    --rsa_method   model-comparison method  (default: kendall)
    --radius       searchlight radius       (default: 3 for D, 4 for H)
    --z_threshold  z threshold              (default: 3.1)
    --mask_type    mask selector            (default: b_GreyMatter2mmB)
    --report       print full report and exit (no interactive menu)

Notes on accuracy
-----------------
Output filenames are reconstructed from the exact conventions in
``rsa_utils.py``. The group/final steps (3, 5, 6, 7, 8, 9, 10) use precise paths;
the per-participant steps (0, 1, 2, 4) use glob patterns to tolerate the small
session-padding inconsistencies that exist in the pipeline. Per-participant
"expected" counts rely on ``get_session_and_run_dict`` (reads the data disk) and
on the RSA-model categories, so they require the data disk to be mounted.
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scheduler.paths import get_paths, get_queue_dir  # noqa: E402

# STEP_LABELS / make_job_id come from the scheduler so labels stay in sync.
try:
    from scheduler.dag import STEP_LABELS, make_job_id
except Exception:  # pragma: no cover - scheduler is expected to be importable
    STEP_LABELS = {
        0: "Beta maps", 1: "Pairwise similarity", 2: "Model similarity",
        3: "Group similarity map", 4: "RND permuted model",
        5: "RND group permutations", 6: "Voxelwise RND distribution",
        7: "Z-maps", 8: "Cluster size distribution", 9: "Cluster correction",
        10: "Create tables",
    }
    make_job_id = None

STEPS = list(range(0, 11))

# ---------------------------------------------------------------------------
# Terminal colours (degrade gracefully when not a TTY)
# ---------------------------------------------------------------------------
_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(text, code):
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def green(t):  return _c(t, "32")
def red(t):    return _c(t, "31")
def yellow(t): return _c(t, "33")
def grey(t):   return _c(t, "90")
def bold(t):   return _c(t, "1")
def cyan(t):   return _c(t, "36")


# Status verdicts a step probe can return.
DONE = "DONE"        # all expected outputs present
PARTIAL = "PARTIAL"  # some but not all present (per-participant or per-perm)
MISSING = "MISSING"  # nothing present
NA = "N/A"           # step not applicable (e.g. beta maps for humans)
UNKNOWN = "UNKNOWN"  # could not determine (missing inputs to probe)

_VERDICT_STYLE = {
    DONE: lambda t: green(t),
    PARTIAL: lambda t: yellow(t),
    MISSING: lambda t: red(t),
    NA: lambda t: grey(t),
    UNKNOWN: lambda t: grey(t),
}


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------
class Ctx:
    """Everything a probe needs to locate a model's files."""

    def __init__(self, datafolder, dataset, model, rsa_model, specie, method,
                 rsa_method, radius, z_threshold, mask_type, reps, reps_group):
        self.datafolder = datafolder
        self.dataset = dataset
        self.model = model
        self.rsa_model = rsa_model
        self.specie = specie
        self.method = method
        self.rsa_method = rsa_method
        self.radius = radius
        self.z_threshold = z_threshold
        self.mask_type = mask_type
        self.reps = reps
        self.reps_group = reps_group

        # Lazily resolved (need the data disk / rsa_utils).
        self.task = dataset
        self.participants = []
        self.stim_types = []
        self.categories = []
        self._resolve_error = None

    # --- path helpers ------------------------------------------------------
    @property
    def rsa_dir(self):
        return os.path.join(self.datafolder, self.dataset, 'results', 'RSA', self.model)

    @property
    def rsa_rnd_dir(self):
        return os.path.join(self.datafolder, self.dataset, 'results', 'RSA_rnd', self.model)

    @property
    def model_mean_dir(self):
        return os.path.join(self.rsa_dir, self.rsa_model, 'mean')

    @property
    def model_dist_dir(self):
        return os.path.join(self.rsa_dir, self.rsa_model, 'dist')

    def core(self, with_mask):
        """The filename stem shared by group/final maps.

        With mask:  ``{mask}-{specie}-r-{radius}_{method}_{rsa_method}``
        Without:    ``{specie}-r-{radius}_{method}_{rsa_method}``
        """
        base = f"{self.specie}-r-{self.radius}_{self.method}_{self.rsa_method}"
        if with_mask and self.mask_type:
            return f"{self.mask_type}-{base}"
        return base

    def sub_folder(self, sub_N):
        return f"{self.specie}-sub-{sub_N:02d}"


def build_ctx(args, datafolder):
    ctx = Ctx(
        datafolder=datafolder,
        dataset=args.dataset,
        model=args.model,
        rsa_model=args.rsa_model,
        specie=args.specie,
        method=args.method,
        rsa_method=args.rsa_method,
        radius=args.radius,
        z_threshold=args.z_threshold,
        mask_type=(None if args.mask_type in ("none", "None", "") else args.mask_type),
        reps=args.reps,
        reps_group=args.reps_group,
    )
    _resolve_dynamic(ctx)
    return ctx


def _resolve_dynamic(ctx):
    """Fill in participants, task, stim_types, categories from config + rsa model.

    Best-effort: records an error string in ctx._resolve_error rather than
    raising, so the group-level probes still work if the disk is unavailable.
    """
    # radius default mirrors searchlight.py (3 for dogs, 4 for humans)
    if ctx.radius is None:
        ctx.radius = 3 if ctx.specie == 'D' else 4

    # config YAML -> task, participants, stim_types
    config_path = os.path.join(
        ctx.datafolder, ctx.dataset, 'config_files', f"{ctx.specie}_{ctx.model}.yaml"
    )
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        ctx.task = config.get('task', ctx.dataset) or ctx.dataset
        ctx.stim_types = config.get('stim_types', []) or []
        # searchlight.py forces humans to participants 1..40
        if ctx.specie == 'H':
            ctx.participants = list(range(1, 41))
        else:
            ctx.participants = config.get('participants', []) or []
    except Exception as e:
        ctx._resolve_error = f"config not read ({config_path}): {e}"
        if ctx.specie == 'H' and not ctx.participants:
            ctx.participants = list(range(1, 41))

    # RSA model categories (for mahalanobis pairwise / model comparison)
    if ctx.rsa_model:
        rsa_model_path = os.path.join(
            ctx.datafolder, ctx.dataset, 'rsa_models', f"{ctx.rsa_model}.csv"
        )
        try:
            import rsa_utils
            ctx.categories = rsa_utils.read_model_dict(rsa_model_path).get('categories', [])
        except Exception as e:
            if ctx._resolve_error:
                ctx._resolve_error += f"; categories not read: {e}"
            else:
                ctx._resolve_error = f"categories not read: {e}"


def _sessions_for(ctx, sub_N):
    """Return the session/run list for a participant, or None if unavailable."""
    try:
        import rsa_utils
        return rsa_utils.get_session_and_run_dict(
            ctx.datafolder, ctx.dataset, ctx.specie, sub_N
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Step probes
# ---------------------------------------------------------------------------
def _result(verdict, summary, expected=None, found=None, detail=None, per_sub=None):
    return {
        "verdict": verdict,
        "summary": summary,
        "expected": expected,
        "found": found,
        "detail": detail or [],
        "per_sub": per_sub or [],   # list of (sub_N, verdict, note)
    }


def _exists(path):
    return os.path.exists(path)


def _count(pattern):
    return len(glob.glob(pattern))


def probe_step0(ctx):
    """Beta maps (dogs only) — GLM pe files per participant/session/run."""
    if ctx.specie == 'H':
        return _result(NA, "humans use pre-existing beta maps")
    glm_dir = os.path.join(ctx.datafolder, ctx.dataset, 'results', 'GLM', ctx.model)
    if not ctx.participants:
        return _result(UNKNOWN, "participant list unavailable")
    per_sub, done, partial = [], 0, 0
    for sub in ctx.participants:
        sess = _sessions_for(ctx, sub)
        sub_glob = os.path.join(glm_dir, ctx.sub_folder(sub), '*', 'stats', 'pe*.nii.gz')
        n = _count(sub_glob)
        if sess is None:
            per_sub.append((sub, UNKNOWN if n == 0 else PARTIAL, f"{n} pe files"))
            continue
        n_runs = len(sess)
        if n_runs and n >= n_runs:  # at least one pe per run
            per_sub.append((sub, DONE, f"{n} pe files / {n_runs} runs"))
            done += 1
        elif n > 0:
            per_sub.append((sub, PARTIAL, f"{n} pe files / {n_runs} runs"))
            partial += 1
        else:
            per_sub.append((sub, MISSING, f"0 / {n_runs} runs"))
    return _summarize(per_sub, done, partial, "beta maps")


def probe_step1(ctx):
    """Pairwise similarity maps per participant."""
    if not ctx.participants:
        return _result(UNKNOWN, "participant list unavailable")
    is_mah = ctx.method == 'mahalanobis'
    # expected number of pairs (upper triangle)
    units = ctx.categories if is_mah else ctx.stim_types
    exp_pairs = len(units) * (len(units) - 1) // 2 if units else None
    per_sub, done, partial = [], 0, 0
    for sub in ctx.participants:
        sub_dir = os.path.join(ctx.rsa_dir, ctx.sub_folder(sub))
        if is_mah:
            # mahalanobis stim-wise: files live directly under the subject folder
            n = _count(os.path.join(sub_dir, f"r-{ctx.radius}_{ctx.method}_*.nii.gz"))
        else:
            # correlation etc.: per-run subfolders
            n = _count(os.path.join(sub_dir, 'ses-*run-*', f"r-{ctx.radius}_{ctx.method}_*.nii.gz"))
        per_sub.append(_grade_count(sub, n, exp_pairs))
        done += per_sub[-1][1] == DONE
        partial += per_sub[-1][1] == PARTIAL
    return _summarize(per_sub, done, partial, "pairwise maps")


def probe_step2(ctx):
    """Model similarity maps (real) per participant."""
    if not ctx.participants:
        return _result(UNKNOWN, "participant list unavailable")
    per_sub, done, partial = [], 0, 0
    fname = f"r-{ctx.radius}_{ctx.method}_{ctx.rsa_method}.nii.gz"
    masked = f"{ctx.mask_type}-{fname}" if ctx.mask_type else fname
    for sub in ctx.participants:
        sub_dir = os.path.join(ctx.rsa_dir, ctx.rsa_model, ctx.sub_folder(sub))
        # tolerate both masked and unmasked names, across run subfolders
        n = _count(os.path.join(sub_dir, 'ses-*run-*', masked))
        if n == 0 and ctx.mask_type:
            n = _count(os.path.join(sub_dir, 'ses-*run-*', fname))
        sess = _sessions_for(ctx, sub)
        exp = len(sess) if sess is not None else None
        per_sub.append(_grade_count(sub, n, exp))
        done += per_sub[-1][1] == DONE
        partial += per_sub[-1][1] == PARTIAL
    return _summarize(per_sub, done, partial, "model-similarity maps")


def probe_step4(ctx):
    """RND permuted model similarity (per participant)."""
    if not ctx.participants:
        return _result(UNKNOWN, "participant list unavailable")
    per_sub, done, partial = [], 0, 0
    for sub in ctx.participants:
        sub_rnd = os.path.join(ctx.rsa_rnd_dir, ctx.rsa_model, ctx.sub_folder(sub))
        n = _count(os.path.join(sub_rnd, '**', f"r-{ctx.radius}_{ctx.method}_{ctx.rsa_method}_*.nii.gz"))
        if n == 0:
            n = _count(os.path.join(sub_rnd, f"r-{ctx.radius}_{ctx.method}_{ctx.rsa_method}_*.nii.gz"))
        if n >= ctx.reps:
            per_sub.append((sub, DONE, f"{n} perms (>= {ctx.reps})")); done += 1
        elif n > 0:
            per_sub.append((sub, PARTIAL, f"{n} / {ctx.reps} perms")); partial += 1
        else:
            per_sub.append((sub, MISSING, f"0 / {ctx.reps} perms"))
    return _summarize(per_sub, done, partial, "permutations")


def probe_step3(ctx):
    """Group model similarity map (single file)."""
    p = os.path.join(ctx.model_mean_dir, f"{ctx.core(with_mask=True)}_mean.nii.gz")
    if _exists(p):
        return _result(DONE, "group mean map present", found=[p])
    # fall back to the no-mask name in case it was run with mask_type=None
    p2 = os.path.join(ctx.model_mean_dir, f"{ctx.core(with_mask=False)}_mean.nii.gz")
    if _exists(p2):
        return _result(DONE, "group mean map present (no-mask name)", found=[p2])
    return _result(MISSING, "group mean map not found", expected=[p])


def probe_step5(ctx):
    """RND group permutation maps — one per group permutation (0..reps_group-1)."""
    mean_dir = os.path.join(ctx.rsa_rnd_dir, ctx.rsa_model, 'mean')
    pattern = os.path.join(
        mean_dir, f"{ctx.specie}-r-{ctx.radius}_{ctx.method}_{ctx.rsa_method}_mean_*.nii.gz"
    )
    n = _count(pattern)
    exp = ctx.reps_group
    if n >= exp:
        return _result(DONE, f"{n} group-perm maps (>= {exp})", expected=[f"{exp} maps"], found=[f"{n} maps"])
    if n > 0:
        return _result(PARTIAL, f"{n} / {exp} group-perm maps", expected=[f"{exp} maps"], found=[f"{n} maps"])
    return _result(MISSING, f"0 / {exp} group-perm maps", expected=[pattern])


def probe_step6(ctx):
    """Voxelwise RND distribution (mean + std, no mask prefix)."""
    base = os.path.join(ctx.rsa_rnd_dir, f"{ctx.specie}-{ctx.rsa_model}")
    mean_p, std_p = f"{base}_mean.nii.gz", f"{base}_std.nii.gz"
    have = [p for p in (mean_p, std_p) if _exists(p)]
    if len(have) == 2:
        return _result(DONE, "null mean+std present", found=have)
    if have:
        return _result(PARTIAL, "only one of mean/std present", found=have,
                       expected=[mean_p, std_p])
    return _result(MISSING, "null distribution not found", expected=[mean_p, std_p])


def probe_step7(ctx):
    """Group z-map (single file)."""
    p = os.path.join(ctx.model_mean_dir, f"{ctx.core(with_mask=True)}_z.nii.gz")
    if _exists(p):
        return _result(DONE, "z-map present", found=[p])
    p2 = os.path.join(ctx.model_mean_dir, f"{ctx.core(with_mask=False)}_z.nii.gz")
    if _exists(p2):
        return _result(DONE, "z-map present (no-mask name)", found=[p2])
    return _result(MISSING, "z-map not found", expected=[p])


def probe_step8(ctx):
    """Cluster-size distribution (.npy, no mask prefix)."""
    p = os.path.join(
        ctx.model_dist_dir,
        f"{ctx.specie}-r-{ctx.radius}_{ctx.method}_{ctx.rsa_method}_dist.npy",
    )
    if _exists(p):
        return _result(DONE, "cluster-size distribution present", found=[p])
    return _result(MISSING, "cluster-size distribution not found", expected=[p])


def probe_step9(ctx):
    """Cluster-corrected z-map (z_threshold in name)."""
    name = f"{ctx.core(with_mask=True)}_zt{ctx.z_threshold}_corrected.nii.gz"
    p = os.path.join(ctx.model_mean_dir, name)
    if _exists(p):
        return _result(DONE, "corrected map present", found=[p])
    p2 = os.path.join(
        ctx.model_mean_dir,
        f"{ctx.core(with_mask=False)}_zt{ctx.z_threshold}_corrected.nii.gz",
    )
    if _exists(p2):
        return _result(DONE, "corrected map present (no-mask name)", found=[p2])
    return _result(MISSING, "corrected map not found", expected=[p])


def probe_step10(ctx):
    """Results table (.csv or .xlsx, z_threshold in name)."""
    stem_m = f"{ctx.core(with_mask=True)}_zt{ctx.z_threshold}"
    stem_n = f"{ctx.core(with_mask=False)}_zt{ctx.z_threshold}"
    candidates = []
    for stem in (stem_m, stem_n):
        for ext in ('.xlsx', '.csv'):
            candidates.append(os.path.join(ctx.model_mean_dir, stem + ext))
    found = [p for p in candidates if _exists(p)]
    if found:
        return _result(DONE, "results table present", found=found)
    return _result(MISSING, "results table not found", expected=candidates[:2])


PROBES = {
    0: probe_step0, 1: probe_step1, 2: probe_step2, 3: probe_step3,
    4: probe_step4, 5: probe_step5, 6: probe_step6, 7: probe_step7,
    8: probe_step8, 9: probe_step9, 10: probe_step10,
}


# ---------------------------------------------------------------------------
# Grading helpers for per-participant steps
# ---------------------------------------------------------------------------
def _grade_count(sub, n, expected):
    if expected is None:
        return (sub, PARTIAL if n > 0 else MISSING, f"{n} files (expected ?)")
    if n >= expected:
        return (sub, DONE, f"{n}/{expected}")
    if n > 0:
        return (sub, PARTIAL, f"{n}/{expected}")
    return (sub, MISSING, f"0/{expected}")


def _summarize(per_sub, done, partial, noun):
    total = len(per_sub)
    if total == 0:
        return _result(UNKNOWN, f"no participants to check for {noun}", per_sub=per_sub)
    if done == total:
        return _result(DONE, f"all {total} participants have {noun}", per_sub=per_sub)
    if done == 0 and partial == 0:
        return _result(MISSING, f"no participant has {noun}", per_sub=per_sub)
    return _result(
        PARTIAL,
        f"{done}/{total} done, {partial} partial, {total - done - partial} missing",
        per_sub=per_sub,
    )


# ---------------------------------------------------------------------------
# "Why" — pull failure info from the job queue
# ---------------------------------------------------------------------------
def find_failure_info(ctx, step):
    """Return a list of (job_id, error, log_path) for failed jobs matching
    this dataset/model/rsa_model/specie/step, or [] if none / unavailable."""
    try:
        queue_dir = get_queue_dir(ctx.datafolder)
    except Exception:
        return []
    failed_dir = Path(queue_dir) / 'failed'
    if not failed_dir.is_dir():
        return []
    prefix = f"{ctx.dataset}__{ctx.model}__{ctx.rsa_model}__{ctx.specie}__step{step:02d}"
    out = []
    for jf in failed_dir.glob(f"{prefix}*.json"):
        try:
            data = json.loads(jf.read_text())
        except Exception:
            data = {}
        out.append((jf.stem, data.get('error'), data.get('log_path')))
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_header(ctx):
    print()
    print(bold("RSA pipeline status"))
    print(f"  dataset   : {ctx.dataset}")
    print(f"  model     : {ctx.model}")
    print(f"  rsa_model : {cyan(ctx.rsa_model)}")
    print(f"  specie    : {ctx.specie}   radius: {ctx.radius}   method: {ctx.method}   "
          f"rsa_method: {ctx.rsa_method}")
    print(f"  mask_type : {ctx.mask_type}   z_threshold: {ctx.z_threshold}   "
          f"reps: {ctx.reps}   reps_group: {ctx.reps_group}")
    if ctx.participants:
        print(f"  participants: {len(ctx.participants)}")
    if ctx._resolve_error:
        print(yellow(f"  note: {ctx._resolve_error}"))
        print(yellow("  (per-participant steps need the data disk mounted)"))
    print()


def render_table(ctx, results):
    hdr = f"{'St':<3}{'Label':<28}{'Status':<10}Detail"
    print(hdr)
    print('-' * max(len(hdr), 60))
    for step in STEPS:
        r = results[step]
        label = STEP_LABELS.get(step, f"Step {step}")
        style = _VERDICT_STYLE.get(r['verdict'], lambda t: t)
        badge = style(f"{r['verdict']:<8}")
        print(f"{step:<3}{label:<28}{badge:<10}{r['summary']}")
    print()


def render_step_detail(ctx, step):
    r = PROBES[step](ctx)
    label = STEP_LABELS.get(step, f"Step {step}")
    print()
    print(bold(f"Step {step} — {label}"))
    style = _VERDICT_STYLE.get(r['verdict'], lambda t: t)
    print(f"  status : {style(r['verdict'])}  {r['summary']}")
    if r['expected']:
        print("  expected:")
        for p in r['expected'][:4]:
            print(grey(f"    {p}"))
    if r['found']:
        print("  found:")
        for p in r['found'][:6]:
            print(grey(f"    {p}"))
    if r['per_sub']:
        print("  per participant:")
        for sub, verdict, note in r['per_sub']:
            style_s = _VERDICT_STYLE.get(verdict, lambda t: t)
            mark = {DONE: '✓', PARTIAL: '~', MISSING: '✗',
                    UNKNOWN: '?'}.get(verdict, ' ')
            print(f"    {style_s(mark)} sub-{sub:02d}  {style_s(verdict):<9} {note}")
    # why (failures)
    fails = find_failure_info(ctx, step)
    if fails:
        print(red("  recorded failures (job queue):"))
        for job_id, err, log_path in fails:
            print(red(f"    {job_id}"))
            if err:
                print(f"      error: {err}")
            if log_path:
                print(grey(f"      log:   {log_path}"))
    print()
    return r


def run_all(ctx):
    return {step: PROBES[step](ctx) for step in STEPS}


def first_incomplete(results):
    """Return the first step that is not DONE/NA, or None if all done."""
    for step in STEPS:
        v = results[step]['verdict']
        if v not in (DONE, NA):
            return step
    return None


def render_report(ctx):
    render_header(ctx)
    results = run_all(ctx)
    render_table(ctx, results)
    fi = first_incomplete(results)
    if fi is None:
        print(green("All steps complete for this model. ✔"))
    else:
        r = results[fi]
        print(yellow(f"First incomplete step: {fi} ({STEP_LABELS.get(fi)}) — {r['summary']}"))
        # if per-participant, name the stragglers
        stuck = [f"sub-{s:02d}" for s, v, _ in r['per_sub'] if v in (MISSING, PARTIAL)]
        if stuck:
            print(yellow(f"  stuck participants: {', '.join(stuck)}"))
        fails = find_failure_info(ctx, fi)
        for job_id, err, log_path in fails:
            print(red(f"  failed job: {job_id}"))
            if err:
                print(f"    error: {err}")
    print()
    return results


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------
def list_rsa_models(datafolder, dataset):
    folder = os.path.join(datafolder, dataset, 'rsa_models')
    if not os.path.isdir(folder):
        return [], folder
    models = sorted(Path(p).stem for p in glob.glob(os.path.join(folder, '*.csv')))
    return models, folder


# ---------------------------------------------------------------------------
# Interactive menu
# ---------------------------------------------------------------------------
def _prompt(msg, default=None):
    suffix = f" [{default}]" if default is not None else ""
    try:
        ans = input(f"{msg}{suffix}: ").strip()
    except EOFError:
        return default
    return ans or default


def interactive(args, datafolder):
    while True:
        models, folder = list_rsa_models(datafolder, args.dataset)
        print()
        print(bold(f"RSA models in {folder}:"))
        if not models:
            print(red("  (none found — check --dataset and the data disk)"))
        for i, m in enumerate(models, 1):
            marker = cyan("  <- selected") if m == args.rsa_model else ""
            print(f"  {i:>2}. {m}{marker}")
        print()
        print("Commands:  number = pick model   d = dataset   s = species/params   "
              "q = quit")
        if args.rsa_model:
            print("           r = full report   .N = drill into step N (e.g. .7)")
        cmd = _prompt("choice")
        if cmd is None or cmd.lower() == 'q':
            print("bye.")
            return
        if cmd.lower() == 'd':
            args.dataset = _prompt("dataset", args.dataset)
            args.rsa_model = None
            continue
        if cmd.lower() == 's':
            args.specie = _prompt("specie (D/H)", args.specie) or args.specie
            args.method = _prompt("method", args.method) or args.method
            args.rsa_method = _prompt("rsa_method", args.rsa_method) or args.rsa_method
            rad = _prompt("radius (blank=auto)", "")
            args.radius = int(rad) if rad else None
            args.mask_type = _prompt("mask_type", args.mask_type) or args.mask_type
            zt = _prompt("z_threshold", str(args.z_threshold))
            args.z_threshold = float(zt) if zt else args.z_threshold
            args.reps = int(_prompt("reps", str(args.reps)) or args.reps)
            args.reps_group = int(_prompt("reps_group", str(args.reps_group)) or args.reps_group)
            continue
        if cmd.isdigit() and 1 <= int(cmd) <= len(models):
            args.rsa_model = models[int(cmd) - 1]
            ctx = build_ctx(args, datafolder)
            render_report(ctx)
            continue
        if cmd.lower() == 'r' and args.rsa_model:
            render_report(build_ctx(args, datafolder))
            continue
        # step drill-down: user typed ".N" (e.g. ".7") to avoid clashing with
        # the model-number picker above.
        if args.rsa_model and cmd.startswith('.') and cmd[1:].isdigit() and int(cmd[1:]) in STEPS:
            render_step_detail(build_ctx(args, datafolder), int(cmd[1:]))
            _prompt("press enter to continue", "")
            continue
        print(red("  unrecognised choice"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Interactive RSA pipeline progress console")
    p.add_argument('--dataset', required=True)
    p.add_argument('--model', default='basic-block')
    p.add_argument('--rsa_model', default=None)
    p.add_argument('--specie', default='D', choices=['D', 'H'])
    p.add_argument('--method', default='mahalanobis')
    p.add_argument('--rsa_method', default='kendall')
    p.add_argument('--radius', type=int, default=None)
    p.add_argument('--z_threshold', type=float, default=3.1)
    p.add_argument('--mask_type', default='b_GreyMatter2mmB')
    p.add_argument('--reps', type=int, default=100)
    p.add_argument('--reps_group', type=int, default=1000)
    p.add_argument('--report', action='store_true',
                   help='Print a one-shot report and exit (needs --rsa_model)')
    p.add_argument('--step', type=int, default=None, choices=STEPS,
                   help='With --report: show detail for a single step')
    return p.parse_args()


def main():
    args = parse_args()
    datafolder, _, _ = get_paths()

    if args.report:
        if not args.rsa_model:
            print(red("--report requires --rsa_model"))
            sys.exit(2)
        ctx = build_ctx(args, datafolder)
        if args.step is not None:
            render_header(ctx)
            render_step_detail(ctx, args.step)
        else:
            render_report(ctx)
        return

    interactive(args, datafolder)


if __name__ == '__main__':
    main()
