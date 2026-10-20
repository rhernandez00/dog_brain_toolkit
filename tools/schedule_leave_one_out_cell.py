# %% Schedule leave-one-out RSA jobs for comparison classes.
import importlib
import sys

sys.path.insert(0, r"C:\github\dog_brain_toolkit")

import scheduler.paths
import scheduler.jobs
import scheduler.dag

for module in (scheduler.paths, scheduler.jobs, scheduler.dag):
    importlib.reload(module)

from scheduler.paths import get_queue_dir
from scheduler.dag import build_job_graph
from scheduler.jobs import create_job

DATASET = "EmoC"
DATAFOLDER = r"P:\userdata\raulh87\data"
MODEL = "basic-block"
SPECIE = "D"
RADIUS = 4
RSA_CLASSES = ["all", "dog", "hum"]
STEPS = [11.2, 11.3, 11.4, 11.5]

DIS_METHOD = "mahalanobis"
MAH_FOLD = "stim-wise"  # Must match the existing step-1 pairwise maps.
RSA_METHOD = "kendall"
MASK_TYPE = None  # Use searchlight.py's species default.
REPS = 100  # Participant permutations (11.4).
REPS_GROUP = 1000  # Group permutations (11.5).
MIN_PERCENTAGE_AVAILABLE = 1.0
PRIORITY = 2
SHUFFLE_PARTICIPANTS = True
REPLACE_FILE = False
REPLACE_RND_FILES = False
DRY_RUN = False
VERBOSE = True

# Step-1 pairwise maps and rsa_models/by_class/class_<name>.csv must exist.
# Build dependencies, deduplicating steps shared by the requested graphs.
jobs_by_id = {}
for rsa_class in RSA_CLASSES:
    for step in STEPS:
        for job in build_job_graph(
            dataset=DATASET,
            model=MODEL,
            rsa_model=None,
            rsa_class=rsa_class,
            specie=SPECIE,
            start_step=11.2,  # Step 1 already exists; schedule only class steps.
            target_step=step,
            radius=RADIUS,
            mask_type=MASK_TYPE,
            rsa_method=RSA_METHOD,
            dis_method=DIS_METHOD,
            mah_fold=MAH_FOLD,
            reps=REPS,
            reps_group=REPS_GROUP,
            min_percentage_available=MIN_PERCENTAGE_AVAILABLE,
            priority=PRIORITY,
            replace_file=REPLACE_FILE,
            replace_rnd_files=REPLACE_RND_FILES,
            verbose=VERBOSE,
        ):
            job["shuffle_participants"] = SHUFFLE_PARTICIPANTS
            jobs_by_id[job["job_id"]] = job

queue_dir = None if DRY_RUN else get_queue_dir(DATAFOLDER)
created = 0
for job in jobs_by_id.values():
    if DRY_RUN:
        print(f"[dry run] {job['status']}: {job['job_id']}")
    else:
        create_job(queue_dir, job)
        created += 1
        print(f"[created] {job['status']}: {job['job_id']}")

print(
    f"\n{len(jobs_by_id)} jobs planned; {created} created. "
    f"Species={SPECIE}, radius={RADIUS}, classes={RSA_CLASSES}, dry_run={DRY_RUN}"
)
