"""
build_rsa_models.py — generate the full factorial battery of RSA model matrices
for the EmoC stimulus design (2 species-shown x 5 emotions x 4 exemplars = 40
conditions), ready to feed into searchlight.py / the scheduler.

Each model is a 40x40 dissimilarity matrix over the 40 conditions:
    0    -> predicted SAME representation
    0.5  -> predicted intermediate (graded models only)
    1    -> predicted DIFFERENT representation
    NaN  -> pair EXCLUDED from the RSA correlation
The diagonal is always 0.

One file is written per model into {datafolder}/EmoC/rsa_models/:
    {name}.csv   — the format searchlight.py / read_model_dict loads, and the
                   format rsa_model_builder.py reads & writes.

By default the output goes to the pipeline's data disk (scheduler/paths.py:
``P:\\userdata\\raulh87\\data`` on Windows, the network mount on Linux) so the models are
immediately runnable by searchlight.py / the scheduler. Use --out_dir to target a
different location (e.g. the Google-Drive results mirror used by the builder UI).

Run:
    & "C:\\ProgramData\\anaconda3\\python.exe" build_rsa_models.py
    & "C:\\ProgramData\\anaconda3\\python.exe" build_rsa_models.py --dry-run

Design dimensions (see EmoC_Stimulus_Design_Configuration_Handoff.md):
    emotions       P=Positive anticipation, H=Happiness, A=Anger, F=Fear, N=Neutral
    species_shown  Dog / Hum   (the agent shown in the video, NOT the brain)
    exemplars      1..4        (treated as the same condition within an emotion cell)

The SAME csv/xlsx serves both the dog-brain (D) and human-brain (H) analyses;
the brain species is chosen at run time via searchlight.py --specie.
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scheduler.paths import get_paths

# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

SPECIES   = ["Dog", "Hum"]
EMOTIONS  = ["P", "H", "A", "F", "N"]          # column / row ordering within a species
EXEMPLARS = [1, 2, 3, 4]

EMOTION_NAME = {
    "P": "Positive anticipation", "H": "Happiness",
    "A": "Anger", "F": "Fear", "N": "Neutral",
}

# (name, species_shown, emotion) in the canonical CSV order
CONDITIONS = [
    (f"{s}{e}{x}", s, e)
    for s in SPECIES
    for e in EMOTIONS
    for x in EXEMPLARS
]
NAMES = [c[0] for c in CONDITIONS]
N = len(CONDITIONS)  # 40

# ---------------------------------------------------------------------------
# Emotion-level property maps
# ---------------------------------------------------------------------------

VALENCE3 = {"P": "pos", "H": "pos", "A": "neg", "F": "neg", "N": "neu"}
THREAT   = {"A": "threat", "F": "threat", "P": "safe", "H": "safe", "N": "safe"}
APPROACH = {"P": "approach", "H": "approach", "A": "avoid", "F": "avoid", "N": None}
EMO_NEU  = {"P": "emo", "H": "emo", "A": "emo", "F": "emo", "N": "neu"}

# Graded axes (rank distance / max -> 0, 0.5, 1)
VAL_RANK = {"P": 2, "H": 2, "N": 1, "A": 0, "F": 0}     # positive .. neutral .. negative
AR_RANK  = {"A": 2, "F": 2, "P": 1, "H": 1, "N": 0}     # threat-high .. positive-mid .. neutral-low

NAN = np.nan

# ---------------------------------------------------------------------------
# Base emotion->emotion dissimilarity rules
# ---------------------------------------------------------------------------

def _same_diff(mapping):
    def rule(ei, ej):
        a, b = mapping[ei], mapping[ej]
        if a is None or b is None:      # category excluded (e.g. Neutral in approach/avoid)
            return NAN
        return 0.0 if a == b else 1.0
    return rule

def _binary_excl_neutral(ei, ej):
    if ei == "N" or ej == "N":
        return NAN
    return 0.0 if VALENCE3[ei] == VALENCE3[ej] else 1.0

def _graded(rank):
    def rule(ei, ej):
        return abs(rank[ei] - rank[ej]) / 2.0
    return rule

BASE_RULES = {
    "emo-id":         (_same_diff({e: e for e in EMOTIONS}),
                       "Emotion identity (5-way): same emotion=0, different=1."),
    "val3":           (_same_diff(VALENCE3),
                       "Valence 3-class: same {pos|neg|neu}=0, different=1."),
    "val-bin":        (_binary_excl_neutral,
                       "Valence binary: positive vs negative=1, within=0; Neutral excluded."),
    "emo-vs-neu":     (_same_diff(EMO_NEU),
                       "Emotional vs Neutral: any emotion grouped together vs Neutral."),
    "threat":         (_same_diff(THREAT),
                       "Threat: {Anger,Fear} vs {others}."),
    "approach-avoid": (_same_diff(APPROACH),
                       "Approach {P,H} vs Avoid {A,F}=1, within=0; Neutral excluded."),
    "grad-val":       (_graded(VAL_RANK),
                       "Graded valence: pos->neu->neg axis, adjacent=0.5, opposite=1."),
    "grad-arousal":   (_graded(AR_RANK),
                       "Graded arousal: {A,F}=high, {P,H}=mid, {N}=low; adjacent=0.5, far=1."),
}

# ---------------------------------------------------------------------------
# Agent-species-shown treatments (how a base rule sees the Dog/Hum dimension)
# ---------------------------------------------------------------------------

def _treat(name):
    if name == "collapse":   # ignore agent species entirely
        return lambda si, sj: True
    if name == "within":     # only Dog-Dog and Hum-Hum pairs
        return lambda si, sj: si == sj
    if name == "cross":      # only Dog-Hum pairs (tests agent-invariant emotion)
        return lambda si, sj: si != sj
    if name == "dog":        # only the Dog block
        return lambda si, sj: si == "Dog" and sj == "Dog"
    if name == "hum":        # only the Hum block
        return lambda si, sj: si == "Hum" and sj == "Hum"
    raise ValueError(name)

TREATMENTS = {
    "collapse": "collapsed across agent species (Dog/Hum pooled)",
    "within":   "within agent species only (Dog-Dog & Hum-Hum)",
    "cross":    "cross agent species only (Dog-Hum) — agent-invariant test",
    "dog":      "Dog-shown block only",
    "hum":      "Hum-shown block only",
}

# ---------------------------------------------------------------------------
# Matrix builders
# ---------------------------------------------------------------------------

def build_matrix(base_rule, treat_ok):
    m = np.full((N, N), NAN, dtype=np.float64)
    for i, (_, si, ei) in enumerate(CONDITIONS):
        for j, (_, sj, ej) in enumerate(CONDITIONS):
            if i == j:
                m[i, j] = 0.0
                continue
            if not treat_ok(si, sj):
                continue
            m[i, j] = base_rule(ei, ej)
    return m

def build_agent_species_identity():
    """Emotion-agnostic: Dog-shown vs Hum-shown."""
    m = np.full((N, N), NAN, dtype=np.float64)
    for i, (_, si, _) in enumerate(CONDITIONS):
        for j, (_, sj, _) in enumerate(CONDITIONS):
            m[i, j] = 0.0 if (i == j or si == sj) else 1.0
    return m

# ---------------------------------------------------------------------------
# Stats / serialization
# ---------------------------------------------------------------------------

def matrix_stats(m):
    iu = np.triu_indices(N, k=1)
    vals = m[iu]
    finite = vals[~np.isnan(vals)]
    return {
        "pairs_used": int(finite.size),
        "n_0":   int(np.sum(finite == 0.0)),
        "n_half": int(np.sum(finite == 0.5)),
        "n_1":   int(np.sum(finite == 1.0)),
        "has_variance": bool(finite.size > 0 and np.ptp(finite) > 0),
    }

def write_model(out_dir, name, m, dry_run=False):
    df = pd.DataFrame(m, index=NAMES, columns=NAMES)
    csv_path = os.path.join(out_dir, f"{name}.csv")
    if not dry_run:
        df.to_csv(csv_path, na_rep="NaN")
        # Remove any stale .npy cache so read_model_dict re-reads the new matrix.
        npy = os.path.join(out_dir, f"{name}.npy")
        if os.path.exists(npy):
            os.remove(npy)
    return csv_path

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate the EmoC RSA model battery.")
    ap.add_argument("--dataset", default="EmoC")
    ap.add_argument("--out_dir", default=None,
                    help="Override output folder (default: {datafolder}/{dataset}/rsa_models)")
    ap.add_argument("--prefix", default="",
                    help="Optional filename prefix, e.g. 'emoc-' ")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan and stats without writing files.")
    args = ap.parse_args()

    if args.out_dir:
        out_dir = args.out_dir
    else:
        root = get_paths()[0]   # pipeline data disk (P:\ on Windows / network mount on Linux)
        out_dir = os.path.join(root, args.dataset, "rsa_models")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Conditions: {N}  ({len(SPECIES)} species x {len(EMOTIONS)} emotions x {len(EXEMPLARS)} exemplars)")
    print(f"Output dir: {out_dir}")
    print(f"Dry run:    {args.dry_run}\n")

    manifest = []

    # Factorial: every base hypothesis x every agent-species treatment.
    for hyp, (base_rule, hyp_desc) in BASE_RULES.items():
        for treat, treat_desc in TREATMENTS.items():
            name = f"{args.prefix}{hyp}__{treat}"
            m = build_matrix(base_rule, _treat(treat))
            st = matrix_stats(m)
            write_model(out_dir, name, m, dry_run=args.dry_run)
            manifest.append({
                "model": name, "hypothesis": hyp, "treatment": treat,
                "description": f"{hyp_desc} | {treat_desc}", **st,
            })

    # Standalone: agent-species identity (emotion-agnostic).
    name = f"{args.prefix}agent-species-id"
    m = build_agent_species_identity()
    st = matrix_stats(m)
    write_model(out_dir, name, m, dry_run=args.dry_run)
    manifest.append({
        "model": name, "hypothesis": "agent-species-id", "treatment": "collapse",
        "description": "Agent species shown: Dog-shown vs Hum-shown (emotion ignored).", **st,
    })

    man_df = pd.DataFrame(manifest)
    # Warn about any degenerate (no-variance) model.
    bad = man_df[~man_df["has_variance"]]
    if len(bad):
        print("WARNING: models with no variance (cannot correlate):")
        print(bad[["model"]].to_string(index=False))

    print(f"\nGenerated {len(manifest)} models "
          f"({len(BASE_RULES)} hypotheses x {len(TREATMENTS)} treatments + 1 standalone).\n")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(man_df[["model", "pairs_used", "n_0", "n_half", "n_1"]].to_string(index=False))

    if not args.dry_run:
        man_path = os.path.join(out_dir, "_MODEL_BATTERY_MANIFEST.csv")
        man_df.to_csv(man_path, index=False)
        print(f"\nManifest written: {man_path}")
        print(f"Wrote {len(manifest)} .csv files to {out_dir}")


if __name__ == "__main__":
    main()
