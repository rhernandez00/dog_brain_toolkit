"""EmoC stimulus design configuration.

Authoritative source: EmoC_Stimulus_Design_Configuration_Handoff.md
(Obsidian/Nexus/output). Keep this module in sync with that document.

The experiment uses video stimuli crossed by emotion label and species.
Labels, species, stimulus repetitions and partitions are *separate*
dimensions of the design — emotion category alone does not determine
repetition status; that depends on both run and label.
"""

# --- Emotion labels -------------------------------------------------------
# Short code -> human-readable name + display color (used for icon chips).
LABEL_DEF = {
    "P": {"name": "Positive anticipation", "color": "#f18b1f"},
    "H": {"name": "Happiness",             "color": "#ffcf00"},
    "A": {"name": "Anger",                 "color": "#e72222"},
    "F": {"name": "Fear",                  "color": "#763596"},
    "N": {"name": "Neutral",               "color": "#7a7a7a"},
}

# --- Species --------------------------------------------------------------
# These are the *stimulus* species (what is shown in the videos), distinct
# from the *subject* species (D=dog, H=human) used elsewhere in the pipeline.
STIM_SPECIES = ["Dog", "Hum"]

# --- Stimulus repetitions -------------------------------------------------
STIM_REPS = [1, 2, 3, 4]

# --- Runs -----------------------------------------------------------------
RUNS = ["run01", "run02", "run03", "run04", "run05", "run06"]

# --- Partition mappings ---------------------------------------------------
# 0 = non-repeated, 1 = partition 1, 2 = partition 2.
# Runs 1-4 may belong to partition 1 or be non-repeated depending on label;
# runs 5-6 are partition 2 for all labels.
PARTITIONS = {
    "run01": {"P": 1, "H": 1, "A": 1, "F": 0, "N": 0},
    "run02": {"P": 1, "H": 1, "A": 1, "F": 0, "N": 0},
    "run03": {"P": 1, "H": 0, "A": 0, "F": 1, "N": 1},
    "run04": {"P": 0, "H": 0, "A": 0, "F": 1, "N": 1},
    "run05": {label: 2 for label in LABEL_DEF},
    "run06": {label: 2 for label in LABEL_DEF},
}


def stimulus_conditions():
    """Return the full list of (species, label) condition codes.

    Each condition is the cell that an RSA model / GLM contrast operates on,
    e.g. ('Dog', 'P'). Returns 10 conditions (2 species x 5 labels).
    """
    return [(sp, lab) for sp in STIM_SPECIES for lab in LABEL_DEF]


def condition_code(species, label):
    """Two-letter-ish code for a condition chip, e.g. 'DgP', 'HuA'.

    Image icons will replace these later; for now a compact text token.
    """
    sp = {"Dog": "Dg", "Hum": "Hu"}.get(species, species[:2])
    return f"{sp}{label}"


def label_color(label):
    return LABEL_DEF.get(label, {}).get("color", "#888888")
