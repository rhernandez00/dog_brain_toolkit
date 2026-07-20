# RSA Pipeline Improvement Plan (`searchlight.py` + `rsa_utils.py`)

This plan lists concrete, incremental improvements to the RSA pipeline. It is
scoped to the two core files — `searchlight.py` (the CLI driver, ~600 lines) and
`rsa_utils.py` (the function library, ~5200 lines) — but touches a few adjacent
files where they are coupled.

The items are ordered by **impact-to-effort ratio**, grouped into phases so that
each phase leaves the repo in a working, shippable state. Nothing here changes
scientific behaviour unless explicitly called out; the goal is
maintainability, correctness guarantees, and documentation.

---

## Phase 0 — Guardrails first (do before refactoring)

Refactoring a 5000-line numerical module with **no tests** is risky. Establish a
safety net first.

1. **Golden-output regression test.** Pick the existing `test-model` on `EmoC`
   (dogs), run steps 2–10 with small `--reps 10 --reps_group 50`, and snapshot the
   resulting z-map + Excel table. Add a script (`tests/test_pipeline_smoke.py`)
   that re-runs and compares against the snapshot within a numerical tolerance.
   This is the single highest-value item — it lets every later change be verified.
2. **Unit tests for the pure helpers.** The self-contained numeric functions are
   easy to pin down and have no I/O: `kendall_tau_a`, `kendall_custom`,
   `shuffle_vector`, `crossnobis`, `world_coords`, `transform_coords`,
   `create_sphere_mask`, `_count_on_3d`, `cluster_masks_3d`. Add fixed-input /
   known-output tests.
3. **CI stub.** Add a minimal GitHub Actions workflow that runs the unit tests and
   `python -c "import rsa_utils, searchlight"` (import smoke test) on push.

---

## Phase 1 — Documentation (low risk, high value)

The pipeline is already usable but the code does not explain itself. Most of this
is additive.

1. **Adopt one docstring standard (NumPy or Google style) across `rsa_utils.py`.**
   Priority functions — the ones each pipeline step calls — are currently the
   least documented:

   | Function | Current state |
   |---|---|
   | `apply_cluster_correction` | Empty `"""  """` docstring |
   | `calculate_z_map_real_data` | No docstring |
   | `calculate_z_maps_rnd` | No docstring (returns `True`) |
   | `calculate_voxelwise_rnd_distribution` | No docstring (returns `True`) |
   | `compare_with_model2` | One-line docstring only |
   | `create_tables` | No docstring |

   Each should document: purpose, every parameter, **the files it reads**, **the
   files it writes**, and **its return contract** (see Phase 2).

2. **Module-level header in `rsa_utils.py`.** A table of contents grouping the
   ~50 functions by concern (beta maps, pairwise similarity, model comparison,
   permutations, z-maps, clustering, tables, plotting, helpers). At 5000 lines a
   reader currently has no map.

3. **Cross-link the step docstring in `searchlight.py`.** The big comment block at
   the top (lines 25–74) is the best existing overview — convert it into a proper
   module docstring and reference `docs/searchlight_usage.md`.

4. **Document the file/naming conventions in one place.** Result paths and the
   parameter-encoded filenames (`{specie}-r-{radius}_{method}_{rsa_method}_zt{z}...`)
   are reconstructed independently inside many functions. Document the scheme and
   (Phase 3) centralize it.

---

## Phase 2 — Correctness & contracts (medium risk)

These address latent bugs and fragile contracts, several tied to the scheduler.

1. **Make the success/return contract explicit and uniform.** The scheduler's
   completion markers depend on step functions returning a truthy value, but the
   contract is inconsistent:
   - `searchlight.py` writes markers unconditionally for steps 0, 1, 2, 4, but
     gates steps 3, 5, 6, 7, 8, 9, 10 on a truthy `result`.
   - Some `rsa_utils` functions `return True` on success; others return `None`
     implicitly even when they succeed.

   **Action:** define the contract — *every* step function returns `True` on
   success and raises on failure — then audit each function against it, and make
   `searchlight.py` gate **all** markers on the return value uniformly.

2. **Fix the hardcoded Windows path in `create_tables`.**
   `res_folder = r"G:\My Drive\Results" + ...` (line ~5132) is a hardcoded
   Google-Drive path that breaks on Linux and on any other user's machine. Route
   it through `scheduler/paths.py` or make it a parameter with a sane default;
   skip the Drive copy when the target does not exist.

3. **Remove `importlib.reload()` from `searchlight.py`.** Lines 214–221 reload
   `utils`, `preprocess_functions`, and `rsa_utils` on every run. This is a
   notebook-development artifact; in a CLI it only adds risk (stale bytecode, import
   ordering surprises). Replace with normal top-level imports.

4. **Validate arguments early.** Fail fast with clear messages when: `--rsa_model`
   is missing for steps ≥ 2; a requested step's upstream outputs are absent;
   `--specie` is not `D`/`H` (note the typo `'`H'` in two `raise` messages).

5. **Normalize the step-7.5 special case.** The `if step == 75` block is a
   footgun (magic number standing in for "7.5"). Either document it prominently or
   replace it with an explicit `--real_z_only` flag.

---

## Phase 3 — De-duplication (medium risk, big payoff)

The module carries substantial duplicated / dead code. Removing it shrinks the
maintenance surface dramatically.

1. **Delete `rsa_utils - Copy.py` (2961 lines).** A stale backup checked into
   git. Version control already preserves history; remove it.

2. **Clarify the `*2` function pairs.** There are old/new versions of two
   functions:
   - `calculate_pairwise_similarity_maps` vs `calculate_pairwise_similarity_maps2`
   - `compare_with_model` vs `compare_with_model2`

   `searchlight.py` only calls the `2` variants — **but the non-`2` originals are
   not dead code**: `compare_with_model2` calls `compare_with_model` internally
   (rsa_utils.py:1791, 1816) and `calculate_pairwise_similarity_maps2` calls
   `calculate_pairwise_similarity_maps` (rsa_utils.py:3375). The `2` versions are
   orchestrators; the originals do the per-run work. **Action:** rename the pair to
   reflect the relationship (e.g. `..._maps` → `..._maps_single_run`, `..._maps2` →
   `..._maps`) and add docstrings stating who calls whom, rather than deleting.

3. **Unify beta-map computation.** `calculate_beta_maps` (dogs) and
   `calculate_beta_mapsH` (humans) likely share most logic. Extract the common
   core and branch only on the species-specific parts.

4. **Centralize path construction.** The
   `{specie}-r-{radius}_{method}_{rsa_method}_...` filename pattern is rebuilt in
   `apply_cluster_correction`, `calculate_z_map_real_data`, `create_tables`,
   `calculate_cluster_size_distribution`, and others. Extract a single
   `rsa_paths.py` (or functions in `scheduler/paths.py`) that returns every
   canonical path from `(datafolder, dataset, specie, model, rsa_model, radius,
   method, rsa_method, z_threshold, mask_type)`. This removes the class of bug
   where two call sites disagree on a filename.

---

## Phase 4 — Structural refactor (higher risk, do last)

Only after Phases 0–3 land and the golden test is trusted.

1. **Split `rsa_utils.py` into a package.** A 5200-line module is hard to
   navigate and slow to import. Suggested split, preserving a re-export shim
   (`rsa_utils.py` that imports from the package) so no call sites break:

   ```
   rsa/
     beta_maps.py        # calculate_beta_maps, calculate_beta_mapsH
     similarity.py       # pairwise + mahalanobis + crossnobis + searchlight
     model_compare.py    # compare_with_model2, group model maps
     permutations.py     # rnd steps 4/5/6
     zmaps.py            # z-map real + rnd
     clustering.py       # cluster distribution, correction, peaks
     tables.py           # create_tables, clusters_to_table
     plotting.py         # plot_rsa_circle_matrix, viz helpers
     paths.py            # centralized path builder (from Phase 3)
     metrics.py          # kendall_tau_a, crossnobis, shuffle_vector, ...
   ```

2. **Introduce a config/params dataclass.** Every step function takes the same
   ~10 positional args (`datafolder, dataset, specie, model, rsa_model, radius,
   method, rsa_method, ...`). A frozen `RSAConfig` dataclass passed as a single
   object would cut signature noise, prevent argument-order mistakes, and make
   `searchlight.py`'s `main()` far shorter.

3. **Replace magic strings with enums/constants.** `method`, `rsa_method`,
   `specie`, `mask_type`, and `mah_fold` are validated ad hoc via `if` chains.
   Central constants (or `Enum`s) give one source of truth and early validation.

4. **Structured logging.** Replace the many `print(...)` calls with the `logging`
   module so verbosity is controllable and the scheduler can capture per-step logs
   cleanly.

---

## Phase 5 — Nice-to-haves

- **Type hints** on public functions (a few already use `typing`); enables
  static checking with `mypy`/`pyright`.
- **`pyproject.toml`** with pinned dependencies (`numpy`, `nibabel`, `pandas`,
  `pyyaml`, `scipy`, `scikit-learn`) so environments are reproducible.
- **Progress/ETA reporting** for the expensive steps (1, 4, 5) via `tqdm`.
- **Remove committed binary artifacts** from the repo root
  (`-mul.nii.gz`, `odin.jpg`, `UserInterface.ai`, `.mmap`) — move to an assets
  folder or Git LFS to keep clones lean.

---

## Suggested sequencing

| Order | Phase | Rationale |
|---|---|---|
| 1 | Phase 0 (tests) | Safety net for everything else. |
| 2 | Phase 1 (docs) | Zero behaviour risk; immediate readability win. |
| 3 | Phase 3.1–3.2 (delete dead code) | Shrinks surface before deeper work. |
| 4 | Phase 2 (contracts + hardcoded path) | Fixes real bugs; scheduler reliability. |
| 5 | Phase 3.3–3.4 (unify + centralize paths) | Removes duplication safely. |
| 6 | Phase 4 (structural) | Largest change; needs the test net in place. |
| 7 | Phase 5 (polish) | Incremental, ongoing. |

Each phase can ship as its own PR. Phases 0–2 are safe to start immediately;
Phase 4 should wait until the golden regression test from Phase 0 is trusted.
