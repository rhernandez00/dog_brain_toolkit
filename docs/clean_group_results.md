# Recalculate group RSA after adding participants

`tools/clean_group_results.py` clears the outputs of `searchlight.py` steps
**3 and 5–10**. Preview first (no files are changed):

```powershell
& "C:\ProgramData\anaconda3\python.exe" tools/clean_group_results.py `
    --dataset EmoC --model basic-block --specie H `
    --rsa_model action_tendency__all
```

Repeat with `--apply` to delete the listed files. Use `--rsa_model name1 name2`
for several models, or replace that option with `--all` to discover all model
folders for the chosen GLM model. `--specie D H` selects both species.
`--datafolder` overrides the usual data disk. `--dis_method` defaults to
`mahalanobis`; `--rsa_method` defaults to `kendall`. Run separately for other
method pairs, such as `--dis_method correlation`.

The cleanup includes real group means/stds and provenance, group permutation
means/stds and z-maps, the voxelwise null mean/std/log, cluster-size caches/logs,
corrected maps/sidecars, CSVs and publication reports. It covers **all masks,
radii and z thresholds** for the selected species/model/method pair: step 6's
null cache does not encode masks, radius or methods, and step 8's cache does not
encode masks or thresholds. This also clears old permutations above a newly
reduced repetition count. The script preserves participant folders, including
steps 1, 2, 4, 4.5 and 7.6, and other analysis families such as regression RSA.

Stop workers before applying cleanup. Finish steps 1, 2 and 4 for the added
participant, and ensure **every participant's step-4 permutation maps are still
available** before rerunning step 5. If step 7.6 purged them, regenerate step 4
or restore the original maps first. Then rerun `searchlight.py` with the same
analysis parameters and `--steps_to_run 3 5 6 7 8 9 10`.

Queue records are untouched: enqueue fresh jobs after cleanup. If using an
explicit dedicated `--job_marker_dir`, passing that directory also clears only
the `3.done`, `5.done` through `10.done` markers. Participant markers stay.

Google Drive report mirrors are preserved by default. To clear those too, add
`--mirror_root "G:\My Drive\Results"`. Review that preview carefully: mirrored
filenames omit the GLM model, radius and methods, so matching reports may be
shared by other analysis settings. This option covers legacy flat mirrors and
the current species/mask folders. Step 10 recreates the selected reports.
