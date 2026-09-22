# Colab searchlight handoff

This repository has an active Google Colab workflow for the searchlight RSA
pipeline. This note is the starting context for a later chat about that work.

## Regression RSA: EmoB

The current Colab regression workflow runs searchlight steps **15**, **15.3**,
**15.4**, and **15.5** for EmoB. It defaults to humans (`SPECIE = 'H'`) and
also supports dogs (`SPECIE = 'D'`). Step 15 fits the real target RDM while
controlling for the RDMs listed by `regression_model` (currently `visual_3`:
`visual1`, `visual2`, and `flow`). Step 15.4 permutes the target model only;
the neural pairwise maps and control models remain real and fixed.

The Drive folder is `G:\My Drive\rsa_colab`. Its relevant contents are:

| Item | Purpose |
|---|---|
| `pkg_EmoB/` | Existing participant input packages. |
| `regression_support_EmoB_visual_3.zip` | Controls, masks, selected target RDMs, and GPU regression code. |
| `colab_rsa_regression.ipynb` | GPU notebook for steps 15, 15.3, 15.4, and 15.5. |
| `results_regression_EmoB/` | Regression participant, checkpoint, and completed group result ZIPs. |
| `regression_inference_support_EmoB.zip` | Independent support package for downstream inference. |
| `colab_rsa_regression_inference.ipynb` | Continuation notebook for steps 15.6–15.10. |
| `results_regression_inference_EmoB/` | Outputs from steps 15.6–15.10. |

The first notebook is resumable: it checkpoints every completed target/run and
writes one completed group ZIP per target/species:
`result_regression_group_<target>_<species>.zip`. The continuation notebook
requires those completed group ZIPs; participant ZIPs alone are insufficient.

The batch settings are performance/memory limits, not analysis parameters:

| Setting | Used for |
|---|---|
| `VOXEL_BATCH` | Number of voxels fitted together in steps 15/15.4; also bounds group averaging chunks. |
| `PERMUTATION_BATCH` | Number of target-model permutation designs fitted together in step 15.4. |
| `GROUP_BATCH` | Number of group permutation averages handled together in step 15.5. |
| `STEP1_BATCH` | Searchlight centres processed together only when step-1 pairwise maps must be computed from beta maps. |

For an assigned T4, begin with `VOXEL_BATCH = 4096`,
`PERMUTATION_BATCH = 16`, `GROUP_BATCH = 16`, and `STEP1_BATCH = 256`.
Increase one of the first two settings at a time only if GPU memory permits.

## Downstream regression inference

Steps **15.6–15.10** consume the real group beta mean from step 15.3 and the
permuted group beta means from step 15.5. They create null mean/std maps,
z-maps, cluster null distributions, corrected maps, and atlas-labelled CSV
reports. The continuation notebook uses the GPU only for the float64 streaming
null mean/std calculation; NIfTI I/O, z-map writing, clustering, correction,
and reports intentionally use the existing CPU implementation so their
statistical behaviour matches `rsa_utils.calculate_regression_inference`.

The authoritative implementation and package-building instructions are in
[`tools/colab_gpu/README.md`](../tools/colab_gpu/README.md). The code involved
is `tools/colab_gpu/run_colab_regression.py`,
`tools/colab_gpu/run_colab_regression_inference.py`,
`tools/create_regression_package.py`, and
`tools/create_regression_inference_package.py`.
