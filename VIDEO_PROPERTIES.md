# Stimulus video properties

Open `stimulus_video_properties.ipynb` in Jupyter to run or review the analysis.
The controls near the top provide `EMOTION_COLORS` (five pooled bars),
`GROUP_COLORS` (ten species/emotion bars), and `PROPERTY_COLORS` for individual
subplot overrides. Point, error-bar, and background colors are also editable.
Edit the palettes and Run All to regenerate plots using saved measurements.
Set `RECOMPUTE_MEASUREMENTS=True` to rescan and validate the frame inputs.
The same analysis can be run from PowerShell:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' stimulus_video_properties.py
```

Defaults use frames in `G:\My Drive\Networks\stimuli_by_frames`, source metadata
from `C:\github\Networks\paradigm\stimuli`, and results in
`outputs\video_properties` under this repository. Inputs are read only. All
240 named stimulus folders are measured, including A/F/H/N/P. Statistical
comparisons select P (positive anticipation) and H (happiness). F is fear.

The analysis inventories every input file, checks that frame indices are
contiguous from zero, decodes every image, verifies consistent dimensions,
and checks frame counts/dimensions against matching source videos. It records
SHA-256 hashes for every image and checks for byte-identical video sequences.
The 240 folders contain 160 unique clips: 16 per species/emotion group.
Measurements retain every folder; plot means, SEMs, and statistical tests count
each identical clip once. `unique_clip_membership.csv` records every mapping.
The `others` folder contains nonvideo control assets. OS metadata and the
landmark annotation CSV in R1DogH4 are inventoried and excluded from image
measurements; unexpected files cause an error.

## Measurements

The starting methods are
[calculateVideoValues.m](https://github.com/rhernandez00/Comparative-study-of-dog-vision/blob/main/videoSelection/calculateVideoValues.m)
and its helper
[calculateMotion.m](https://github.com/rhernandez00/Comparative-study-of-dog-vision/blob/main/videoSelection/calculateMotion.m).

| Outcome | Definition |
|---|---|
| Hue | Arithmetic pixel mean of HSV hue, then equal-weight mean across frames, as in MATLAB. Hue wraps at 0/1; this reproduces the original linear metric rather than a circular statistic. |
| Saturation | Pixel mean of HSV saturation, then mean across frames, on 0–1 scale. |
| Brightness | Mean uint8 grayscale intensity (0–255), using MATLAB rgb2gray coefficients and integer rounding. |
| RMS contrast | sqrt(mean(gray²)), then mean across frames. This is the original uncentered RMS definition and is affected by brightness. |
| Block motion total | Sum of squared block displacements across all blocks and N−1 transitions. Python reproduction of 35×35, three-step MSE matching, maximum displacement ±7, steps 4/2/1, no overlap. |
| Block motion per transition | Original block total divided by N−1, to expose the video-length dependence of the total. |
| Dense flow displacement | Spatial mean of Farneback flow magnitude, then temporal mean across transitions, in pixels of the resized image per frame. |
| Dense flow speed | Spatial/temporal mean flow magnitude ×100 / resized frame diagonal × source fps, in percent diagonal per second. |
| Frame count | Number of extracted images. |
| Duration | Frame count / source fps. |

The default output of MATLAB
[vision.BlockMatcher](https://www.mathworks.com/help/vision/ref/vision.blockmatcher-system-object.html)
is **magnitude squared**, and its default matching criterion is MSE. Therefore
the original helper sums squared displacements, not flow magnitudes.
The Python reproduction uses zero padding for partial edge blocks and a
center-first rule for equal-cost candidates. These edge/tie conventions have
not been validated against the proprietary MATLAB implementation. Block-motion
columns and report explicitly identify this limitation. Color definitions and
dense-flow parameters reproduce the supplied methods; the Python block matcher
should not be described as numerically identical to MATLAB.

Dense flow uses exactly the supplied local script's grayscale conversion,
INTER_AREA resizing (longest side ≤640), and Farneback parameters:
`pyr_scale=.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0`.
No z-scoring is applied when comparing physical stimulus properties.
Here timing comes from verified source-video fps. In the FSL script, speed
instead uses N/logged event duration. Those are equal only when the logged
duration equals N/fps. JPEG extraction also introduces recompression differences
relative to reading the original video directly.

## Statistics and figures

Each unique clip contributes one observation. Error bars are SEM across unique
clips, using sample SD (`ddof=1`). Frames and repeated presentations are not
treated as independent samples.
Both figures contain a subplot for each of the 10 outcomes, with individual
video points overlaid on the bars. All categories are plotted, while tests
remain restricted to P/H:

- `properties_all_categories_pooled`: A, F, N, P, H, combining Dog and Hum.
- `properties_all_categories_by_species`: all ten Dog/Hum × A/F/N/P/H groups.

Two-sided Welch two-sample tests compare pooled P/H and all six pairs of the
four groups, including P/H within each species. CSVs contain test statistics,
Welch degrees of freedom, sample sizes, means/SEMs, P−H or group-A−group-B
differences, 95% difference confidence intervals, Hedges g, and raw/corrected p.

The two-way ANOVA tests species, emotion, and species × emotion interaction,
using type III sums of squares with sum contrasts. A separate model adds run
as a fixed nuisance effect, using the first run containing each unique clip.
Main plots show raw ANOVA p-values; the CSVs/report
also include BH correction across the 10 outcomes per term and across all 30
ANOVA tests. Pooled tests use BH across 10 outcomes. Four-group comparisons
include Holm correction across six pairs per outcome and BH across all 60 tests.
Levene variance and residual Shapiro normality diagnostics are saved.

Distinct clips are assumed independent. Exact duplicate sequences are collapsed
by hash for descriptive summaries and inference, while all folder-level
measurements are retained. The representative is the first occurrence sorted by
run and video ID. Identical clips with conflicting species/emotion labels or
properties/timing cause an error. Byte hashes cannot identify different clips of the same
actor/source recording: if those are shared, a grouping metadata file would be
needed for an additional clustered analysis. Raw-RMS contrast and movement totals
should be interpreted with their brightness and duration dependencies in mind.

## Outputs and rerunning

`REPORT.md` in the output folder contains the results and plot previews.
Figures are exported as PNG, PDF, and SVG. All five emotions appear in
`video_properties.csv`; every frame and transition appear in their respective
CSVs. `input_file_inventory.csv`, `condition_counts.csv`, and `provenance.json`
record coverage, exclusions, timing, parameters, and source/version hashes.
`plot_settings.json` records the palettes and per-subplot overrides used for
the current figure exports. `group_summary.csv` contains means, SDs, and SEMs
for all five pooled emotions and all ten species/emotion groups.

Successful video measurements are cached individually and reused only when the
measurement source hash, parameters, library versions, timing metadata, frame file
sizes, and modification times still match. Use `--force` to recompute regardless
of the cache. Partial failures are reported and do not silently enter inference.

```powershell
# Recompute statistics and plots without reading the frames again:
& 'C:\ProgramData\anaconda3\python.exe' stimulus_video_properties.py --statistics-only

# Example custom locations / parallelism:
& 'C:\ProgramData\anaconda3\python.exe' stimulus_video_properties.py `
  --frames 'G:\My Drive\Networks\stimuli_by_frames' `
  --videos 'C:\github\Networks\paradigm\stimuli' --workers 4

# Verify scientific computations and agreement with the supplied dense-flow script:
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
& 'C:\ProgramData\anaconda3\python.exe' -m pytest test_stimulus_video_properties.py -q
```

`--max-dimension 0` disables resizing for dense flow. Color and block motion
always use stored full-resolution images. If source videos are unavailable,
`--fps` accepts an explicitly verified extraction frame rate; no frame rate is
guessed. Dependencies are in `requirements-video-properties.txt`.
