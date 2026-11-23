"""Measure every extracted frame and compare stimulus properties at video level.

See VIDEO_PROPERTIES.md for definitions, provenance, and the statistical design.
No stimulus files or source videos are modified. Run this module with --help.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import inspect
import itertools
import json
import math
from pathlib import Path
import re
import sys
import time

import cv2
import numpy as np
import pandas as pd
from numba import njit
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

DEFAULT_FRAMES = Path(r"G:\My Drive\Networks\stimuli_by_frames")
DEFAULT_VIDEOS = Path(r"C:\github\Networks\paradigm\stimuli")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs" / "video_properties"
VIDEO_ID = re.compile(r"R(?P<run>\d+)(?P<species>Dog|Hum)(?P<emotion>[AFHNP])(?P<number>\d+)$", re.I)
FRAME_ID = re.compile(r"frame_(\d+)$", re.I)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
EMOTIONS = {"A": "Anger", "F": "Fear", "H": "Happiness", "N": "Neutral", "P": "Positive anticipation"}
GROUPS = ["DogP", "DogH", "HumP", "HumH"]
EMOTION_ORDER = ["A", "F", "N", "P", "H"]
EMOTION_COLORS = {"A": "#c65b61", "F": "#8f79b8", "N": "#929da5", "P": "#4187bd", "H": "#dd9250"}
GROUP_COLORS = {"DogA": "#c65b61", "DogF": "#8f79b8", "DogN": "#929da5", "DogP": "#4187bd", "DogH": "#dd9250",
                "HumA": "#e49a9e", "HumF": "#baa7d7", "HumN": "#c1c9ce", "HumP": "#72b6c7", "HumH": "#e7bf68"}
# All inferential outcomes and units are explicit; no run-wise z-scoring.
PROPERTIES = {
    "hue": ("Mean hue", "HSV hue (0–1; arithmetic mean)"),
    "saturation": ("Mean saturation", "HSV saturation (0–1)"),
    "brightness": ("Mean brightness", "Grayscale intensity (0–255)"),
    "contrast_rms": ("RMS contrast (original definition)", "Raw grayscale RMS (0–255)"),
    "block_motion_total": ("Block motion total (Python TSS)", "Sum of squared displacement (px²)"),
    "block_motion_mean_per_transition": ("Block motion/transition (Python TSS)", "Mean summed squared displacement (px²)"),
    "dense_flow_pixels_per_frame": ("Dense optical flow: displacement", "Mean magnitude (resized px/frame)"),
    "dense_flow_percent_diagonal_per_second": ("Dense optical flow: speed", "Mean speed (% frame diagonal/s)"),
    "number_of_frames": ("Number of frames", "Frames/video"),
    "duration_seconds": ("Video duration", "Seconds (N / source fps)"),
}
METHOD_VERSION = 1


def parse_video_id(name: str) -> dict:
    match = VIDEO_ID.fullmatch(name)
    if not match:
        raise ValueError(f"Unrecognized video folder: {name}")
    values = match.groupdict()
    species = "Dog" if values["species"].lower() == "dog" else "Hum"
    emotion = values["emotion"].upper()
    return {"video_id": name, "run": int(values["run"]), "species": species,
            "emotion": emotion, "emotion_name": EMOTIONS[emotion],
            "video_number": int(values["number"]), "group": species + emotion}


def inventory(root: Path) -> tuple[list[dict], pd.DataFrame]:
    """Enumerate every file recursively; reject unknown/missing/duplicate frames."""
    if not root.is_dir():
        raise NotADirectoryError(root)
    records, audit, by_id = [], [], {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        metadata = path.name.lower() in {"desktop.ini", ".ds_store", "thumbs.db"}
        if metadata:
            status = "excluded_os_metadata"
        elif relative.parts[0].lower() == "others":
            status = "excluded_nonvideo_control_asset"
        elif len(relative.parts) == 2 and VIDEO_ID.fullmatch(relative.parts[0]):
            if path.suffix.lower() == ".csv" and path.stem == relative.parts[0]:
                with path.open(encoding="utf-8-sig") as handle:
                    header = [handle.readline().split(",")[0] for _ in range(3)]
                if header != ["scorer", "bodyparts", "coords"]:
                    raise ValueError(f"Unrecognized CSV sidecar: {path}")
                status = "excluded_landmark_annotation_csv"
            elif path.suffix.lower() not in IMAGE_EXTENSIONS or not FRAME_ID.fullmatch(path.stem):
                raise ValueError(f"Unexpected file in video folder: {path}")
            else:
                status = "included_video_frame"
                by_id.setdefault(relative.parts[0], []).append(path)
        else:
            raise ValueError(f"Unclassified input file (review before proceeding): {path}")
        stat = path.stat()
        audit.append({"relative_path": str(relative), "status": status,
                      "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and VIDEO_ID.fullmatch(p.name)):
        frames = sorted(by_id.get(folder.name, []), key=lambda p: int(FRAME_ID.fullmatch(p.stem)[1]))
        indices = [int(FRAME_ID.fullmatch(p.stem)[1]) for p in frames]
        if len(indices) < 2 or indices != list(range(len(indices))):
            raise ValueError(f"Frames must be unique and contiguous from zero, with N>=2: {folder}")
        records.append({**parse_video_id(folder.name), "folder": str(folder),
                        "frames": [str(p) for p in frames]})
    if not records:
        raise ValueError(f"No named stimulus frame folders under {root}")
    return records, pd.DataFrame(audit)


def source_metadata(video_root: Path, records: list[dict], fps_override: float | None) -> None:
    index = {}
    if video_root.is_dir():
        for path in video_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS and VIDEO_ID.fullmatch(path.stem):
                key = path.stem.lower()
                if key in index:
                    raise ValueError(f"Ambiguous source video: {index[key]} and {path}")
                index[key] = path
    for record in records:
        path = index.get(record["video_id"].lower())
        info = {"source_video": str(path) if path else "", "source_frame_count": None,
                "source_width": None, "source_height": None}
        if path:
            cap = cv2.VideoCapture(str(path))
            try:
                if not cap.isOpened():
                    raise RuntimeError(f"Cannot open source metadata: {path}")
                source_fps = float(cap.get(cv2.CAP_PROP_FPS))
                info.update(source_frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                            source_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            source_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                if info["source_frame_count"] != len(record["frames"]):
                    raise ValueError(f"Extracted/source frame-count mismatch for {path.name}: "
                                     f"{len(record['frames'])} vs {info['source_frame_count']}")
                stat = path.stat()
                info.update(source_bytes=stat.st_size, source_mtime_ns=stat.st_mtime_ns)
            finally:
                cap.release()
        elif fps_override is None:
            raise ValueError(f"No source video for {record['video_id']}; provide a verified --fps")
        else:
            source_fps = fps_override
        info["fps"] = fps_override if fps_override is not None else source_fps
        info["timing_source"] = "explicit_fps_override" if fps_override is not None else "original_video_metadata"
        if not math.isfinite(info["fps"]) or info["fps"] <= 0:
            raise ValueError(f"Invalid fps for {record['video_id']}: {info['fps']}")
        record.update(info)


@njit(nogil=True, cache=True)
def color_properties(bgr: np.ndarray) -> tuple:
    """MATLAB rgb2hsv arithmetic means and rounded uint8 rgb2gray / raw RMS."""
    sum_h = sum_s = sum_g = sum_g2 = 0.0
    gray = np.empty(bgr.shape[:2], np.uint8)
    for y in range(bgr.shape[0]):
        for x in range(bgr.shape[1]):
            b, g, r = float(bgr[y, x, 0]), float(bgr[y, x, 1]), float(bgr[y, x, 2])
            hi, lo = max(r, g, b), min(r, g, b)
            delta = hi - lo
            hue = 0.0
            if delta > 0:
                if hi == r:
                    hue = ((g - b) / delta) % 6.0
                elif hi == g:
                    hue = (b - r) / delta + 2.0
                else:
                    hue = (r - g) / delta + 4.0
                hue /= 6.0
            sum_h += hue
            sum_s += delta / hi if hi > 0 else 0.0
            # MATLAB rgb2gray coefficients (not OpenCV's 0.299/0.587/0.114).
            value = math.floor(0.298936021293775 * r + 0.587043074451121 * g +
                               0.114020904255103 * b + 0.5)
            gray[y, x] = value
            sum_g += value
            sum_g2 += value * value
    n = bgr.shape[0] * bgr.shape[1]
    return sum_h / n, sum_s / n, sum_g / n, math.sqrt(sum_g2 / n), gray


@njit(nogil=True, cache=True)
def block_motion_three_step(previous: np.ndarray, current: np.ndarray) -> float:
    """Python reproduction of 35x35, MSE, +/-7 px, steps 4/2/1, squared output.

    Include partial edge blocks with zero padding. Ties prefer the current
    search center. MATLAB's proprietary edge/tie details are not validated.
    """
    height, width = previous.shape
    block = 35
    total = 0.0
    for y in range(0, height, block):
        for x in range(0, width, block):
            cy = cx = 0
            for step in (4, 2, 1):
                best_error = np.inf
                best_y, best_x = cy, cx
                # Center first prevents spurious motion on constant frames.
                for candidate in range(9):
                    if candidate == 0:
                        dy, dx = cy, cx
                    else:
                        q = candidate - 1
                        gy, gx = q // 3 - 1, q % 3 - 1
                        if q >= 4:
                            q += 1
                            gy, gx = q // 3 - 1, q % 3 - 1
                        dy, dx = cy + gy * step, cx + gx * step
                    if abs(dy) > 7 or abs(dx) > 7:
                        continue
                    error = 0.0
                    for by in range(block):
                        sy, ty = y + by, y + by + dy
                        for bx in range(block):
                            sx, tx = x + bx, x + bx + dx
                            a = float(previous[sy, sx]) if sy < height and sx < width else 0.0
                            b = float(current[ty, tx]) if 0 <= ty < height and 0 <= tx < width else 0.0
                            error += (a - b) ** 2
                        # All costs are nonnegative; stop a losing candidate.
                        if error > best_error:
                            break
                    if error < best_error:
                        best_error, best_y, best_x = error, dy, dx
                cy, cx = best_y, best_x
            total += cy * cy + cx * cx
    return total


def resize_flow_gray(gray: np.ndarray, max_dimension: int) -> np.ndarray:
    h, w = gray.shape
    if max_dimension == 0 or max(h, w) <= max_dimension:
        return gray
    scale = max_dimension / max(h, w)
    return cv2.resize(gray, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)


def measure_video(record: dict, max_dimension: int) -> dict:
    color_rows, flow_rows, frame_rows = [], [], []
    previous_block = previous_flow = None
    sequence_hash = hashlib.sha256()
    shape = None
    for frame_index, filename in enumerate(record["frames"]):
        path = Path(filename)
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        sequence_hash.update(bytes.fromhex(digest))
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.dtype != np.uint8:
            raise ValueError(f"Unreadable/non-uint8 image: {path}")
        if shape is None:
            shape = frame.shape
            if record.get("source_width") and (shape[1] != record["source_width"] or shape[0] != record["source_height"]):
                raise ValueError(f"Extracted/source dimensions differ for {path}")
        if frame.shape != shape:
            raise ValueError(f"Frame dimensions changed in {record['video_id']}: {path}")
        hue, saturation, brightness, contrast, gray_block = color_properties(frame)
        color_rows.append([hue, saturation, brightness, contrast])
        # Precisely the grayscale conversion and resizing used in the supplied script.
        gray_flow = resize_flow_gray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), max_dimension)
        frame_rows.append({"video_id": record["video_id"], "frame_index": frame_index,
                           "relative_path": str(path.parent.name + "/" + path.name),
                           "sha256": digest, "width": shape[1], "height": shape[0],
                           "hue": hue, "saturation": saturation, "brightness": brightness,
                           "contrast_rms": contrast})
        if previous_flow is not None:
            flow = cv2.calcOpticalFlowFarneback(previous_flow, gray_flow, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude = np.hypot(flow[..., 0], flow[..., 1])
            pixels = float(np.mean(magnitude))
            displacement = 100.0 * pixels / math.hypot(*gray_flow.shape)
            block_value = float(block_motion_three_step(previous_block, gray_block))
            if not all(math.isfinite(v) for v in [pixels, block_value]):
                raise ValueError(f"Nonfinite motion in {path}")
            flow_rows.append({"video_id": record["video_id"], "from_frame": frame_index - 1,
                              "to_frame": frame_index, "block_motion_sum_squared": block_value,
                              "dense_flow_pixels_per_frame": pixels,
                              "dense_flow_percent_diagonal_per_frame": displacement,
                              "dense_flow_percent_diagonal_per_second": displacement * record["fps"]})
        previous_block, previous_flow = gray_block, gray_flow
    colors = np.mean(color_rows, axis=0)
    summary = {k: v for k, v in record.items() if k not in {"frames", "folder"}}
    summary.update(zip(["hue", "saturation", "brightness", "contrast_rms"], colors.tolist()))
    summary.update(number_of_frames=len(frame_rows), number_of_transitions=len(flow_rows),
                   duration_seconds=len(frame_rows) / record["fps"], width=shape[1], height=shape[0],
                   flow_width=gray_flow.shape[1], flow_height=gray_flow.shape[0],
                   sequence_sha256=sequence_hash.hexdigest(),
                   block_motion_total=math.fsum(row["block_motion_sum_squared"] for row in flow_rows),
                   block_motion_mean_per_transition=float(np.mean([row["block_motion_sum_squared"] for row in flow_rows])))
    for prop in ["dense_flow_pixels_per_frame", "dense_flow_percent_diagonal_per_second"]:
        summary[prop] = float(np.mean([row[prop] for row in flow_rows]))
    return {"summary": summary, "frames": frame_rows, "transitions": flow_rows}


def cache_key(record: dict, max_dimension: int) -> str:
    measurement_functions = [parse_video_id, inventory, source_metadata, color_properties,
                             block_motion_three_step, resize_flow_gray, measure_video]
    measurement_source = "\n".join(inspect.getsource(func) for func in measurement_functions)
    config = {"version": METHOD_VERSION, "max_dimension": max_dimension,
              "measurement_sha256": hashlib.sha256(measurement_source.encode()).hexdigest(),
              "opencv": cv2.__version__, "numpy": np.__version__, "record": record,
              "frame_stats": [(Path(f).stat().st_size, Path(f).stat().st_mtime_ns) for f in record["frames"]]}
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def cached_measure(record: dict, max_dimension: int, cache_dir: Path, force: bool) -> tuple[dict, bool]:
    key = cache_key(record, max_dimension)
    path = cache_dir / (record["video_id"] + ".json")
    if not force and path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("key") == key:
            return cached["result"], True
    result = measure_video(record, max_dimension)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"key": key, "result": result}), encoding="utf-8")
    temporary.replace(path)
    return result, False


def analyze_dataset(frames_root: Path = DEFAULT_FRAMES, video_root: Path = DEFAULT_VIDEOS,
                    output: Path = DEFAULT_OUTPUT, *, workers: int = 4,
                    max_dimension: int = 640, fps: float | None = None, force: bool = False) -> pd.DataFrame:
    if workers < 1 or max_dimension < 0 or (fps is not None and (not math.isfinite(fps) or fps <= 0)):
        raise ValueError("workers >= 1, max_dimension >= 0, fps finite and > 0 required")
    output.mkdir(parents=True, exist_ok=True)
    records, audit = inventory(frames_root)
    audit.to_csv(output / "input_file_inventory.csv", index=False)
    source_metadata(video_root, records, fps)
    cache_dir = output / "cache"
    cache_dir.mkdir(exist_ok=True)
    cv2.setNumThreads(1)
    # Compile once before dispatching threads. Numba loops release the GIL.
    dummy = np.zeros((35, 35, 3), np.uint8)
    color_properties(dummy)
    block_motion_three_step(dummy[..., 0].copy(), dummy[..., 0].copy())
    print(f"Measuring {len(records)} videos / {(audit.status == 'included_video_frame').sum():,} frames; {workers} workers", flush=True)
    start = time.monotonic()
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        tasks = {executor.submit(cached_measure, rec, max_dimension, cache_dir, force): rec for rec in records}
        for done, future in enumerate(as_completed(tasks), 1):
            rec = tasks[future]
            try:
                result, hit = future.result()
                results.append(result)
                print(f"[{done}/{len(records)}] {rec['video_id']} {'cached' if hit else 'measured'} "
                      f"({time.monotonic() - start:.0f}s)", flush=True)
            except Exception as exc:
                errors.append({"video_id": rec["video_id"], "error": str(exc)})
                print(f"ERROR {rec['video_id']}: {exc}", flush=True)
    if errors:
        pd.DataFrame(errors).to_csv(output / "measurement_errors.csv", index=False)
        raise RuntimeError(f"{len(errors)} video(s) failed; inspect measurement_errors.csv. Successful videos cached; no inference run.")
    results.sort(key=lambda result: (result["summary"]["run"], result["summary"]["species"],
                                    result["summary"]["emotion"], result["summary"]["video_number"]))
    videos = pd.DataFrame([result["summary"] for result in results])
    videos.to_csv(output / "video_properties.csv", index=False)
    pd.DataFrame(itertools.chain.from_iterable(result["frames"] for result in results)).to_csv(output / "frame_properties.csv", index=False)
    pd.DataFrame(itertools.chain.from_iterable(result["transitions"] for result in results)).to_csv(output / "transition_motion.csv", index=False)
    videos.groupby(["run", "species", "emotion"], sort=True).size().rename("n_videos").to_csv(output / "condition_counts.csv")
    duplicates = videos[videos.sequence_sha256.duplicated(keep=False)]
    duplicates.to_csv(output / "duplicate_sequences.csv", index=False)
    unique = independent_videos(videos)
    representatives = unique.set_index("sequence_sha256").video_id
    membership = videos[["video_id", "run", "species", "emotion", "sequence_sha256"]].copy()
    membership["representative_video_id"] = membership.sequence_sha256.map(representatives)
    membership["included_as_unique_clip"] = membership.video_id == membership.representative_video_id
    membership.to_csv(output / "unique_clip_membership.csv", index=False)
    provenance = {"frames_root": str(frames_root.resolve()), "video_root": str(video_root.resolve()),
                  "output": str(output.resolve()), "method_version": METHOD_VERSION,
                  "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "n_videos": len(videos), "n_frames": int(videos.number_of_frames.sum()),
                  "n_unique_clips": len(unique), "duplicate_policy": "Retain all folder measurements; one observation per unique byte-identical frame sequence for plots and tests",
                  "n_transitions": int(videos.number_of_transitions.sum()),
                  "max_flow_dimension": max_dimension, "workers": workers, "fps_override": fps,
                  "opencv_version": cv2.__version__, "numpy_version": np.__version__,
                  "python": sys.version, "farneback_parameters": [0.5, 3, 15, 3, 5, 1.2, 0],
                  "block_match": "Python TSS reproduction: 35x35, MSE, +/-7, steps 4/2/1, zero-padded edges, center-first ties; not MATLAB-validated",
                  "matlab_source": "https://github.com/rhernandez00/Comparative-study-of-dog-vision/blob/main/videoSelection/calculateVideoValues.m",
                  "matlab_color_source_blob": "792eea55e7d44d30516a21e80f03024fa4da02d3",
                  "matlab_motion_source_blob": "03ee3b07092de9f476c111e7c1a1ab9024f22679"}
    flow_source = Path(r"C:\github\Networks\dense_optical_flow_to_fsl.py")
    if flow_source.is_file():
        provenance["dense_flow_source_sha256"] = hashlib.sha256(flow_source.read_bytes()).hexdigest()
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return videos


def independent_videos(videos: pd.DataFrame) -> pd.DataFrame:
    """Keep the first occurrence of each identical clip, preserving all categories."""
    if videos.video_id.duplicated().any():
        raise ValueError("Video IDs must be unique")
    if "sequence_sha256" not in videos:
        return videos.copy()
    if videos.sequence_sha256.isna().any():
        raise ValueError("Missing sequence hashes: cannot identify independent clips")
    for digest, group in videos.groupby("sequence_sha256", sort=False):
        if len(group) > 1:
            if len(group[["species", "emotion"]].drop_duplicates()) != 1:
                raise ValueError(f"Identical clip assigned conflicting species/emotion labels: {group.video_id.tolist()}")
            if not np.allclose(group[list(PROPERTIES)].to_numpy(float), group[list(PROPERTIES)].iloc[0].to_numpy(float), rtol=1e-10, atol=1e-10):
                raise ValueError(f"Identical clip has inconsistent properties/timing: {group.video_id.tolist()}")
    return videos.sort_values(["run", "video_id"], kind="stable").drop_duplicates("sequence_sha256").copy()


def adjust_pvalues(table: pd.DataFrame, column: str, destination: str, method: str = "fdr_bh") -> None:
    table[destination] = np.nan
    valid = np.isfinite(table[column])
    if valid.any():
        table.loc[valid, destination] = multipletests(table.loc[valid, column], method=method)[1]


def welch_comparison(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if min(len(a), len(b)) < 2 or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Welch comparisons require >=2 finite, independent videos per group")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    se2 = va / len(a) + vb / len(b)
    if se2 <= 0:
        raise ValueError("Welch test is undefined when both groups have zero variance")
    df = se2**2 / ((va / len(a))**2 / (len(a)-1) + (vb / len(b))**2 / (len(b)-1))
    test = stats.ttest_ind(a, b, equal_var=False)
    difference = float(np.mean(a) - np.mean(b))
    half = stats.t.ppf(.975, df) * np.sqrt(se2)
    pooled_sd = np.sqrt(((len(a)-1)*va + (len(b)-1)*vb) / (len(a)+len(b)-2))
    correction = 1 - 3 / (4 * (len(a) + len(b)) - 9)
    return {"n_a": len(a), "n_b": len(b), "mean_a": np.mean(a), "mean_b": np.mean(b),
            "sem_a": stats.sem(a), "sem_b": stats.sem(b), "difference_a_minus_b": difference,
            "difference_ci95_low": difference-half, "difference_ci95_high": difference+half,
            "t": float(test.statistic), "df": df, "p": float(test.pvalue),
            "hedges_g": correction * difference / pooled_sd}


def statistical_analysis(videos: pd.DataFrame, output: Path = DEFAULT_OUTPUT) -> dict[str, pd.DataFrame]:
    output.mkdir(parents=True, exist_ok=True)
    videos = independent_videos(videos)
    data = videos[videos.emotion.isin(["P", "H"])].copy()
    if set(data.group.unique()) != set(GROUPS) or data.video_id.duplicated().any():
        raise ValueError("Need unique videos in all four groups DogP, DogH, HumP, HumH")
    if not np.isfinite(data[list(PROPERTIES)].to_numpy(float)).all():
        raise ValueError("Nonfinite properties in the inferential dataset")
    summaries = []
    for grouping, frame in [("pooled", videos.assign(plot_group=videos.emotion)),
                            ("by_species", videos.assign(plot_group=videos.group))]:
        for group, group_data in frame.groupby("plot_group", sort=False):
            for prop in PROPERTIES:
                values = group_data[prop].to_numpy(float)
                summaries.append({"comparison": grouping, "group": group, "property": prop,
                                  "n": len(values), "mean": np.mean(values),
                                  "sd": np.std(values, ddof=1), "sem": stats.sem(values)})
    pooled, pairwise, anova, run_adjusted, diagnostics = [], [], [], [], []
    for prop in PROPERTIES:
        pooled.append({"property": prop, "group_a": "P", "group_b": "H",
                       **welch_comparison(data.loc[data.emotion == "P", prop].to_numpy(), data.loc[data.emotion == "H", prop].to_numpy())})
        for group_a, group_b in itertools.combinations(GROUPS, 2):
            pairwise.append({"property": prop, "group_a": group_a, "group_b": group_b,
                             **welch_comparison(data.loc[data.group == group_a, prop].to_numpy(), data.loc[data.group == group_b, prop].to_numpy())})
        for adjusted, destination in [(False, anova), (True, run_adjusted)]:
            formula = f"{prop} ~ C(species, Sum) * C(emotion, Sum)" + (" + C(run, Sum)" if adjusted else "")
            fit = ols(formula, data=data).fit()
            table = anova_lm(fit, typ=3)
            for term in ["C(species, Sum)", "C(emotion, Sum)", "C(species, Sum):C(emotion, Sum)"]:
                row = table.loc[term]
                destination.append({"property": prop, "term": {"C(species, Sum)": "species", "C(emotion, Sum)": "emotion", "C(species, Sum):C(emotion, Sum)": "species:emotion"}[term],
                                    "sum_sq": row.sum_sq, "df": row.df, "F": row.F,
                                    "p": row["PR(>F)"], "residual_df": fit.df_resid,
                                    "partial_eta_squared": row.sum_sq / (row.sum_sq + table.loc["Residual", "sum_sq"])})
            if not adjusted:
                levene = stats.levene(*(data.loc[data.group == group, prop] for group in GROUPS), center="median")
                shapiro = stats.shapiro(fit.resid)
                diagnostics.append({"property": prop, "levene_statistic": levene.statistic,
                                    "levene_p": levene.pvalue, "shapiro_residual_W": shapiro.statistic,
                                    "shapiro_residual_p": shapiro.pvalue})
    tables = {"group_summary": pd.DataFrame(summaries), "ttest_pooled_P_vs_H": pd.DataFrame(pooled),
              "ttest_four_groups": pd.DataFrame(pairwise), "anova_species_emotion": pd.DataFrame(anova),
              "anova_run_adjusted": pd.DataFrame(run_adjusted), "anova_diagnostics": pd.DataFrame(diagnostics)}
    adjust_pvalues(tables["ttest_pooled_P_vs_H"], "p", "p_fdr_bh")
    pair = tables["ttest_four_groups"]
    adjust_pvalues(pair, "p", "p_fdr_bh_all")
    pair["p_holm_within_property"] = np.nan
    for prop, indices in pair.groupby("property").groups.items():
        pair.loc[indices, "p_holm_within_property"] = multipletests(pair.loc[indices, "p"], method="holm")[1]
    for name in ["anova_species_emotion", "anova_run_adjusted"]:
        table = tables[name]
        table["p_fdr_bh_within_term"] = np.nan
        for term, indices in table.groupby("term").groups.items():
            table.loc[indices, "p_fdr_bh_within_term"] = multipletests(table.loc[indices, "p"], method="fdr_bh")[1]
        adjust_pvalues(table, "p", "p_fdr_bh_all_terms")
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    return tables


def make_plots(videos: pd.DataFrame, tables: dict[str, pd.DataFrame], output: Path = DEFAULT_OUTPUT,
               *, emotion_colors: dict | None = None, group_colors: dict | None = None,
               property_colors: dict | None = None, point_color: str = "#202b33",
               error_color: str = "#303030", background_color: str = "white",
               font_size: float = 10, line_width: float = 0.8,
               bar_width: float = 0.155, figure_size: tuple = (10, 6),
               show_titles: bool = False, show_subtitles: bool = False,
               bar_gap: float = 0.3, show_legend: bool = False) -> list[Path]:
    """Plot the six selected properties for P/H comparisons.

    property_colors can override individual bars in any subplot, e.g.
    {'brightness': {'P': '#123456', 'DogP': '#654321'}}.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    data = independent_videos(videos)
    data = data[data.emotion.isin(["P", "H"])].copy()
    plot_properties = {key: value for key, value in PROPERTIES.items() if key not in {
        "block_motion_total", "block_motion_mean_per_transition", "number_of_frames",
        "dense_flow_percent_diagonal_per_second", "duration_seconds"}}
    pooled_colors = {**EMOTION_COLORS, **(emotion_colors or {})}
    species_colors = {**GROUP_COLORS, **(group_colors or {})}
    overrides = property_colors or {}
    all_groups = GROUPS
    if not set(all_groups).issubset(set(data.group)):
        raise ValueError("All-category plots require A/F/N/P/H videos in both species")
    output.mkdir(parents=True, exist_ok=True)
    (output / "plot_settings.json").write_text(json.dumps({
        "emotion_order": ["P", "H"], "plotted_properties": list(plot_properties), "emotion_colors": pooled_colors,
        "group_colors": species_colors, "property_colors": overrides,
        "point_color": point_color, "error_color": error_color,
        "background_color": background_color,
        "font_size": font_size, "line_width": line_width, "bar_width": bar_width,
        "figure_size": figure_size, "show_titles": show_titles, "show_subtitles": show_subtitles,
        "bar_gap": bar_gap, "show_legend": show_legend,
        "statistical_categories": ["P", "H"], "error_bars": "SEM across videos",
        "sampling_unit": "unique clip; exact repeated sequences counted once",
    }, indent=2), encoding="utf-8")
    paths = []
    with plt.rc_context({"font.size": font_size, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": font_size, "axes.labelsize": font_size,
                         "xtick.labelsize": font_size, "ytick.labelsize": font_size,
                         "axes.linewidth": line_width, "xtick.major.width": line_width,
                         "ytick.major.width": line_width, "grid.linewidth": line_width,
                         "savefig.facecolor": background_color,
                         "figure.facecolor": background_color, "axes.facecolor": background_color}):
        for mode, groups in [("by_species", all_groups)]:
            fig, axes = plt.subplots(2, 3, figsize=figure_size, constrained_layout=True)
            rng = np.random.default_rng(20261108)
            positions = np.arange(len(groups)) * (bar_width + bar_gap)
            for ax, (prop, (title, unit)) in zip(axes.flat, plot_properties.items()):
                means, sems = [], []
                for x, group in enumerate(groups):
                    x = positions[x]
                    subset = data.loc[(data.emotion if mode == "pooled" else data.group) == group, prop]
                    means.append(float(subset.mean()))
                    sems.append(float(stats.sem(subset)))
                    ax.scatter(x + rng.uniform(-bar_width*.35, bar_width*.35, len(subset)), subset, s=8, color=point_color, alpha=.4, zorder=3, linewidths=0)
                colors = pooled_colors if mode == "pooled" else species_colors
                ax.bar(positions, means, yerr=sems, capsize=3, width=bar_width,
                       color=[overrides.get(prop, {}).get(g, colors[g]) for g in groups],
                       error_kw={"ecolor": error_color, "elinewidth": line_width, "capthick": line_width},
                       linewidth=line_width, alpha=.85, edgecolor=background_color, zorder=2)
                if mode == "pooled":
                    ax.set_xticks([])
                else:
                    ax.set_xticks([(positions[0]+positions[1])/2, (positions[2]+positions[3])/2], ["dog", "human"])
                if mode == "by_species":
                    ax.axvline((positions[1]+positions[2])/2, color=error_color, linewidth=line_width, alpha=.25)
                simple_labels = {"hue": "Hue", "saturation": "Saturation",
                                 "brightness": "Brightness", "contrast_rms": "RMS intensity",
                                 "dense_flow_pixels_per_frame": "Movement (optical flow)"}
                ax.set_ylabel(simple_labels.get(prop, unit))
                ax.grid(axis="y", alpha=.18, zorder=0)
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 5), useMathText=True)
                if mode == "pooled":
                    test = tables["ttest_pooled_P_vs_H"].set_index("property").loc[prop]
                    annotation = f"P vs H only: Welch p={test.p:.3g}; BH q={test.p_fdr_bh:.3g}"
                else:
                    a = tables["anova_species_emotion"]
                    a = a[a.property == prop].set_index("term")
                    annotation = "P/H only — ANOVA p: species={:.3g}\nemotion={:.3g}; interaction={:.3g}".format(a.loc["species", "p"], a.loc["emotion", "p"], a.loc["species:emotion", "p"])
                heading = title if show_titles else ""
                if show_subtitles:
                    heading += ("\n" if heading else "") + annotation
                ax.set_title(heading, pad=6)
                legend_labels = {"P": "Positive anticipation", "H": "Happiness",
                                 "DogP": "Dog P", "DogH": "Dog H", "HumP": "Human P", "HumH": "Human H"}
                handles = [Patch(facecolor=overrides.get(prop, {}).get(g, colors[g]),
                                 label=legend_labels[g], alpha=.85) for g in groups]
                if show_legend:
                    ax.legend(handles=handles, frameon=False, fontsize=font_size*.7,
                              ncol=1 if mode == "pooled" else 2, loc="upper right")
                ax.set_ylim(bottom=0)
                ax.margins(y=.15)
            for ax in axes.flat[len(plot_properties):]:
                ax.set_visible(False)
            subtitle = "P and H pooled across species" if mode == "pooled" else "P and H by species: Dog and Human"
            for extension in ["png", "pdf", "svg"]:
                path = output / f"properties_all_categories_{mode}.{extension}"
                fig.savefig(path, dpi=180)
                paths.append(path)
            # Export each property panel with its full axis labels and ticks.
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for ax, prop in zip(axes.flat, plot_properties):
                extent = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
                for extension in ["png", "pdf", "svg"]:
                    path = output / f"{prop}_P_H_by_species.{extension}"
                    fig.savefig(path, dpi=300, bbox_inches=extent, pad_inches=0.04)
                    paths.append(path)
            plt.close(fig)
    return paths


def write_report(videos: pd.DataFrame, tables: dict[str, pd.DataFrame], output: Path) -> Path:
    unique = independent_videos(videos)
    selected = unique[unique.emotion.isin(["P", "H"])]
    lines = ["# Stimulus video properties", "", f"Measured {len(videos)} videos, {int(videos.number_of_frames.sum()):,} frames, and {int(videos.number_of_transitions.sum()):,} consecutive-frame transitions.",
             f"The {len(videos)} folders contain {len(unique)} unique clips. Exact repeats across runs are measured and retained, but counted once in plot summaries and statistical tests.",
             f"P/H analysis: {len(selected)} unique clips; " + "; ".join(f"{g}: n={sum(selected.group == g)}" for g in GROUPS) + ".", "",
             "## Methods", "", "Every frame is measured at its stored resolution for color and block matching. Video values average frame properties; flow values average spatial mean magnitudes across N−1 transitions. SEM = sample SD / sqrt(number of unique clips).",
             "Hue is the original arithmetic HSV mean (not a circular statistic). Brightness uses rounded MATLAB grayscale coefficients. Original contrast is raw RMS intensity, not mean-centered contrast.",
             "Block motion reproduces 35×35 MSE three-step matching (±7 pixels) with magnitude-squared output summed across blocks and transitions; zero-padded edges and center-first ties are explicit Python choices, not validated against MATLAB. A per-transition version removes the transition-count scaling, but still depends on resolution and block count.",
             "Dense flow uses the supplied script's Farneback parameters and OpenCV grayscale, downscaled to longest side ≤640 unless configured otherwise. Pixel displacement is measured at the flow resolution. Normalized speed = 100 × mean magnitude / frame diagonal × source fps. No z-scoring is applied. Source frame counts and dimensions are checked against extracted frames.",
             "Two-sided Welch independent two-sample t-tests compare pooled P/H and all six pairs of DogP/DogH/HumP/HumH. Two-way OLS ANOVA uses species × emotion, type III sums of squares and sum contrasts. A separate sensitivity analysis adds fixed effects for the first run containing each unique clip. Distinct clips are assumed independent. Exact repeats are collapsed by sequence hash; related actors/source recordings require external metadata to model clustering.",
             "BH FDR controls across the 10 pooled outcomes; four-group tests include Holm correction over six pairs within each property and BH across all 60 tests. ANOVA includes BH within each of three terms and across all 30 tests. Unadjusted p-values are also saved. Effect sizes and Welch 95% confidence intervals are in CSV tables. Levene and residual Shapiro diagnostics flag possible limitations of classical ANOVA.", "",
             "## P versus H pooled across species", "", "Difference = P − H. q uses BH across properties.", "",
             "| Property | P mean ± SEM | H mean ± SEM | t (df) | p | q |", "|---|---:|---:|---:|---:|---:|"]
    for row in tables["ttest_pooled_P_vs_H"].itertuples():
        lines.append(f"| {PROPERTIES[row.property][0]} | {row.mean_a:.5g} ± {row.sem_a:.3g} | {row.mean_b:.5g} ± {row.sem_b:.3g} | {row.t:.3f} ({row.df:.1f}) | {row.p:.4g} | {row.p_fdr_bh:.4g} |")
    lines += ["", "## Species × emotion ANOVA", "", "q uses BH separately across properties for each ANOVA term.", "", "| Property | Term | F | p | q | Partial η² |", "|---|---|---:|---:|---:|---:|"]
    for row in tables["anova_species_emotion"].itertuples():
        lines.append(f"| {PROPERTIES[row.property][0]} | {row.term} | {row.F:.4g} | {row.p:.4g} | {row.p_fdr_bh_within_term:.4g} | {row.partial_eta_squared:.3g} |")
    lines += ["", "## Figures", "", "Only P and H are plotted, using six selected properties; statistical tables retain all measured outcomes.", "", "![All emotions pooled](properties_all_categories_pooled.png)", "", "![All emotions by species](properties_all_categories_by_species.png)", "",
              "## Files", "", "`video_properties.csv`: all emotions, one row per video folder. `frame_properties.csv`: every frame and SHA-256. `transition_motion.csv`: every consecutive pair. `input_file_inventory.csv`: every input file, including excluded control assets and OS metadata. `condition_counts.csv`: run/species/emotion folder counts. `duplicate_sequences.csv`: all occurrences of repeated clips. `unique_clip_membership.csv`: links every folder to its unique representative. `provenance.json`: parameters, source hashes, and versions. All inferential tables are separate CSVs.", "",
              "JPEG frame measurements can differ slightly from decoding the original videos because frame extraction recompressed the images. Different flow algorithms are different movement measures; the dense flow and block-motion outcomes should not be treated as interchangeable.", ""]
    path = output / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--videos", type=Path, default=DEFAULT_VIDEOS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-dimension", type=int, default=640)
    parser.add_argument("--fps", type=float, help="Explicit verified extraction fps; otherwise use matching source metadata")
    parser.add_argument("--force", action="store_true", help="Ignore cached video measurements")
    parser.add_argument("--statistics-only", action="store_true", help="Replot/retest an existing output/video_properties.csv")
    args = parser.parse_args()
    if args.statistics_only:
        videos = pd.read_csv(args.output / "video_properties.csv")
    else:
        videos = analyze_dataset(args.frames, args.videos, args.output, workers=args.workers,
                                 max_dimension=args.max_dimension, fps=args.fps, force=args.force)
    tables = statistical_analysis(videos, args.output)
    make_plots(videos, tables, args.output)
    report = write_report(videos, tables, args.output)
    print(f"Finished: {report.resolve()}", flush=True)


if __name__ == "__main__":
    main()
