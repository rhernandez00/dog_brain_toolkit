"""Scientific regression tests; run python -m pytest test_stimulus_video_properties.py."""
import importlib.util
from pathlib import Path
import sys

import cv2
from matplotlib.colors import rgb_to_hsv
import numpy as np
import pandas as pd
import pytest
from scipy import stats

import stimulus_video_properties as m


def test_color_matches_independent_hsv_reference_and_raw_rms():
    rng = np.random.default_rng(123)
    bgr = rng.integers(0, 256, (53, 47, 3), dtype=np.uint8)
    h, s, brightness, contrast, gray = m.color_properties(bgr)
    reference = rgb_to_hsv(bgr[..., ::-1].astype(float) / 255)
    assert h == pytest.approx(reference[..., 0].mean(), abs=1e-12)
    assert s == pytest.approx(reference[..., 1].mean(), abs=1e-12)
    assert brightness == pytest.approx(gray.astype(float).mean())
    assert contrast == pytest.approx(np.sqrt(np.mean(gray.astype(float)**2)))
    _, _, brightness, contrast, gray = m.color_properties(np.full((11, 9, 3), 128, np.uint8))
    assert brightness == contrast == 128  # The original RMS is not centered contrast.
    colors = np.array([[[0, 0, 255], [0, 255, 0], [255, 0, 0]]], np.uint8)
    assert m.color_properties(colors)[-1].tolist() == [[76, 150, 29]]


def test_block_matching_static_and_known_translation():
    y, x = np.mgrid[:35, :35]
    image = (255 * np.exp(-((y-17)**2 + (x-17)**2)/30)).astype(np.uint8)
    image[:7] = image[-7:] = 0
    image[:, :7] = image[:, -7:] = 0
    shifted = np.zeros_like(image)
    shifted[:, 4:] = image[:, :-4]
    assert m.block_motion_three_step(image, image) == 0
    assert m.block_motion_three_step(np.full((43, 51), 128, np.uint8), np.full((43, 51), 128, np.uint8)) == 0
    assert m.block_motion_three_step(image, shifted) == 16  # dx^2 + dy^2, not magnitude.


def test_inventory_orders_frames_and_audits_nonvideo_files(tmp_path):
    folder = tmp_path / "R12DogP3"
    folder.mkdir()
    for i in [2, 0, 1]:
        (folder / f"frame_{i:04}.jpg").write_bytes(b"inventory only")
    (folder / "desktop.ini").write_text("metadata")
    (folder / "R12DogP3.csv").write_text("scorer,user\nbodyparts,LM01\ncoords,x\n")
    (tmp_path / "others").mkdir()
    (tmp_path / "others" / "baseline.png").write_bytes(b"control")
    records, audit = m.inventory(tmp_path)
    assert records[0]["run"] == 12
    assert records[0]["group"] == "DogP"
    assert [Path(p).name for p in records[0]["frames"]] == [f"frame_{i:04}.jpg" for i in range(3)]
    assert len(audit) == 6
    assert set(audit.status) == {"included_video_frame", "excluded_os_metadata", "excluded_landmark_annotation_csv", "excluded_nonvideo_control_asset"}
    (folder / "frame_0001.jpg").unlink()
    with pytest.raises(ValueError, match="contiguous"):
        m.inventory(tmp_path)


def test_welch_against_scipy_and_ci():
    a = np.array([2, 3, 5, 7, 8.])
    b = np.array([1, 2, 2, 3, 3, 4.])
    result = m.welch_comparison(a, b)
    reference = stats.ttest_ind(a, b, equal_var=False)
    assert result["t"] == pytest.approx(reference.statistic)
    assert result["p"] == pytest.approx(reference.pvalue)
    assert result["sem_a"] == pytest.approx(stats.sem(a))
    assert result["difference_ci95_low"] < result["difference_a_minus_b"] < result["difference_ci95_high"]
    assert result["n_a"] == 5


def test_factorial_analysis_recovers_main_effects_at_video_level(tmp_path):
    rng = np.random.default_rng(18)
    rows = []
    for group in m.GROUPS:
        for i in range(24):
            value = 5 + 6 * (group.startswith("Hum")) + 2 * (group.endswith("H")) + rng.normal(0, .3)
            rows.append({"video_id": f"R{i//4+1}{group}{i%4+1}", "species": group[:3],
                         "emotion": group[-1], "group": group, "run": i//4+1,
                         **{prop: value for prop in m.PROPERTIES}})
    videos = pd.DataFrame(rows)
    tables = m.statistical_analysis(videos, tmp_path)
    assert len(tables["ttest_pooled_P_vs_H"]) == len(m.PROPERTIES)
    assert (tables["ttest_pooled_P_vs_H"].n_a == 48).all()
    assert len(tables["ttest_four_groups"]) == 6 * len(m.PROPERTIES)
    anova = tables["anova_species_emotion"]
    assert (anova.residual_df == 92).all()
    assert (anova.loc[anova.term.isin(["species", "emotion"]), "p"] < 1e-20).all()
    assert (anova.loc[anova.term == "species:emotion", "p"] > .05).all()
    pairwise = tables["ttest_four_groups"]
    assert (pairwise.p_holm_within_property >= pairwise.p - 1e-15).all()


def test_dense_flow_matches_supplied_reference_on_lossless_frames(tmp_path):
    source = Path(r"C:\github\Networks\dense_optical_flow_to_fsl.py")
    if not source.exists():
        pytest.skip("Supplied reference script is not available on this machine")
    spec = importlib.util.spec_from_file_location("supplied_dense_flow_reference", source)
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    rng = np.random.default_rng(500)
    initial = rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)
    images = [np.roll(initial, i*2, axis=1) for i in range(4)]
    video = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"FFV1"), 20, (96, 96))
    if not writer.isOpened():
        pytest.skip("OpenCV FFV1 lossless codec unavailable")
    folder = tmp_path / "R1DogP1"
    folder.mkdir()
    for i, image in enumerate(images):
        writer.write(image)
        cv2.imwrite(str(folder / f"frame_{i:04}.png"), image)
    writer.release()
    record = {**m.parse_video_id(folder.name), "fps": 20, "frames": [str(p) for p in sorted(folder.glob("*.png"))]}
    result = m.measure_video(record, 64)
    flow = reference.analyze_video(video, statistic="mean", max_dimension=64, cv2=cv2, np=np)
    expected = float(np.mean(flow.displacement_pixels))
    assert result["summary"]["dense_flow_pixels_per_frame"] == pytest.approx(expected, abs=1e-7)
    expected_speed = float(np.mean(flow.displacement_percent_diagonal)) * 20
    assert result["summary"]["dense_flow_percent_diagonal_per_second"] == pytest.approx(expected_speed, abs=1e-7)
    assert len(result["frames"]) == 4
    assert len(result["transitions"]) == 3


def test_source_count_mismatch_is_rejected(tmp_path):
    video = tmp_path / "R1HumH1.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"FFV1"), 30, (64, 64))
    if not writer.isOpened():
        pytest.skip("FFV1 unavailable")
    for _ in range(3):
        writer.write(np.zeros((64, 64, 3), np.uint8))
    writer.release()
    record = {**m.parse_video_id("R1HumH1"), "frames": ["one.png", "two.png"]}
    with pytest.raises(ValueError, match="frame-count mismatch"):
        m.source_metadata(tmp_path, [record], None)


def test_all_categories_are_plotted_with_custom_colors_but_only_ph_are_tested(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.colors import to_rgba
    rng = np.random.default_rng(54)
    rows = []
    for species in ["Dog", "Hum"]:
        for emotion in m.EMOTION_ORDER:
            for i in range(12):
                value = 5 + rng.random() + (1000 if emotion in "AFN" else 0)
                rows.append({"video_id": f"R{i//2+1}{species}{emotion}{i%2+1}",
                             "species": species, "emotion": emotion, "group": species+emotion,
                             "run": i//2+1, **{prop: value for prop in m.PROPERTIES}})
    videos = pd.DataFrame(rows)
    tables = m.statistical_analysis(videos, tmp_path)
    assert (tables["ttest_pooled_P_vs_H"].n_a == 24).all()
    assert (tables["ttest_pooled_P_vs_H"].n_b == 24).all()
    assert (tables["ttest_pooled_P_vs_H"].mean_a < 7).all()
    assert set(tables["group_summary"].group) == set(m.EMOTION_ORDER + [s+e for s in ["Dog", "Hum"] for e in m.EMOTION_ORDER])
    assert not tables["ttest_four_groups"].group_a.str.contains("[AFN]$").any()
    captures = []
    def inspect_plot(figure, path, **kwargs):
        if Path(path).suffix == ".png":
            captures.append(figure)
    monkeypatch.setattr(Figure, "savefig", inspect_plot)
    m.make_plots(videos, tables, tmp_path,
                 emotion_colors={"A": "black"}, group_colors={"HumF": "pink"},
                 property_colors={"brightness": {"A": "lime", "DogP": "navy"}})
    pooled, by_species = captures
    assert len(pooled.axes[0].patches) == 5
    assert len(by_species.axes[0].patches) == 10
    assert pooled.axes[0].patches[0].get_facecolor() == to_rgba("black", alpha=.85)
    assert by_species.axes[0].patches[6].get_facecolor() == to_rgba("pink", alpha=.85)
    assert pooled.axes[2].patches[0].get_facecolor() == to_rgba("lime", alpha=.85)
    assert by_species.axes[2].patches[3].get_facecolor() == to_rgba("navy", alpha=.85)
    assert "P vs H only" in pooled.axes[0].get_title()
    assert "P/H only" in by_species.axes[0].get_title()


def test_repeated_clips_do_not_inflate_sem_or_inferential_sample(tmp_path):
    rng = np.random.default_rng(28)
    rows = []
    for group in m.GROUPS:
        for i in range(12):
            value = 5 + rng.normal(0, .5) + (group.endswith('P'))
            rows.append({'video_id': f'R{i//2+1}{group}{i%2+1}', 'species': group[:3],
                         'emotion': group[-1], 'group': group, 'run': i//2+1,
                         'sequence_sha256': group+str(i), **{p: value for p in m.PROPERTIES}})
    originals = pd.DataFrame(rows)
    repeated = originals.iloc[:8].copy()
    repeated['video_id'] = [f'R99DogP{i+1}' for i in range(len(repeated))]
    repeated['run'] = 99
    combined = pd.concat([originals, repeated], ignore_index=True)
    assert len(m.independent_videos(combined)) == len(originals)
    a = m.statistical_analysis(originals, tmp_path / 'original')
    b = m.statistical_analysis(combined, tmp_path / 'repeated')
    pd.testing.assert_frame_equal(a['ttest_pooled_P_vs_H'], b['ttest_pooled_P_vs_H'])
    pd.testing.assert_frame_equal(a['group_summary'], b['group_summary'])
    assert (b['ttest_pooled_P_vs_H'].n_a == 24).all()
    conflict = repeated.iloc[:1].copy()
    conflict['emotion'] = 'H'
    with pytest.raises(ValueError, match='conflicting'):
        m.independent_videos(pd.concat([originals, conflict], ignore_index=True))
