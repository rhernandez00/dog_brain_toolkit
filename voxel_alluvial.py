"""Thresholded model -> lobe -> network -> region alluvial figures (Figure H).

Widths count model-voxel associations; repeated voxels across models are intentional.
Networks are annotations of atlas regions, not independently measured voxel networks.
"""
from pathlib import Path
import ast
import json
import re
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle

LEVELS = ["model", "lobe", "network", "region"]


def figure_c_settings(notebook):
    """Read selected literal settings without executing any notebook analysis."""
    cells = json.loads(Path(notebook).read_text(encoding="utf-8"))["cells"]
    tree = ast.parse("\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code"))
    wanted = {"MODELS", "RADIUS_SEARCHLIGHT", "Z_THRESHOLD", "RSA_METHOD", "MASK_PREFIX"}
    result = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    result[target.id] = ast.literal_eval(node.value)
    return result


def read_dictionary(path):
    table = pd.read_excel(path)
    table.columns = [str(c).strip().lower() for c in table.columns]
    required = {"number", "region", "lobe", "hemisphere"}
    if not required.issubset(table):
        raise ValueError(f"Missing dictionary columns: {required - set(table)}")
    table = table.dropna(subset=["number", "region"]).copy()
    table["number"] = table["number"].astype(int)
    table = table[table.number != 0]
    if table.number.duplicated().any():
        raise ValueError(f"Repeated atlas IDs in {path}")
    for column in ("region", "lobe", "hemisphere"):
        table[column] = table[column].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    return table.set_index("number")


def label_voxels(xyz_mm, ijk, atlas_path, dictionary_path):
    """Figure F rule: rounded affine lookup, then first integer-mm sphere majority.

    Fallback has no maximum distance; ties choose the lowest numeric atlas ID.
    nearest_labeled_mm is nearest distance, not the integer sphere radius.
    """
    dictionary = read_dictionary(dictionary_path)
    atlas = nib.load(atlas_path)
    data = atlas.get_fdata()
    if data.ndim != 3:
        raise ValueError("Expected a 3D label atlas")
    label_ijk = np.rint(nib.affines.apply_affine(np.linalg.inv(atlas.affine), xyz_mm)).astype(int)
    inside = np.all((label_ijk >= 0) & (label_ijk < np.asarray(data.shape)), axis=1)
    ids = np.zeros(len(xyz_mm), dtype=int)
    samples = data[tuple(label_ijk[inside].T)]
    ids[inside] = np.where(np.isfinite(samples), samples, 0).astype(int)
    direct = np.isin(ids, dictionary.index)
    method = np.where(direct, "direct", "unassigned").astype(object)
    distance = np.where(direct, 0.0, np.nan)
    valid = np.isfinite(data) & (data != 0) & np.isin(data, dictionary.index)
    atlas_ijk = np.argwhere(valid)
    if len(atlas_ijk):
        tree = cKDTree(nib.affines.apply_affine(atlas.affine, atlas_ijk))
        atlas_ids = data[tuple(atlas_ijk.T)].astype(int)
        missing = np.flatnonzero(~direct)
        nearest, _ = tree.query(xyz_mm[missing])
        for index, nearest_mm in zip(missing, nearest):
            radius = max(1, int(np.ceil(nearest_mm - 1e-8)))
            neighbors = tree.query_ball_point(xyz_mm[index], radius + 1e-8)
            values, counts = np.unique(atlas_ids[neighbors], return_counts=True)
            ids[index] = values[np.argmax(counts)]
            method[index] = "sphere-majority"
            distance[index] = nearest_mm
    names = dictionary.reindex(ids).reset_index(drop=True)
    unresolved = np.where(inside, "Unknown", "OutOfAtlas")
    result = pd.DataFrame({"atlas_id": ids,
        "region": names["region"].fillna(pd.Series(unresolved)),
        "lobe": names["lobe"].fillna("Unknown"),
        "hemisphere": names["hemisphere"].fillna("Unknown"),
        "assignment": method, "nearest_labeled_mm": distance})
    for j, name in enumerate(("i", "j", "k")):
        result[name] = ijk[:, j]
    for j, name in enumerate(("x_mm", "y_mm", "z_mm")):
        result[name] = xyz_mm[:, j]
    if "abbreviation" in names:
        result["abbreviation"] = names["abbreviation"].fillna("")
    return result


def normalize_dog_name(name):
    name = re.sub(r"\s+[LR]$", "", str(name).strip())
    name = re.sub(r"^[A-Za-z]+_\d+\s+", "", name)
    name = re.sub(r"^Area\s+", "", name, flags=re.I)
    return re.sub(r"[^a-z0-9]", "", name.lower())


def network_lookup(species, dictionary_path, network_dir, override=None):
    """Conservative named-region correspondence. Never use dog positional ROI IDs."""
    dictionary = read_dictionary(dictionary_path)
    result = dictionary[["region", "hemisphere"]].reset_index().rename(columns={"number": "atlas_id"})
    result["network"] = "Unmapped"
    result["network_assignment"] = "no verified correspondence"
    result["network_source_region"] = ""
    result["network_qc"] = ""
    if species == "H":
        src = pd.read_csv(Path(network_dir) / "AAL3_166_to_networks_with_QC.csv")
        if src.AAL3_ID.duplicated().any():
            raise ValueError("Duplicate AAL3 IDs in network table")
        src = src.set_index("AAL3_ID")
        for i, row in result.iterrows():
            if row.atlas_id not in src.index:
                continue
            match = src.loc[row.atlas_id]
            if re.sub(r"\s*\([^)]*\)$", "", row.region) != match.AAL3_region:
                result.loc[i, "network_qc"] = f"AAL3 ID/name mismatch: source {match.AAL3_region}"
                continue
            if pd.notna(match.Yeo7_family_recommended):
                result.loc[i, "network"] = match.Yeo7_family_recommended
            result.loc[i, ["network_assignment", "network_source_region", "network_qc"]] = [
                "AAL3 ID and name", match.AAL3_region, match.QC_status]
    else:
        src = pd.read_csv(Path(network_dir) / "dog-roi_to_network.csv")
        src["key"] = src.anatomical_region.map(normalize_dog_name)
        # Only explicit lexical synonyms; no fuzzy or whole-lobe assignment.
        aliases = {"caudatenucleus": "nucleuscaudatus",
                   "fissuraesuprasylviaeanterior": "fissurasuprasylvalisanterior"}
        for i, row in result.iterrows():
            key = normalize_dog_name(row.region)
            key = aliases.get(key, key)
            hemi = {"Left": "L", "Right": "R"}.get(row.hemisphere)
            candidates = src[(src.key == key) & (src.hemisphere == hemi)]
            # XMII has a Roman-numeral/code conflict in the source; do not infer.
            candidates = candidates[~candidates.anatomical_region.str.startswith("XMII")]
            if len(candidates) == 1:
                match = candidates.iloc[0]
                if pd.notna(match.network_standardized):
                    result.loc[i, "network"] = match.network_standardized
                result.loc[i, ["network_assignment", "network_source_region", "network_qc"]] = [
                    "explicit normalized region name", match.anatomical_region,
                    str(match.qc_note) if pd.notna(match.qc_note) else str(match.assignment_status)]
    if override is not None:
        edits = pd.read_csv(override)
        if edits.atlas_id.duplicated().any() or not set(edits.atlas_id).issubset(set(result.atlas_id)):
            raise ValueError("Override must have unique atlas IDs from this dictionary")
        for _, edit in edits.iterrows():
            if pd.notna(edit.network):
                i = result.index[result.atlas_id == edit.atlas_id]
                result.loc[i, "network"] = edit.network
                result.loc[i, "network_assignment"] = "user override"
    return result


def find_map(root, model, species, radius, method, mask, suffix):
    folder = Path(root) / model / "mean"
    hits = sorted(folder.glob(f"*-{species}-r-{radius}_*{method}{suffix}.nii.gz"))
    if mask:
        hits = [p for p in hits if p.name.startswith(mask + "-")]
    if len(hits) > 1:
        raise ValueError(f"Ambiguous maps; select a distance method explicitly: {hits}")
    return hits[0] if hits else None


def collect_species(species, models, root, radius, threshold, method, mask, atlas, dictionary,
                    network_dir, output, network_override=None):
    if not models or len(models) != len(set(models)):
        raise ValueError("Select at least one model, without duplicate names")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Z threshold must be positive and finite")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lookup = network_lookup(species, dictionary, network_dir, network_override)
    lookup.to_csv(output / f"{species}_atlas_network_audit.csv", index=False)
    records, statuses, scale_values = [], [], []
    cache = {}
    for model in models:
        suffixes = {"mean": "_mean", "z": "_z", "corrected": f"_zt{threshold:g}_corrected"}
        paths = {k: find_map(root, model, species, radius, method, mask, suffix)
                 for k, suffix in suffixes.items()}
        status = {"model": model, "radius_voxels": radius,
                  **{k + "_path": str(v) if v else "" for k, v in paths.items()}}
        missing = [k for k, v in paths.items() if v is None]
        # Derive exact companion filenames when any map in the family exists.
        existing = next((k for k, path in paths.items() if path is not None), None)
        if existing is not None:
            stem = str(paths[existing])[:-len(suffixes[existing] + ".nii.gz")]
            expected = {k: stem + suffixes[k] + ".nii.gz" for k in missing}
        else:
            expected = {k: str(Path(root) / model / "mean" /
                f"{mask or '*'}-{species}-r-{radius}_*{method}{suffixes[k]}.nii.gz") for k in missing}
        status["missing_files"] = "\n".join(expected.values())
        status.update(status="missing maps" if missing else "ready",
                      detail=", ".join(missing), retained_voxels=0)
        if missing:
            statuses.append(status)
            continue
        images = {k: nib.load(p) for k, p in paths.items()}
        ref = images["mean"]
        if ref.ndim != 3:
            raise ValueError(f"Expected 3D mean map: {paths['mean']}")
        for k, im in images.items():
            if im.shape != ref.shape or not np.allclose(im.affine, ref.affine, atol=1e-5, rtol=0):
                raise ValueError(f"{species} {model}: {k} grid/affine differs; no implicit resampling")
        mean, z, corrected = [images[k].get_fdata() for k in ("mean", "z", "corrected")]
        finite_mean = mean[np.isfinite(mean) & (mean != 0)]
        if len(finite_mean):
            scale_values.append((finite_mean.min(), finite_mean.max()))
        retained = np.isfinite(corrected) & (corrected != 0) & np.isfinite(mean) & np.isfinite(z)
        status["invalid_retained_values"] = int((np.isfinite(corrected) & (corrected != 0) &
                                                ~(np.isfinite(mean) & np.isfinite(z))).sum())
        ijk = np.argwhere(retained)
        status["retained_voxels"] = len(ijk)
        if not len(ijk):
            status.update(status="empty corrected map", detail="No finite retained voxels")
            statuses.append(status)
            continue
        # Cache by map-grid voxel coordinate; preserve identical F assignment across models.
        key = (ref.shape, ref.affine.tobytes())
        if key not in cache:
            cache[key] = {}
        grid_cache = cache[key]
        unseen = np.asarray([v for v in ijk if tuple(v) not in grid_cache], dtype=int).reshape(-1, 3)
        if len(unseen):
            labels = label_voxels(nib.affines.apply_affine(ref.affine, unseen), unseen, atlas, dictionary)
            labels = labels.merge(lookup.drop(columns=["region", "hemisphere"]), on="atlas_id",
                                  how="left", validate="many_to_one")
            labels["network"] = labels.network.fillna("Unmapped")
            grid_cache.update({tuple(v): row for v, row in zip(unseen, labels.to_dict("records"))})
        table = pd.DataFrame([grid_cache[tuple(v)] for v in ijk])
        table["model"] = model
        table["mean_correlation"] = mean[tuple(ijk.T)]
        table["z"] = z[tuple(ijk.T)]
        table["corrected_z"] = corrected[tuple(ijk.T)]
        table["opacity"] = np.clip(table.z.to_numpy() / threshold, 0, 1) ** 2
        records.append(table)
        statuses.append(status)
    status_table = pd.DataFrame(statuses)
    status_table.to_csv(output / f"{species}_map_status.csv", index=False)
    if records:
        voxels = pd.concat(records, ignore_index=True)
        voxels.to_csv(output / f"{species}_voxel_assignments.csv.gz", index=False)
        flows = voxels.groupby(LEVELS, sort=True, dropna=False).agg(
            voxel_count=("mean_correlation", "size"),
            mean_correlation=("mean_correlation", "mean"),
            mean_opacity=("opacity", "mean"),
            min_correlation=("mean_correlation", "min"),
            max_correlation=("mean_correlation", "max")).reset_index()
        assert flows.voxel_count.sum() == len(voxels)
    else:
        voxels = pd.DataFrame()
        flows = pd.DataFrame(columns=LEVELS + ["voxel_count", "mean_correlation", "mean_opacity"])
    flows.to_csv(output / f"{species}_alluvial_flows.csv", index=False)
    scale = (min(x[0] for x in scale_values), max(x[1] for x in scale_values)) if scale_values else (0, 1)
    summary = {"species": species, "radius_voxels": radius, "z_threshold": threshold,
        "models": models, "model_voxel_associations": len(voxels), "color_limits": list(scale),
        "width": "model-voxel associations; voxels may occur in several models",
        "selection": "finite nonzero corrected map and finite mean and Z",
        "color": "mean Kendall correlation per complete model/lobe/network/region path; shared species scale",
        "opacity": "mean of clip(Z / threshold, 0, 1)**2 per path",
        "assignment": "Figure F: rounded atlas lookup, then integer-mm sphere majority, no maximum distance"}
    if len(voxels):
        summary.update(unique_voxels=len(voxels[["x_mm", "y_mm", "z_mm"]].drop_duplicates()),
            unmapped_network_associations=int(voxels.network.eq("Unmapped").sum()),
            fallback_associations=int(voxels.assignment.eq("sphere-majority").sum()),
            opacity_range=[float(voxels.opacity.min()), float(voxels.opacity.max())])
    (output / f"{species}_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return flows, status_table, summary


def draw_alluvial(flows, status, metadata, output, *, color_option=None, show_title=True,
                  transparent_background=False, background_color="white", dpi=180):
    """Complete paths remain separate through intermediate nodes; no edge-only merging."""
    if color_option is None:
        color_option = dict(name="map_range", mode="map_range", cmap="viridis")
    norm, cmap, color_info = color_normalization(flows, metadata, color_option)
    species = metadata["species"]
    name = {"H": "Human", "D": "Dog"}[species]
    missing = status[status.status != "ready"]
    nregions = flows.region.nunique() if len(flows) else 0
    fig = plt.figure(figsize=(23, max(9, nregions * .20 + 4)))
    ax = fig.add_axes([.02, .22 if len(missing) else .10, .96, .65 if len(missing) else .77])
    xs = [0, 1.25, 2.5, 3.75]
    ax.set_xlim(-.04, 4.95)
    ax.set_ylim(-.025, 1.06)
    ax.axis("off")
    for x, level in zip(xs, LEVELS):
        ax.text(x, 1.04, level.title(), fontsize=13, weight="bold")
    if len(flows):
        total = flows.voxel_count.sum()
        # Same count scale in every column, even with differing numbers of nodes.
        node_orders = {}
        for level in LEVELS:
            labels = flows[level].unique()
            node_orders[level] = ([m for m in metadata["models"] if m in labels] if level == "model"
                                  else sorted(labels, key=lambda x: (x in ("Unknown", "Unmapped"), x)))
        max_nodes = max(map(len, node_orders.values()))
        gap = min(.008, .16 / max(1, max_nodes - 1))
        unit = (1 - gap * (max_nodes - 1)) / total
        slots = {}
        for stage, level in enumerate(LEVELS):
            y = 1.
            for label in node_orders[level]:
                subset = flows[flows[level] == label].sort_values(LEVELS, kind="stable")
                height = subset.voxel_count.sum() * unit
                top = y
                for index, row in subset.iterrows():
                    bottom = y - row.voxel_count * unit
                    slots[stage, index] = (bottom, y)
                    y = bottom
                ax.add_patch(Rectangle((xs[stage], y), .028, height,
                                       facecolor="#344054", edgecolor="none", zorder=4))
                # Labels are vertically repelled below, preserving true count widths.
                slots[stage, label] = (y + top) / 2
                y -= gap
        for index, row in flows.iterrows():
            color = cmap(norm(row.mean_correlation))
            for stage in range(3):
                a, b = xs[stage] + .028, xs[stage + 1]
                y0, y1 = slots[stage, index]
                z0, z1 = slots[stage + 1, index]
                dx = (b - a) * .45
                verts = [(a,y0),(a+dx,y0),(b-dx,z0),(b,z0),(b,z1),
                         (b-dx,z1),(a+dx,y1),(a,y1),(a,y0)]
                codes = [MplPath.MOVETO] + [MplPath.CURVE4]*3 + [MplPath.LINETO] + [MplPath.CURVE4]*3 + [MplPath.CLOSEPOLY]
                ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color,
                                      edgecolor="none", alpha=float(row.mean_opacity), zorder=1))
        for stage, level in enumerate(LEVELS):
            labels = node_orders[level]
            centers = np.array([slots[stage, label] for label in labels])
            positions = centers.copy()
            positions[0] = min(.995, positions[0])
            sep = min(.022, .95 / max(1, len(labels)-1))
            for i in range(1, len(positions)):
                positions[i] = min(positions[i], positions[i-1] - sep)
            if len(positions) and positions[-1] < .005:
                positions[-1] = .005
                for i in range(len(positions)-2, -1, -1):
                    positions[i] = max(positions[i], positions[i+1] + sep)
            for label, center, y in zip(labels, centers, positions):
                x = xs[stage] + .036
                ax.plot([x, x+.035], [center, y], color="#777777", lw=.35, zorder=5)
                ax.text(x+.04, y, label.replace("__", " / ").replace("_", " "),
                        fontsize=6.7 if level == "region" else 8, va="center",
                        zorder=6, bbox=dict(facecolor="none" if transparent_background else background_color,
                        alpha=0 if transparent_background else .85, edgecolor="none", pad=.3))
        cax = fig.add_axes([.72, .045, .23, .012])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                          orientation="horizontal",
                          extend="both" if color_info["ribbons_below"] or color_info["ribbons_above"] else "neither")
        ticks = np.asarray(norm.inverse(np.linspace(0, 1, 6)), dtype=float)
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{t:.3f}" for t in ticks])
        cb.set_label(f"Mean Kendall correlation — {color_option['name']} (shared species scale)", fontsize=9)
    else:
        ax.text(2.2, .60, "No thresholded model maps available", ha="center", fontsize=19, color="#667085")
        ax.text(2.2, .50, "Flows will appear when the required maps are added.", ha="center", fontsize=12)
    if len(missing):
        # Separate missingness marks have no quantitative flow width.
        entries = [f"{r.model}: {r.status} [{r.detail}]" for r in missing.itertuples()]
        cols = 3
        rows = int(np.ceil(len(entries)/cols))
        fig.text(.025, .19, "Missing / empty maps (dashed markers carry no voxel count)", fontsize=10, weight="bold")
        for i, entry in enumerate(entries):
            col, row = divmod(i, rows)
            x, y = .025 + col*.32, .165 - row*.018
            fig.add_artist(Rectangle((x,y-.004), .007, .009, transform=fig.transFigure,
                                     fill=False, linestyle="--", edgecolor="#b54708", lw=.8))
            fig.text(x+.011, y, entry, fontsize=6.3, va="center", color="#93370d")
    if show_title:
        fig.suptitle(f"{name}: thresholded model responses -> lobe -> network -> region\n"
                     f"Searchlight radius {metadata['radius_voxels']} voxels; corrected Z map, zt={metadata['z_threshold']:g}\n"
                     f"{metadata['model_voxel_associations']:,} model-voxel associations",
                     fontsize=17, y=.97)
    note = ("Width = model-voxel associations (overlap across models retained). "
            "Color = mean correlation; opacity = mean clip(Z/zt, 0, 1)^2.\n"
            "Networks annotate atlas regions; unmatched regions remain Unmapped. "
            "Lobe and network columns are descriptive assignments, not a nested hierarchy.")
    fig.text(.025, .025, note, fontsize=8, va="bottom")
    settings = dict(color_option, show_title=show_title,
                    transparent_background=transparent_background, background_color=background_color, dpi=dpi)
    base = Path(output) / f"{species}_alluvial_r{metadata['radius_voxels']}_{export_tag(settings)}"
    fig.savefig(str(base) + ".png", dpi=dpi, transparent=transparent_background,
                facecolor="none" if transparent_background else background_color)
    plt.close(fig)
    audit = dict(settings=settings, color_normalization=color_info,
                 opacity="Original mean_opacity preserved for every ribbon",
                 source_metadata=metadata)
    Path(str(base) + ".json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return base



def color_normalization(flows, metadata, option):
    """One scale shared by every model in a species; ribbon quantiles are unweighted."""
    from matplotlib.colors import TwoSlopeNorm, FuncNorm
    mode = option["mode"]
    values = flows.mean_correlation.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if mode not in {"map_range", "ribbon_range", "ribbon_robust", "zero_centered", "quantile", "fixed"}:
        raise ValueError(f"Unknown color mode: {mode}")
    info = dict(option)
    if mode == "fixed":
        lo, hi = float(option["vmin"]), float(option["vmax"])
        if not np.isfinite([lo, hi]).all() or lo >= hi:
            raise ValueError("Fixed color limits require finite vmin < vmax")
    elif not len(values):
        lo, hi = 0., 1.
        info["no_data"] = True
    elif mode == "map_range":
        lo, hi = metadata["color_limits"]
    elif mode in {"ribbon_range", "quantile"}:
        lo, hi = float(values.min()), float(values.max())
    elif mode == "ribbon_robust":
        low, high = option.get("percentiles", [2, 98])
        if not 0 <= low < high <= 100:
            raise ValueError("Require 0 <= lower < upper <= 100")
        lo, hi = np.percentile(values, [low, high])
    else:
        limit = float(np.max(np.abs(values)))
        lo, hi = -limit, limit
    if lo == hi:
        lo, hi = lo - 1e-6, hi + 1e-6
    norm = Normalize(lo, hi, clip=True)
    if mode == "zero_centered" and len(values):
        norm = TwoSlopeNorm(vmin=lo, vcenter=0, vmax=hi)
    if mode == "quantile" and len(np.unique(values)) > 1:
        knots, counts = np.unique(values, return_counts=True)
        ranks = (np.cumsum(counts) - counts / 2) / counts.sum()
        ranks = (ranks - ranks[0]) / (ranks[-1] - ranks[0])
        norm = FuncNorm((lambda x: np.interp(x, knots, ranks),
                         lambda y: np.interp(y, ranks, knots)), vmin=lo, vmax=hi, clip=True)
        info["quantile_weighting"] = "each complete ribbon has equal weight"
    info.update(vmin=float(lo), vmax=float(hi),
                ribbons_below=int((values < lo).sum()), ribbons_above=int((values > hi).sum()))
    return norm, plt.get_cmap(option.get("cmap", "viridis")), info


def export_tag(settings):
    """Stable option names: changing appearance or limits overwrites the same export."""
    return re.sub(r"[^A-Za-z0-9_-]+", "-", settings.get("name", "plot")).strip("-")


def draw_distributions(status, flows, metadata, output, *, show_title=True,
                       transparent_background=False, background_color="white", dpi=180, bins=80,
                       label_color="white", font_family="DejaVu Sans", label_font_size=10,
                       model_labels=None, label_halo_width=0):
    """Compare mean-map voxels with corrected voxels and actual ribbon averages.

    All-map panel excludes zero-valued background, as in Figure C. Exact valid
    zero correlations cannot be distinguished from background without an explicit mask.
    """
    output = Path(output)
    species = metadata["species"]
    assignment_path = output / f"{species}_voxel_assignments.csv.gz"
    if metadata["model_voxel_associations"]:
        retained = pd.read_csv(assignment_path, usecols=["model", "mean_correlation"])
    else:
        retained = pd.DataFrame(columns=["model", "mean_correlation"])
    model_labels = model_labels or {}
    arrays, summaries, labels, z_arrays = [], [], [], []
    for row in status.itertuples():
        path = str(row.mean_path)
        values = np.array([])
        if path and path != "nan":
            data = nib.load(path).get_fdata()
            values = data[np.isfinite(data) & (data != 0)]
        kept = retained.loc[retained.model == row.model, "mean_correlation"].to_numpy(dtype=float)
        for scope, vals in [("all nonzero mean-map voxels", values), ("corrected retained voxels", kept)]:
            if len(vals):
                q = np.percentile(vals, [2, 25, 50, 75, 98])
                summaries.append(dict(model=row.model, scope=scope, n=len(vals), mean=float(vals.mean()),
                    minimum=float(vals.min()), p02=q[0], p25=q[1], median=q[2], p75=q[3], p98=q[4], maximum=float(vals.max())))
            else:
                summaries.append(dict(model=row.model, scope=scope, n=0))
        arrays.append(values)
        zpath = str(getattr(row, "z_path", ""))
        if zpath and zpath != "nan":
            zdata = nib.load(zpath).get_fdata()
            z_arrays.append(zdata[np.isfinite(zdata) & (zdata != 0)])
        labels.append(model_labels.get(row.model, row.model) + (" [mean missing/empty]" if not len(values) else ""))
    fig = plt.figure(figsize=(22, 14), layout="constrained")
    grid = fig.add_gridspec(2, 3)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    axes += [fig.add_subplot(grid[1, :2]), fig.add_subplot(grid[1, 2])]
    pooled = np.concatenate(arrays) if arrays else np.array([])
    ribbon = flows.mean_correlation.to_numpy(dtype=float)
    panels = [(pooled, "All available mean maps", "Nonzero finite map voxels (models pooled)"),
              (retained.mean_correlation.to_numpy(dtype=float), "Corrected retained voxels", "Model-voxel associations"),
              (ribbon, "Ribbon mean correlations", "Complete ribbons (equal weight)")]
    for ax, (values, title, ylabel) in zip(axes[:3], panels):
        if len(values):
            ax.hist(values, bins=bins, color="#366e9f", alpha=.8)
            ax.axvline(np.median(values), color="#b54708", linestyle="--", label=f"Median {np.median(values):.4f}")
            ax.legend(loc="upper left")
            ax.text(.98, .96, f"n = {len(values):,}\nrange {values.min():.4f} to {values.max():.4f}",
                    transform=ax.transAxes, va="top", ha="right")
        else:
            ax.text(.5, .5, "No available values", transform=ax.transAxes, ha="center")
        ax.axvline(0, color="0.6", lw=.6)
        ax.set(title=title, xlabel="Mean Kendall correlation", ylabel=ylabel)
    ax = axes[3]
    for i, values in enumerate(arrays, start=1):
        if len(values) > 1 and np.ptp(values) > 0:
            violin = ax.violinplot([values], positions=[i], vert=False, showmedians=True,
                                  showextrema=False, widths=.8, points=100)
            for body in violin["bodies"]:
                body.set_facecolor("#366e9f")
                body.set_alpha(.65)
            violin["cmedians"].set_color("#b54708")
        elif len(values):
            ax.plot([values[0]], [i], marker="|", color="#b54708")
    ax.set_yticks(range(1, len(labels)+1), labels)
    ax.set_ylim(len(labels)+.7, .3)
    ax.set(title="Mean-map distribution by model (violins; orange marks = medians)",
           xlabel="Mean Kendall correlation")
    zax = axes[4]
    if z_arrays and any(len(a) for a in z_arrays):
        zvalues = np.concatenate(z_arrays)
        zax.hist(zvalues, bins=bins, color="#366e9f", alpha=.8)
    else:
        zax.text(.5, .5, "Z maps unavailable", transform=zax.transAxes, ha="center")
    threshold = metadata.get("z_threshold", 3.1)
    zax.axvline(threshold, color="#c01048", linestyle="--", linewidth=2,
               label=f"Z threshold = {threshold:g}")
    zax.legend(loc="upper right")
    zax.set(title="Available group Z maps (nonzero finite voxels)",
            xlabel="Group Z (separate from mean correlation)", ylabel="Model-voxel values")
    ax.tick_params(axis="y", labelsize=8)
    if show_title:
        fig.suptitle(f"{species}: distributions of mean correlation; radius {metadata['radius_voxels']} voxels", fontsize=16)
    from matplotlib.text import Text
    for text in fig.findobj(match=Text):
        text.set_fontsize(label_font_size)
    style_figure_text(fig, label_color, font_family, label_halo_width)
    settings = dict(name="correlation_distributions",label_color=label_color,
                    font_family=font_family,label_font_size=label_font_size, show_title=show_title,
                    transparent_background=transparent_background, background_color=background_color,
                    dpi=dpi, bins=bins)
    base = output / f"{species}_alluvial_r{metadata['radius_voxels']}_{export_tag(settings)}"
    fig.savefig(str(base)+".png", dpi=dpi, transparent=transparent_background,
                facecolor="none" if transparent_background else background_color)
    plt.close(fig)
    pd.DataFrame(summaries).to_csv(output / f"{species}_mean_correlation_distribution_summary.csv", index=False)
    return base

def unique_voxel_tracks(voxels, status, models):
    """One record per world-space voxel; preserve all selected-model memberships.

    Universe: union of corrected retained voxels in selected available maps.
    Missing models are 'Unavailable', never treated as negative membership.
    Color/opacity: first average matching-model values within each unique voxel;
    later cohort aggregation gives every voxel equal weight.
    """
    xyz = ["x_mm", "y_mm", "z_mm"]
    anatomy = ["lobe", "network", "region"]
    stages = [f"model_{i:02d}" for i in range(len(models))]
    empty_columns = xyz + anatomy + stages + ["mean_correlation", "mean_opacity"]
    if voxels.empty:
        return pd.DataFrame(columns=empty_columns), stages
    voxels = voxels[voxels.model.isin(models)].copy()
    if voxels.duplicated(xyz + ["model"]).any():
        raise ValueError("Duplicate spatial voxel within a model")
    grouped = voxels.groupby(xyz, sort=True, dropna=False)
    if (grouped[anatomy].nunique(dropna=False) > 1).any().any():
        raise ValueError("Anatomical assignments differ for the same world-space voxel")
    result = grouped[anatomy].first()
    means = grouped.agg(mean_correlation=("mean_correlation", "mean"),
                        mean_opacity=("opacity", "mean"), matching_models=("model", "nunique"))
    result = result.join(means)
    availability = status.set_index("model").status
    memberships = voxels.assign(present=True).pivot(index=xyz, columns="model", values="present")
    for model, stage in zip(models, stages):
        known = model in availability and availability[model] in ("ready", "empty corrected map")
        if not known:
            result[stage] = "Unavailable"
        elif model in memberships:
            result[stage] = np.where(memberships[model].reindex(result.index).eq(True), "Matching", "Not matching")
        else:
            result[stage] = "Not matching"
    return result.reset_index(), stages

def anatomical_label_layout(centers, columns=3, min_gap=.006):
    """Spread annotation text over columns; never alter any voxel/group coordinates."""
    if columns not in (1, 2, 3):
        raise ValueError("LABEL_COLUMNS must be 1, 2, or 3")
    last = np.full(columns, np.inf)
    result = []
    for center in centers:
        available = np.flatnonzero(center <= last-min_gap)
        column = int(available[0]) if len(available) else int(np.argmax(last))
        position = min(float(center), last[column]-min_gap)
        last[column] = position
        result.append((column, position))
    return result


def border_label_layout(centers, min_gap):
    """Keep a single text lane and distribute collisions above and below their anchors."""
    upper = np.asarray(centers, dtype=float).copy()
    lower = upper.copy()
    for i in range(1, len(lower)):
        lower[i] = min(lower[i], lower[i-1] - min_gap)
    for i in range(len(upper)-2, -1, -1):
        upper[i] = max(upper[i], upper[i+1] + min_gap)
    return [(0, position) for position in (upper + lower)/2]


def style_figure_text(fig, color, family, halo_width=0):
    """Style existing artists without changing process-wide plotting defaults."""
    from matplotlib import patheffects
    from matplotlib.text import Text
    if not np.isfinite(halo_width) or halo_width < 0:
        raise ValueError("Label halo width must be nonnegative and finite")
    for text in fig.findobj(match=Text):
        if halo_width:
            text.set_path_effects([patheffects.withStroke(linewidth=halo_width, foreground="black")])
        text.set_color(color)
        text.set_fontfamily(family)
    for ax in fig.axes:
        ax.tick_params(colors=color)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)


LOBE_DISPLAY_ORDER = {
    "H": ["Frontal", "Frontal / Parietal", "Parietal", "Temporal", "Temporal / Occipital",
          "Occipital", "Insular", "Limbic", "Subcortical (basal ganglia)", "Subcortical (thalamus)",
          "Brainstem (midbrain)", "Brainstem (midbrain / pons)", "Brainstem (pons)", "Cerebellum"],
    "D": ["Frontal", "Sensory-motor", "Parietal", "Temporal", "Occipital", "Cingulate", "Subcortical"],
}


def display_anatomical_order(tracks, species, lobe_order=None, network_order=None):
    """Project display convention; network barycenters use unique selected voxel counts."""
    labels = tracks.lobe.unique().tolist() if len(tracks) else []
    base = lobe_order if lobe_order is not None else LOBE_DISPLAY_ORDER.get(species, [])
    lobes = [x for x in base if x in labels] + sorted(set(labels)-set(base))
    if not len(tracks):
        return lobes, [], []
    ranks = {lobe:i for i,lobe in enumerate(lobes)}
    summary = tracks.assign(lobe_position=tracks.lobe.map(ranks)).groupby("network").agg(
        mean_lobe_position=("lobe_position","mean"), unique_voxels=("lobe_position","size")).reset_index()
    unknown = {"Unmapped","Unassigned","Unknown"}
    known = summary[~summary.network.isin(unknown)].sort_values(["mean_lobe_position","network"])
    networks = known.network.tolist() + sorted(set(summary.network)&unknown)
    if network_order is not None:
        networks = [x for x in network_order if x in networks] + [x for x in networks if x not in network_order]
    return lobes, networks, summary.to_dict("records")


def model_display_slots(flows, stages, models, status, fallback_voxels=1):
    counts = {m:int(flows.loc[flows[stage].eq("Matching"),"voxel_count"].sum()) for m,stage in zip(models,stages)}
    positive = [n for n in counts.values() if n>0]
    blank = max(0,min(positive)-1) if positive else fallback_voxels
    states=status.set_index("model").status.to_dict()
    return [dict(model=m,actual_voxels=counts[m],display_voxels=counts[m] if counts[m]>0 else blank,
                 placeholder=counts[m]==0,status=states.get(m,"missing maps"),
                 fallback_size=not bool(positive) and counts[m]==0) for m in models]


def draw_merged_voxel_tracks(tracks, stages, models, status, metadata, output, *, color_option,
                      show_title=True, transparent_background=False, background_color="white", dpi=180, model_colors=None, show_region_labels=False, label_columns=3,
                      model_column_scale=1.0, label_color="white", font_family="DejaVu Sans",
                      label_font_size=8, heading_font_size=10, title_font_size=18,
                      legend_font_size=9, note_font_size=8, model_scale_mode="manual",
                      lobe_order=None, network_order=None, divider_width=1.0,
                      missing_model_fallback_voxels=1, incoming_ribbon_opacity=1.0,
                      model_labels=None, label_halo_width=0, anatomical_column_order=("network", "lobe"), flip=False, ax=None, show_column_headings=None):
    """Complete identity-preserving bands; only identical full routes are bundled."""
    if not np.isfinite(incoming_ribbon_opacity) or not 0 <= incoming_ribbon_opacity <= 1:
        raise ValueError("Incoming ribbon opacity must be between 0 and 1")
    if not np.isfinite(model_column_scale) or not 0 < model_column_scale <= 1:
        raise ValueError("MODEL_COLUMN_SCALE must be greater than 0 and at most 1")
    for size in [label_font_size, heading_font_size, title_font_size, legend_font_size, note_font_size]:
        if not np.isfinite(size) or size <= 0:
            raise ValueError("Font sizes must be positive and finite")
    if model_scale_mode not in {"manual","exact","voxel"}:
        raise ValueError("MODEL_SCALE_MODE must be exact, manual, or voxel")
    from matplotlib.collections import PathCollection
    tracks = tracks.copy()
    if "lobe" in tracks:
        tracks["lobe_original"] = tracks["lobe"]
        tracks["lobe"] = tracks["lobe"].replace({"Occiptal":"Occipital"})
    levels = tuple(anatomical_column_order)
    if levels not in (("network", "lobe"), ("lobe", "network")):
        raise ValueError("Anatomical column order must be network/lobe or lobe/network")
    first_level, second_level = levels
    lobes, networks, order_summary = display_anatomical_order(tracks,metadata["species"],lobe_order,network_order)
    grouping_levels = stages + ["network", "lobe"]
    output = Path(output)
    flows = tracks.groupby(grouping_levels, dropna=False, sort=True).agg(
        voxel_count=("mean_correlation", "size"),
        mean_correlation=("mean_correlation", "mean"),
        mean_opacity=("mean_opacity", "mean")).reset_index() if len(tracks) else pd.DataFrame(
            columns=grouping_levels + ["voxel_count", "mean_correlation", "mean_opacity"])
    embedded = ax is not None
    if not embedded:
        flows.to_csv(output / f"{metadata['species']}_unique_voxel_flows.csv", index=False)
    model_mode = color_option.get("mode") == "model"
    if model_mode:
        from matplotlib.colors import to_rgba, to_hex
        palette = plt.get_cmap(color_option.get("cmap", "tab20"))
        model_colors = model_colors or {m: to_hex(palette(i % palette.N)) for i,m in enumerate(models)}
        if set(models) - set(model_colors):
            raise ValueError("MODEL_COLORS must include every selected model")
        color_info = dict(mode="model", colors=model_colors)
    else:
        norm, cmap, color_info = color_normalization(flows, metadata, color_option)
    model_labels = model_labels or {}
    slots = model_display_slots(flows,stages,models,status,missing_model_fallback_voxels)
    n = 3
    if embedded:
        fig = ax.figure
    else:
        fig = plt.figure(figsize=(23, 20))
        ax = fig.add_axes([.015,.17,.97,.70])
    xlim = (-.08, 2.026 if embedded else 3.2)
    ax.set(xlim=xlim[::-1] if flip else xlim, ylim=(-.015,1.07))
    ax.axis("off")
    alignment = "right" if flip else "left"
    titles = ["Models (associations + marked blanks)"] + [f"{level.title()} (unique voxels)" for level in levels]
    show_column_headings = show_column_headings or {}
    for j, (level, title) in enumerate(zip(("model", *levels), titles)):
        if show_column_headings.get(level, True) and (not embedded or j < n-1):
            ax.text(j,1.04,title,fontsize=heading_font_size,ha=alignment)
    if len(flows):
        total = int(flows.voxel_count.sum())
        assert total == len(tracks)
        orders = {"model":models,"network":networks,"lobe":lobes}
        associations = sum(slot["actual_voxels"] for slot in slots)
        display_count = sum(slot["display_voxels"] for slot in slots)
        if model_scale_mode == "exact":
            unit, source_unit = 1/total, 1/display_count
        else:
            unit = 1/max(display_count,total)
            source_unit = unit*(model_column_scale if model_scale_mode == "manual" else 1)
        bottom = np.zeros((len(flows), n)); top = bottom.copy()
        source_slots = {}
        for j,level in enumerate(("model", *levels)):
            stage_total = display_count if j == 0 else total
            column_unit = source_unit if j == 0 else unit
            span = stage_total*column_unit
            y = (1+span)/2
            centers=[]
            boundaries=[y]
            for label in orders[level]:
                if j == 0 and slots[models.index(label)]["placeholder"]:
                    slot = slots[models.index(label)]
                    h = slot["display_voxels"]*source_unit
                    centers.append(y-h/2)
                    ax.add_patch(Rectangle((j,y-h),.018,h,facecolor="none",edgecolor=label_color,
                                           linestyle="--",linewidth=.8,zorder=5))
                    if h == 0:
                        ax.plot([j-.008,j+.026],[y,y],color=label_color,linestyle="--",lw=.8,zorder=5)
                    y -= h
                    continue
                ix = flows.index[flows[stages[models.index(label)]].eq("Matching") if j == 0
                                 else flows[level].eq(label)].to_numpy()
                height = flows.loc[ix,"voxel_count"].to_numpy()*column_unit
                top[ix,j] = y - np.r_[0,np.cumsum(height[:-1])]
                bottom[ix,j] = top[ix,j]-height
                if j == 0:
                    for index in ix:
                        source_slots[label,int(index)] = (bottom[index,j],top[index,j])
                center = (y+bottom[ix[-1],j])/2
                centers.append(center)
                ax.add_patch(Rectangle((j,bottom[ix[-1],j]),.018,height.sum(),facecolor=model_colors[label] if model_mode and j == 0 else "#344054",zorder=4))
                y=bottom[ix[-1],j]
                boundaries.append(y)
            if j > 0:
                for boundary in boundaries:
                    ax.plot([j-.008,j+.026],[boundary,boundary],color=label_color,
                            lw=divider_width,zorder=7)
            points_per_y = fig.get_figheight()*ax.get_position().height*72 / (ax.get_ylim()[1]-ax.get_ylim()[0])
            min_height = (label_font_size + 2) / points_per_y
            if j == 0:
                # Stagger crowded model labels horizontally, keeping their real block centers.
                columns = anatomical_label_layout(centers, label_columns, min_height * 2)
                layout = [(column, center) for center, (column, _) in zip(centers, columns)]
            elif not embedded and j == n-1:
                layout = border_label_layout(centers, min_height)
            else:
                layout = anatomical_label_layout(centers, label_columns, min_height)
            for label,center,(column,pos) in zip(orders[level],centers,layout):
                slot = slots[models.index(label)] if j == 0 else None
                text_label = label.replace("__", " / ").replace("_", " ")
                if slot is not None:
                    text_label = model_labels.get(label, text_label)
                    if slot["placeholder"] and slot["status"] != "empty corrected map":
                        text_label += "\nMap unavailable"
                # Keep center labels on their species' side of the junction.
                inward = embedded and j == n-1
                direction = -1 if inward else 1
                label_x = j + direction*(.025 + column*.25)
                if column > 0 or abs(pos-center) > 1e-10:
                    edge = j if inward else j+.019
                    ax.plot([edge,label_x-direction*.004],[center,pos],color=label_color,lw=.35,
                            linestyle=":",zorder=5)
                label_align = "right" if flip != inward else "left"
                ax.text(label_x,pos,text_label,fontsize=label_font_size,va="center",ha=label_align,zorder=6)
        if not embedded and all(c in tracks for c in ["x_mm","y_mm","z_mm"]):
            routes=tracks[["x_mm","y_mm","z_mm"]+stages].copy()
            routes.insert(0,"voxel_id",["|".join(float(x).hex() for x in row)
                                       for row in routes[["x_mm","y_mm","z_mm"]].to_numpy()])
            groups=tracks.groupby(grouping_levels,dropna=False,sort=True)
            bundle=groups.ngroup().to_numpy(dtype=int)
            within=groups.cumcount().to_numpy(dtype=int)
            routes["bundle_id"]=bundle
            for column,level in enumerate(levels,start=1):
                routes[level+"_slot_bottom"]=bottom[bundle,column]+within*unit
                routes[level+"_slot_top"]=routes[level+"_slot_bottom"]+unit
            routes.to_csv(output/f"{metadata['species']}_voxel_slot_routes.csv.gz",index=False)
        if model_mode:
            paths=[]; colors=[]
            memberships = [[m for m,stage in zip(models,stages) if row[stage] == "Matching"]
                           for _,row in flows.iterrows()]
            # Every occurrence of a shared voxel lands on its FULL destination slot.
            for (model,i),(lo,hi) in source_slots.items():
                dst_lo,dst_hi = bottom[i,1],top[i,1]
                verts=[(.018,lo),(.46,lo),(.54,dst_lo),(1,dst_lo),
                       (1,dst_hi),(.54,dst_hi),(.46,hi),(.018,hi),(.018,lo)]
                codes=[MplPath.MOVETO]+[MplPath.CURVE4]*3+[MplPath.LINETO]+[MplPath.CURVE4]*3+[MplPath.CLOSEPOLY]
                paths.append(MplPath(verts,codes))
                colors.append(to_rgba(model_colors[model],float(flows.loc[i,"mean_opacity"])*incoming_ribbon_opacity))
            for i in range(len(flows)):
                lows,highs=bottom[i,1:],top[i,1:]
                verts=[(1.018,lows[0])];codes=[MplPath.MOVETO]
                for j in range(1,n-1):
                    verts.extend([(j+.46,lows[j-1]),(j+.54,lows[j]),(j+1,lows[j]),(j+1+.018,lows[j])])
                    codes.extend([MplPath.CURVE4]*3+[MplPath.LINETO])
                verts.append((n-1+.018,highs[-1]));codes.append(MplPath.LINETO)
                for j in range(n-2,0,-1):
                    verts.extend([(j+1,highs[j]),(j+.54,highs[j]),(j+.46,highs[j-1]),(j+.018,highs[j-1])])
                    codes.extend([MplPath.LINETO]+[MplPath.CURVE4]*3)
                verts.append(verts[0]);codes.append(MplPath.CLOSEPOLY)
                paths.append(MplPath(verts,codes))
                # A single downstream track; equal RGB blend represents model membership.
                rgb=np.mean([to_rgba(model_colors[m])[:3] for m in memberships[i]],axis=0)
                colors.append((*rgb,float(flows.loc[i,"mean_opacity"])))
            ax.add_collection(PathCollection(paths,facecolors=colors,edgecolors="none",transform=ax.transData,zorder=1))
        else:
            paths=[]
            path_indices=[]
            # Duplicate association tracks converge onto the SAME spatial slot at the first anatomical column.
            for (model,i),(lo,hi) in source_slots.items():
                verts=[(.018,lo),(.46,lo),(.54,bottom[i,1]),(1,bottom[i,1]),
                       (1,top[i,1]),(.54,top[i,1]),(.46,hi),(.018,hi),(.018,lo)]
                codes=[MplPath.MOVETO]+[MplPath.CURVE4]*3+[MplPath.LINETO]+[MplPath.CURVE4]*3+[MplPath.CLOSEPOLY]
                paths.append(MplPath(verts,codes)); path_indices.append(i)
            for i in range(len(flows)):
                verts=[(1.018,bottom[i,1])]; codes=[MplPath.MOVETO]
                for j in range(1,n-1):
                    verts.extend([(j+.46,bottom[i,j]),(j+.54,bottom[i,j+1]),(j+1,bottom[i,j+1]),
                                  (j+1+.018,bottom[i,j+1])])
                    codes.extend([MplPath.CURVE4]*3+[MplPath.LINETO])
                verts.append((n-1+.018,top[i,-1])); codes.append(MplPath.LINETO)
                for j in range(n-2,0,-1):
                    verts.extend([(j+1,top[i,j+1]),(j+.54,top[i,j+1]),(j+.46,top[i,j]),(j+.018,top[i,j])])
                    codes.extend([MplPath.LINETO]+[MplPath.CURVE4]*3)
                verts.append(verts[0]); codes.append(MplPath.CLOSEPOLY)
                paths.append(MplPath(verts,codes)); path_indices.append(i)
            rgba=cmap(norm(flows.mean_correlation.to_numpy()))
            rgba[:,3]=flows.mean_opacity.to_numpy()
            path_colors = rgba[np.asarray(path_indices)]
            path_colors[:len(source_slots), 3] *= incoming_ribbon_opacity
            ax.add_collection(PathCollection(paths,facecolors=path_colors,edgecolors="none",transform=ax.transData,zorder=1))
            bounds = ax.get_position()
            colorbar_position = [bounds.x0+bounds.width*.27,.09,bounds.width*.46,.012] if embedded else [.74,.065,.22,.012]
            cax=fig.add_axes(colorbar_position)
            cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation="horizontal")
            ticks=np.asarray(norm.inverse(np.linspace(0,1,6)))
            cb.set_ticks(ticks); cb.set_ticklabels([f"{t:.3f}" for t in ticks])
            cb.set_label("Mean correlation across matching models, then across unique voxels",fontsize=9)
    else:
        # No smallest positive model exists: explicitly schematic fallback blank slots.
        total_blank = sum(slot["display_voxels"] for slot in slots)
        step = 1/max(1,total_blank)
        y=1.
        for slot in slots:
            h=slot["display_voxels"]*step
            ax.add_patch(Rectangle((0,y-h),.018,h,fill=False,edgecolor=label_color,
                                   linestyle="--",linewidth=.8))
            model = slot["model"]
            text_label = model_labels.get(model, model)
            if slot["status"] != "empty corrected map":
                text_label += "\nMap unavailable"
            ax.text(.025,y-h/2,text_label,
                    fontsize=label_font_size,va="center",ha=alignment)
            y -= h
        ax.text(1.5,.5,"No corrected voxel tracks available",ha="center",fontsize=heading_font_size)
    if embedded:
        return ax
    if show_title:
        fig.suptitle(f"{metadata['species']}: model associations merge into unique voxels at {first_level} | radius {metadata['radius_voxels']} voxels"
                     f"\n{metadata['model_voxel_associations']:,} model associations -> {len(tracks):,} unique voxels",fontsize=title_font_size)
    style_figure_text(fig, label_color, font_family, label_halo_width)
    base=output/f"{metadata['species']}_alluvial_r{metadata['radius_voxels']}_{export_tag(color_option)}"
    fig.savefig(str(base)+".png",dpi=dpi,transparent=transparent_background,
                facecolor="none" if transparent_background else background_color)
    plt.close(fig)
    Path(str(base)+".json").write_text(json.dumps(dict(
        flip=bool(flip),incoming_ribbon_opacity=incoming_ribbon_opacity,
        model_labels=model_labels,label_halo_width=label_halo_width,
        color_option=color_option,color_normalization=color_info,show_region_labels=show_region_labels,label_columns=label_columns,model_column_scale=model_column_scale,model_scale_mode=model_scale_mode,model_slots=slots,
        network_order=networks,lobe_order=lobes,network_order_statistics=order_summary,divider_width=divider_width,
        label_color=label_color,font_family=font_family,label_font_size=label_font_size,
        heading_font_size=heading_font_size,title_font_size=title_font_size,
        legend_font_size=legend_font_size,note_font_size=note_font_size,anatomical_group_gap=0,unique_voxels=len(tracks),model_voxel_associations=metadata["model_voxel_associations"],
        stages=dict(zip(stages,models)),width=("independently normalized columns with equal heights; model includes blank display slots" if model_scale_mode == "exact" else "shared anatomical voxel unit with configured model display scale"),
        color=("incoming model colors; one downstream full-width track with equal RGB blend of matching-model colors" if model_mode else "mean of matching models within each voxel, then equal-weight voxel mean per route"),
        anatomical_column_order=levels,
        layout=f"model associations -> merge at {first_level} -> unique {second_level}"),indent=2),encoding="utf-8")
    return base

def draw_combined_voxel_tracks(results, models, output, *, color_option, flips,
                               lobe_orders, network_orders, **style):
    """Place independent species plots with their final columns facing the center."""
    if set(results) != {"D", "H"} or flips["D"] == flips["H"]:
        raise ValueError("The combined figure requires dog and human with opposite flip settings")
    fig = plt.figure(figsize=(32, 16))
    try:
        for species, result in results.items():
            flipped = flips[species]
            ax = fig.add_axes([.502 if flipped else .02, .19, .478, .67])
            draw_merged_voxel_tracks(
                result["tracks"], result["stages"], models, result["status"], result["metadata"], output,
                color_option=color_option, flip=flipped, ax=ax,
                lobe_order=lobe_orders[species], network_order=network_orders[species], **style,
            )
            name = {"D": "Dog", "H": "Human"}[species]
            ax.text(.5, 1.09, f"{name} · {len(result['tracks']):,} unique voxels",
                    transform=ax.transAxes, ha="center", fontsize=style.get("title_font_size", 18))
        final_level = style.get("anatomical_column_order", ("network", "lobe"))[-1]
        reference_ax = ax
        center_y = fig.transFigure.inverted().transform(reference_ax.transData.transform((2, 1.04)))[1]
        headings = style.get("show_column_headings") or {}
        if headings.get(final_level, True):
            fig.text(.5, center_y, f"{final_level.title()} (unique voxels)", ha="center",
                     fontsize=style.get("heading_font_size", 10))
        if style.get("show_title", True):
            fig.suptitle("Model associations by species", y=.96, fontsize=style.get("title_font_size", 18))
        style_figure_text(fig, style.get("label_color", "white"), style.get("font_family", "DejaVu Sans"),
                          style.get("label_halo_width", 0))
        radii = [str(results[species]["metadata"]["radius_voxels"]) for species in ("D", "H")]
        radius_tag = f"r{radii[0]}" if radii[0] == radii[1] else f"D_r{radii[0]}_H_r{radii[1]}"
        base = Path(output) / f"DH_alluvial_{radius_tag}_{export_tag(color_option)}"
        transparent = style.get("transparent_background", False)
        fig.savefig(base.with_suffix(".png"), dpi=style.get("dpi", 180), transparent=transparent,
                    facecolor="none" if transparent else style.get("background_color", "white"))
        return base
    finally:
        plt.close(fig)
