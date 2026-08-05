"""Pure NIfTI / rendering helpers shared by the viewer apps.

No Dash, no global state — every function takes its inputs explicitly so the
same code serves the dual-species viewer, the legacy single-species viewer,
and (later) the static failsafe exporter.
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import plotly.graph_objects as go

_VIZ_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = os.path.dirname(_VIZ_DIR)  # repo root (Atlas/ lives here)

ATLAS_PATHS = {
    "D": {
        "low":  os.path.join(SCRIPT_DIR, "Atlas", "Dog", "Nitzsche", "Czeibert_brain2mm.nii.gz"),
        "high": os.path.join(SCRIPT_DIR, "Atlas", "Dog", "Nitzsche", "Czeibert_brain.nii.gz"),
    },
    "H": {
        "low":  os.path.join(SCRIPT_DIR, "Atlas", "Hum", "MNI152_T1_2mm_brain.nii.gz"),
        # High-res human atlas is a placeholder — reuse the 2mm for now.
        "high": os.path.join(SCRIPT_DIR, "Atlas", "Hum", "MNI152_T1_2mm_brain.nii.gz"),
    },
}

# Region-label sources for click-to-name (mirrors export_static / create_tables).
# Dog: Czeibert labels (2mm) + dictionary. Human: AAL3 lives per-dataset under
# {datafolder}/{dataset}/ROI/AAL3.nii.gz, so it is resolved at load time.
LABEL_ATLAS_PATHS = {
    "D": os.path.join(SCRIPT_DIR, "Atlas", "Dog", "Nitzsche", "Czeibert_labels2mm.nii.gz"),
    "H": None,  # resolved per-dataset
}
LABEL_DICT_CSV = {
    "D": os.path.join(SCRIPT_DIR, "Atlas", "Dog", "Czeibert_dictionary.csv"),
    "H": os.path.join(SCRIPT_DIR, "Atlas", "Hum", "AAL_dictionary.csv"),
}

# Warm sequential scale for stat overlays, readable on a white background.
OVERLAY_COLORSCALE = "YlOrRd"
# Slices are drawn radiology-style: the anatomy is bright on black, so the
# grayscale atlas scale is *reversed* (plain "Greys" runs white→black, which
# would paint tissue dark on a white field).
ATLAS_COLORSCALE = "Greys"
ATLAS_REVERSE = True
SLICE_BG = "#000000"        # canvas behind a slice — paper *and* plot area
SLICE_INK = "#ffffff"       # title / orientation labels on that canvas
PANEL_BG = "#ffffff"
INK = "#222222"
SURFACE_COLOR = "#b9c4da"

# Anatomical direction each in-plane axis should point in a rendered slice:
# (towards the right edge, towards the top edge). Neurological convention
# (subject's left on the image's left), superior up, and — on an axial slice —
# **anterior up**, so the frontal pole is at the top of the picture.
SLICE_ORIENT = {0: ("A", "S"), 1: ("R", "S"), 2: ("R", "A")}
_OPPOSITE = {"R": "L", "L": "R", "A": "P", "P": "A", "S": "I", "I": "S"}


# --- IO / geometry --------------------------------------------------------

def load_nifti(path):
    img = nib.load(path)
    return np.asanyarray(img.dataobj, dtype=np.float32), img.affine, img.header


def voxel_to_world(vox, affine):
    v = np.array([*vox, 1.0])
    return tuple((affine @ v)[:3])


def world_to_voxel(world, affine):
    inv = np.linalg.inv(affine)
    w = np.array([*world, 1.0])
    return tuple(np.round((inv @ w)[:3]).astype(int))


def resample_lowres_to_highres(low_data, low_affine, high_shape, high_affine):
    """Nearest-neighbour resample a low-res volume onto a high-res grid."""
    inv_low = np.linalg.inv(low_affine)
    ii, jj, kk = np.mgrid[0:high_shape[0], 0:high_shape[1], 0:high_shape[2]]
    flat = np.vstack([ii.ravel(), jj.ravel(), kk.ravel(), np.ones(ii.size)])
    world = high_affine @ flat
    vox_low = np.round((inv_low @ world)[:3]).astype(int)
    lo_s = np.array(low_data.shape).reshape(3, 1)
    mask = np.all((vox_low >= 0) & (vox_low < lo_s), axis=0)
    out = np.zeros(ii.size, dtype=np.float32)
    out[mask] = low_data[vox_low[0, mask], vox_low[1, mask], vox_low[2, mask]]
    return out.reshape(high_shape)


def load_atlas(specie):
    """Return (hi_norm, hi_affine, lo_affine, lo_shape) for a species atlas."""
    paths = ATLAS_PATHS[specie]
    hi, hi_aff, _ = load_nifti(paths["high"])
    lo, lo_aff, _ = load_nifti(paths["low"])
    hi_max = np.percentile(hi[hi > 0], 99.5) if np.any(hi > 0) else 1
    hi = np.clip(hi / hi_max, 0, 1)
    return hi, hi_aff, lo_aff, lo.shape


def load_label_atlas(specie, datafolder=None, dataset=None):
    """Return (label_data, label_affine, {number: region_name}) for click-to-name.

    Returns (None, None, {}) if the label atlas can't be found (e.g. the network
    disk is unavailable), so the caller can degrade gracefully.
    """
    src = LABEL_ATLAS_PATHS.get(specie)
    if specie == "H" and datafolder and dataset:
        src = os.path.join(datafolder, dataset, "ROI", "AAL3.nii.gz")
    mapping = {}
    dcsv = LABEL_DICT_CSV.get(specie)
    if dcsv and os.path.exists(dcsv):
        df = pd.read_csv(dcsv)
        for _, row in df.iterrows():
            num = row.get("Number")
            if pd.isna(num):
                continue
            mapping[int(num)] = str(row.get("Region", "Unknown"))
    if not src or not os.path.exists(src):
        return None, None, mapping
    data, aff, _ = load_nifti(src)
    return data, aff, mapping


def region_name_at(vox_overlay, overlay_affine, label_data, label_affine, label_dict):
    """Name the region under an overlay-grid voxel via the label atlas + dictionary."""
    if label_data is None or overlay_affine is None or label_affine is None:
        return None
    world = voxel_to_world(vox_overlay, overlay_affine)
    lv = world_to_voxel(world, label_affine)
    if any(c < 0 or c >= s for c, s in zip(lv, label_data.shape)):
        return "outside atlas"
    num = int(label_data[lv[0], lv[1], lv[2]])
    if num == 0:
        return "outside atlas"
    return label_dict.get(num, f"label {num}")


# --- Figures --------------------------------------------------------------

def empty_fig(title="", height=360, dark=False):
    """Placeholder figure. ``dark=True`` matches the black slice canvas, so a card
    that swaps a missing map for a real one doesn't flash white."""
    bg, ink, ghost = (SLICE_BG, "#ffffff", "#666666") if dark else (PANEL_BG, INK, "#aaa")
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=ink)),
        margin=dict(l=5, r=5, t=30, b=5),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor=bg, paper_bgcolor=bg, font_color=ink, height=height,
        annotations=[dict(text="No data", showarrow=False, font=dict(size=15, color=ghost),
                          xref="paper", yref="paper", x=0.5, y=0.5)],
    )
    return fig


def _threshold(ov, z_threshold):
    """Return overlay with |value| < threshold set to NaN (None passes through)."""
    if ov is None:
        return None
    return np.where(np.abs(ov) >= float(z_threshold), ov, np.nan)


def slice_orientation(affine, axis):
    """(transpose order, flip_rows, flip_cols, edge labels) for one slice axis.

    Given the volume's ``affine``, work out how to lay the two in-plane array
    axes on screen so the picture follows ``SLICE_ORIENT[axis]``. Without this,
    a slice is drawn in whatever order the array happens to store its voxels —
    which is how an axial slice ends up with the frontal pole at the *bottom*.

    Returned ``order`` indexes the two remaining array axes (in ascending order)
    as they should become (rows, cols); Plotly's y axis increases upward, so row
    0 is the bottom of the image. ``labels`` is (left, right, bottom, top).
    Returns ``None`` when the affine can't be read, so the caller can fall back.
    """
    if affine is None:
        return None
    try:
        codes = nib.aff2axcodes(np.asarray(affine, dtype=float))
    except Exception:
        return None
    want_right, want_up = SLICE_ORIENT[int(axis)]
    in_axes = [a for a in range(3) if a != int(axis)]
    place = {}
    for pos, a in enumerate(in_axes):
        c = codes[a]
        for want in (want_right, want_up):
            if c == want:
                place[want] = (pos, False)
            elif c == _OPPOSITE[want]:
                place[want] = (pos, True)      # axis runs the wrong way → flip
    if want_right not in place or want_up not in place:
        return None                            # degenerate/oblique beyond rescue
    (up_pos, up_flip), (right_pos, right_flip) = place[want_up], place[want_right]
    if up_pos == right_pos:
        return None
    labels = (_OPPOSITE[want_right], want_right, _OPPOSITE[want_up], want_up)
    return (up_pos, right_pos), up_flip, right_flip, labels


def _oriented(vol, axis, idx, orient):
    """Slice ``vol`` at ``idx`` and lay it out per ``slice_orientation``."""
    if vol is None:
        return None
    sl = np.take(vol, idx, axis=int(axis))
    if orient is None:
        return np.rot90(sl)                    # legacy layout (no affine given)
    order, up_flip, right_flip, _ = orient
    sl = np.transpose(sl, order)
    if up_flip:
        sl = sl[::-1, :]
    if right_flip:
        sl = sl[:, ::-1]
    return sl


def make_slice_fig(atlas, overlay, axis, slice_idx, opacity, z_threshold,
                   vmin, vmax, show_crosshair=False, cross=None, title="", height=360,
                   colorscale=None, affine=None):
    """One orthogonal slice: grayscale atlas + thresholded overlay, on black.

    The anatomy is drawn radiology-style — bright tissue on a black canvas (the
    atlas scale is reversed, see ``ATLAS_REVERSE``) — because a stat overlay in
    a warm colormap has to sit on a dark background to read.

    ``affine`` is the voxel→world affine of *both* volumes (they share a grid).
    It is what puts the slice the right way up: the layout follows
    ``SLICE_ORIENT`` regardless of array storage order, and the edges are marked
    with the corresponding anatomical letters (L/R, A/P, S/I). Passing ``None``
    keeps the old unlabeled ``rot90`` layout.

    ``colorscale`` names the overlay colour map (any Plotly colorscale, e.g.
    ``"Hot"``); ``None`` falls back to the module default. Sub-threshold voxels
    are NaN'd (``_threshold``) so they render fully transparent — i.e. alpha=0
    below ``z_threshold`` — and everything at/above uses the chosen scale."""
    if atlas is None:
        return empty_fig(title, height, dark=True)
    axis = int(axis)
    idx = int(np.clip(slice_idx, 0, atlas.shape[axis] - 1))
    orient = slice_orientation(affine, axis)
    bg = _oriented(atlas, axis, idx, orient)
    ov = _oriented(overlay, axis, idx, orient) if overlay is not None else None

    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=bg, colorscale=ATLAS_COLORSCALE, reversescale=ATLAS_REVERSE,
                             showscale=False, hoverinfo="skip"))
    ov_t = _threshold(ov, z_threshold)
    if ov_t is not None and not np.all(np.isnan(ov_t)):
        fig.add_trace(go.Heatmap(
            z=ov_t, colorscale=(colorscale or OVERLAY_COLORSCALE), opacity=opacity,
            showscale=True, zmin=vmin, zmax=vmax,
            colorbar=dict(title=dict(text="z", font=dict(color=SLICE_INK)), len=0.6,
                          thickness=10, tickfont=dict(color=SLICE_INK), outlinewidth=0),
            hoverinfo="skip"))
    if show_crosshair and cross:
        cx, cy = cross
        fig.add_shape(type="line", x0=cx, x1=cx, y0=0, y1=bg.shape[0] - 1,
                      line=dict(color="cyan", width=1, dash="dot"))
        fig.add_shape(type="line", x0=0, x1=bg.shape[1] - 1, y0=cy, y1=cy,
                      line=dict(color="cyan", width=1, dash="dot"))
    fig.update_layout(
        title=dict(text=title, font=dict(size=12, color=SLICE_INK)),
        margin=dict(l=4, r=4, t=26, b=4),
        xaxis=dict(visible=False, scaleanchor="y", constrain="domain"),
        yaxis=dict(visible=False, constrain="domain"),
        plot_bgcolor=SLICE_BG, paper_bgcolor=SLICE_BG, font_color=SLICE_INK, height=height,
        annotations=_orientation_annotations(orient, bg.shape))
    return fig


def _orientation_annotations(orient, shape):
    """L/R, A/P, S/I letters just inside the four edges of the slice.

    Anchored in *data* coordinates rather than paper coordinates: the x axis is
    scale-anchored with ``constrain="domain"``, so the plotting domain shrinks to
    the brain's aspect ratio and paper-anchored labels would drift away from it.
    """
    if orient is None:
        return []
    nr, nc = shape
    left, right, bottom, top = orient[3]
    font = dict(size=13, color=SLICE_INK)
    pad_x, pad_y = 0.02 * nc, 0.02 * nr
    return [
        dict(x=pad_x, y=(nr - 1) / 2, text=left, xanchor="left", yanchor="middle",
             showarrow=False, font=font),
        dict(x=nc - 1 - pad_x, y=(nr - 1) / 2, text=right, xanchor="right", yanchor="middle",
             showarrow=False, font=font),
        dict(x=(nc - 1) / 2, y=pad_y, text=bottom, xanchor="center", yanchor="bottom",
             showarrow=False, font=font),
        dict(x=(nc - 1) / 2, y=nr - 1 - pad_y, text=top, xanchor="center", yanchor="top",
             showarrow=False, font=font),
    ]


def _brain_surface_trace(atlas_lowres, iso=0.12, max_dim=64):
    """Faint translucent brain surface (go.Isosurface) for 3D anatomical context.

    The atlas is strided down so the surface dim stays <= max_dim, keeping the
    mesh light enough for the human MNI grid.
    """
    if atlas_lowres is None:
        return None
    a = atlas_lowres
    stride = max(1, int(np.ceil(max(a.shape) / max_dim)))
    a = a[::stride, ::stride, ::stride]
    X, Y, Z = np.mgrid[0:a.shape[0], 0:a.shape[1], 0:a.shape[2]] * stride
    return go.Isosurface(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(), value=a.flatten(),
        isomin=iso, isomax=iso, surface_count=1, showscale=False,
        colorscale=[[0, SURFACE_COLOR], [1, SURFACE_COLOR]], opacity=0.18,
        caps=dict(x_show=False, y_show=False, z_show=False), hoverinfo="skip")


def make_volume_fig(overlay_lowres, z_threshold, vmin, vmax, title="", height=420,
                    atlas_lowres=None):
    """3D rendering of the supra-threshold z-map over a faint brain surface.

    The overlay is drawn as a Scatter3d point cloud of supra-threshold voxels:
    statistical maps are sparse (tens of voxels), for which ``go.Volume`` renders
    essentially nothing, so markers are used instead so the result is actually
    visible. ``atlas_lowres`` (on the overlay grid) adds a translucent brain
    surface for anatomical context.
    """
    if overlay_lowres is None:
        return empty_fig(title, height)
    data = overlay_lowres
    supra = np.abs(data) >= float(z_threshold)
    n_supra = int(supra.sum())

    def _layout(fig, suffix):
        fig.update_layout(
            title=dict(text=f"{title} {suffix}", font=dict(size=12, color=INK)),
            margin=dict(l=0, r=0, t=26, b=0), height=height,
            paper_bgcolor=PANEL_BG, font_color=INK,
            scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False),
                       zaxis=dict(visible=False), bgcolor=PANEL_BG, aspectmode="data"))
        return fig

    fig = go.Figure()
    surf = _brain_surface_trace(atlas_lowres)
    if surf is not None:
        fig.add_trace(surf)
    if n_supra == 0:
        if surf is None:
            return empty_fig(title, height)
        return _layout(fig, "(0 supra-threshold voxels)")

    xs, ys, zs = np.where(supra)
    cvals = data[xs, ys, zs]
    # Larger square markers read as contiguous blobs (closer to a regular fMRI
    # overlay) than the sparse pin-prick dots that small circles produce.
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs, mode="markers",
        marker=dict(size=9, symbol="square", color=cvals, colorscale=OVERLAY_COLORSCALE,
                    cmin=vmin, cmax=vmax, opacity=1.0, line=dict(width=0),
                    colorbar=dict(title="z", len=0.6, thickness=12)),
        hovertemplate="z=%{marker.color:.2f}<br>voxel (%{x}, %{y}, %{z})"
                      "<extra>click to name region</extra>"))
    return _layout(fig, f"({n_supra} voxels)")
