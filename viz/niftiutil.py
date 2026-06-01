"""Pure NIfTI / rendering helpers shared by the viewer apps.

No Dash, no global state — every function takes its inputs explicitly so the
same code serves the dual-species viewer, the legacy single-species viewer,
and (later) the static failsafe exporter.
"""

import os
import numpy as np
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

OVERLAY_COLORSCALE = "Hot"
ATLAS_COLORSCALE = "Gray"
PANEL_BG = "#1a1a2e"

# Above this many voxels we switch 3D from go.Volume to a Scatter3d point
# cloud of supra-threshold voxels (keeps the conference demo responsive).
VOLUME_VOXEL_CAP = 300_000


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


# --- Figures --------------------------------------------------------------

def empty_fig(title="", height=360):
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        margin=dict(l=5, r=5, t=30, b=5),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor="black", paper_bgcolor=PANEL_BG, font_color="white", height=height,
        annotations=[dict(text="No data", showarrow=False, font=dict(size=15, color="#555"),
                          xref="paper", yref="paper", x=0.5, y=0.5)],
    )
    return fig


def _threshold(ov, z_threshold):
    """Return overlay with |value| < threshold set to NaN (None passes through)."""
    if ov is None:
        return None
    return np.where(np.abs(ov) >= float(z_threshold), ov, np.nan)


def make_slice_fig(atlas, overlay, axis, slice_idx, opacity, z_threshold,
                   vmin, vmax, show_crosshair=False, cross=None, title="", height=360):
    """One orthogonal slice: grayscale atlas + thresholded hot overlay."""
    if atlas is None:
        return empty_fig(title, height)
    idx = int(np.clip(slice_idx, 0, atlas.shape[axis] - 1))
    if axis == 0:
        bg = np.rot90(atlas[idx, :, :]); ov = overlay[idx, :, :] if overlay is not None else None
    elif axis == 1:
        bg = np.rot90(atlas[:, idx, :]); ov = overlay[:, idx, :] if overlay is not None else None
    else:
        bg = np.rot90(atlas[:, :, idx]); ov = overlay[:, :, idx] if overlay is not None else None
    if ov is not None:
        ov = np.rot90(ov)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=bg, colorscale=ATLAS_COLORSCALE, showscale=False, hoverinfo="skip"))
    ov_t = _threshold(ov, z_threshold)
    if ov_t is not None and not np.all(np.isnan(ov_t)):
        fig.add_trace(go.Heatmap(
            z=ov_t, colorscale=OVERLAY_COLORSCALE, opacity=opacity, showscale=True,
            zmin=vmin, zmax=vmax, colorbar=dict(title="z", len=0.6, thickness=10),
            hoverinfo="skip"))
    if show_crosshair and cross:
        cx, cy = cross
        fig.add_shape(type="line", x0=cx, x1=cx, y0=0, y1=bg.shape[0] - 1,
                      line=dict(color="cyan", width=1, dash="dot"))
        fig.add_shape(type="line", x0=0, x1=bg.shape[1] - 1, y0=cy, y1=cy,
                      line=dict(color="cyan", width=1, dash="dot"))
    fig.update_layout(
        title=dict(text=title, font=dict(size=12)),
        margin=dict(l=4, r=4, t=26, b=4),
        xaxis=dict(visible=False, scaleanchor="y", constrain="domain"),
        yaxis=dict(visible=False, constrain="domain"),
        plot_bgcolor="black", paper_bgcolor=PANEL_BG, font_color="white", height=height)
    return fig


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
        colorscale=[[0, "#9aa6c8"], [1, "#9aa6c8"]], opacity=0.12,
        caps=dict(x_show=False, y_show=False, z_show=False), hoverinfo="skip")


def make_volume_fig(overlay_lowres, z_threshold, vmin, vmax, title="", height=420,
                    atlas_lowres=None):
    """3D rendering of the supra-threshold z-map over a faint brain surface.

    Uses Plotly ``go.Volume`` when the grid is small enough; otherwise falls
    back to a Scatter3d point cloud of supra-threshold voxels. When
    ``atlas_lowres`` is supplied a translucent brain isosurface is drawn for
    anatomical context.
    """
    if overlay_lowres is None:
        return empty_fig(title, height)
    data = overlay_lowres
    supra = np.abs(data) >= float(z_threshold)
    n_supra = int(supra.sum())
    fig = go.Figure()
    surf = _brain_surface_trace(atlas_lowres)
    if surf is not None:
        fig.add_trace(surf)
    if n_supra == 0:
        if surf is None:
            return empty_fig(title, height)
        fig.update_layout(
            title=dict(text=f"{title} (0 supra-threshold vox)", font=dict(size=12)),
            margin=dict(l=0, r=0, t=26, b=0), height=height,
            paper_bgcolor=PANEL_BG, font_color="white",
            scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False),
                       zaxis=dict(visible=False), bgcolor="black", aspectmode="data"))
        return fig

    if data.size <= VOLUME_VOXEL_CAP:
        X, Y, Z = np.mgrid[0:data.shape[0], 0:data.shape[1], 0:data.shape[2]]
        vals = np.where(supra, data, np.nan)
        fig.add_trace(go.Volume(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(), value=vals.flatten(),
            isomin=float(vmin) if vmin is not None else None,
            isomax=float(vmax) if vmax is not None else None,
            opacity=0.15, surface_count=15, colorscale=OVERLAY_COLORSCALE,
            caps=dict(x_show=False, y_show=False, z_show=False),
            colorbar=dict(title="z", len=0.6, thickness=10)))
        mode = "go.Volume"
    else:
        xs, ys, zs = np.where(supra)
        cvals = data[xs, ys, zs]
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="markers",
            marker=dict(size=2.5, color=cvals, colorscale=OVERLAY_COLORSCALE,
                        cmin=vmin, cmax=vmax, opacity=0.6,
                        colorbar=dict(title="z", len=0.6, thickness=10)),
            hoverinfo="skip"))
        mode = "point cloud"
    fig.update_layout(
        title=dict(text=f"{title} ({mode}, {n_supra} vox)", font=dict(size=12)),
        margin=dict(l=0, r=0, t=26, b=0), height=height,
        paper_bgcolor=PANEL_BG, font_color="white",
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False),
                   zaxis=dict(visible=False), bgcolor="black",
                   aspectmode="data"))
    return fig
