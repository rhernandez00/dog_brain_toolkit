#!/usr/bin/env python
"""
fsf_design.py — read an FSL FEAT ``.fsf`` design file and swap in a new
EV / contrast model, leaving everything else byte-for-byte alone.

This is the engine behind :mod:`glm_designer`. It exists as a separate,
Dash-free module so the rewrite can be scripted or unit-tested without a
browser.

What it touches
---------------
A FEAT ``.fsf`` is a flat Tcl script of ``set fmri(key) value`` lines grouped
into comment-headed blocks separated by blank lines. This module splits the
template into those blocks and classifies each one as

* **EV**    — ``evtitle{n}``, ``shape{n}``, ``convolve{n}``, ``convolve_phase{n}``,
  ``tempfilt_yn{n}``, ``deriv_yn{n}``, ``custom{n}``, ``ortho{n}.{m}``, …
* **contrast** — ``con_mode``, ``conpic_{real,orig}.{c}``,
  ``conname_{real,orig}.{c}``, ``con_{real,orig}{c}.{e}``, ``conmask{i}_{j}``, …
* **keep**  — everything else (paths, TR, smoothing, registration, thresholds).

The EV blocks and the contrast blocks are dropped and regenerated from the
model you pass in; the regenerated blocks are inserted at the position the
originals occupied. ``keep`` blocks pass through untouched apart from the seven
counters that *have* to follow the model (``evs_orig``, ``evs_real``,
``evs_vox``, ``ncon_orig``, ``ncon_real``, ``nftests_orig``, ``nftests_real``).

Per-EV settings (waveform shape, convolution, temporal filtering, temporal
derivative) are **read out of the template's first EV** and reused for every
generated EV — the point of this tool is that those knobs stay where the
supplied design already put them. Only the EV *names*, the EV *custom files*
and the *contrasts* are yours to set.

orig vs real contrasts
----------------------
The generated file is written in ``con_mode orig``: you give one weight per
condition and FEAT's "real" design matrix is derived. When an EV carries a
temporal derivative it occupies two real columns, so real column of EV *i* is
``1 + sum(1 + deriv_yn(k) for k < i)`` and the derivative column gets 0. With
the usual uniform ``deriv_yn 1`` that is the familiar ``2*i - 1``.

F-tests are **not** generated. :func:`template_summary` reports how many the
template had so the caller can warn that they will be dropped.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

# --- line / key classification --------------------------------------------

# `set fmri(key) value`  — the only statement shape this module rewrites.
_SET_RE = re.compile(r'^\s*set\s+fmri\(([^)]+)\)\s+(.*?)\s*$')

_EV_KEY_RE = re.compile(
    r'^(?:'
    r'(?:evtitle|shape|convolve|convolve_phase|tempfilt_yn|deriv_yn|custom'
    r'|gammasigma|gammadelay|basisfnum|basisfwidth|bfcustom|basisorth'
    r'|skip|off|on|phase|stop|period)\d+'
    r'|ortho\d+\.\d+'
    r'|interactions\d+\.\d+'
    r')$'
)

_CON_KEY_RE = re.compile(
    r'^(?:'
    r'con_mode|con_mode_old'
    r'|conpic_(?:real|orig)\.\d+'
    r'|conname_(?:real|orig)\.\d+'
    r'|con_(?:real|orig)\d+\.\d+'
    r'|conmask\d+_\d+'
    r'|conmask_zerothresh_yn'
    r'|ftest_(?:real|orig)\d+\.\d+'
    r')$'
)

# Counters that must agree with the model we write.
_COUNT_KEYS = ('evs_orig', 'evs_real', 'evs_vox',
               'ncon_orig', 'ncon_real', 'nftests_orig', 'nftests_real')


class FsfError(ValueError):
    """Raised when a template cannot be understood well enough to rewrite."""


# --- data model ------------------------------------------------------------

@dataclass
class EVOptions:
    """Per-EV FEAT settings, lifted from the template's first EV."""
    shape: int = 3            # 3 = custom, 3-column format
    convolve: int = 3         # 3 = double-gamma HRF
    convolve_phase: float = 0
    tempfilt_yn: int = 1
    deriv_yn: int = 1

    @property
    def real_per_ev(self) -> int:
        return 2 if int(self.deriv_yn) else 1


@dataclass
class Contrast:
    """One contrast in *orig* space: a weight per condition."""
    name: str
    weights: list = field(default_factory=list)


# --- parsing ---------------------------------------------------------------

def read_template(path):
    """Read a ``.fsf`` file as text (FEAT writes plain ASCII/latin-1)."""
    with open(path, 'r', encoding='utf-8', errors='replace') as fh:
        return fh.read()


def parse_settings(text):
    """Return ``{key: raw_value}`` for every ``set fmri(key) value`` line.

    Values keep their surrounding quotes exactly as written; use
    :func:`_unquote` when you want the bare string.
    """
    out = {}
    for line in text.splitlines():
        m = _SET_RE.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _unquote(val):
    if val is None:
        return None
    val = val.strip()
    if len(val) >= 2 and val[0] == '"' and val[-1] == '"':
        return val[1:-1]
    return val


def _as_int(val, default=0):
    try:
        return int(float(_unquote(val)))
    except (TypeError, ValueError):
        return default


def ev_options_from_template(text, ev_index=1):
    """Lift the per-EV settings out of the template's EV ``ev_index``.

    Falls back to FEAT's usual first-level defaults for anything the template
    does not define, so a stripped-down template still produces a valid design.
    """
    s = parse_settings(text)
    i = ev_index
    if f'evtitle{i}' not in s and f'shape{i}' not in s:
        raise FsfError(
            f"template defines no EV {i} — cannot infer the per-EV settings "
            f"(shape / convolution / temporal derivative) to reuse."
        )
    return EVOptions(
        shape=_as_int(s.get(f'shape{i}'), 3),
        convolve=_as_int(s.get(f'convolve{i}'), 3),
        convolve_phase=_as_int(s.get(f'convolve_phase{i}'), 0),
        tempfilt_yn=_as_int(s.get(f'tempfilt_yn{i}'), 1),
        deriv_yn=_as_int(s.get(f'deriv_yn{i}'), 1),
    )


def template_ev_names(text):
    """EV titles in the template, in EV order."""
    s = parse_settings(text)
    n = _as_int(s.get('evs_orig'), 0)
    return [_unquote(s.get(f'evtitle{i}', '')) or f'EV{i}' for i in range(1, n + 1)]


def template_custom_dir(text, ev_index=1):
    """Directory holding the template's custom EV timing files (may be '')."""
    s = parse_settings(text)
    custom = _unquote(s.get(f'custom{ev_index}', ''))
    if not custom:
        return ''
    # FEAT designs are written on Linux; keep POSIX separators.
    return custom.rsplit('/', 1)[0] if '/' in custom else os.path.dirname(custom)


def template_contrasts(text):
    """Read the template's *orig* contrasts back out as :class:`Contrast` objects."""
    s = parse_settings(text)
    n_con = _as_int(s.get('ncon_orig'), 0)
    n_ev = _as_int(s.get('evs_orig'), 0)
    out = []
    for c in range(1, n_con + 1):
        weights = []
        for e in range(1, n_ev + 1):
            try:
                weights.append(float(_unquote(s.get(f'con_orig{c}.{e}', '0'))))
            except ValueError:
                weights.append(0.0)
        out.append(Contrast(name=_unquote(s.get(f'conname_orig.{c}', f'con{c}')),
                            weights=weights))
    return out


def template_summary(text):
    """A dict of the facts the UI needs to describe a loaded template."""
    s = parse_settings(text)
    n_ev = _as_int(s.get('evs_orig'), 0)
    try:
        opts = ev_options_from_template(text)
    except FsfError:
        opts = None
    return {
        'n_evs': n_ev,
        'n_evs_real': _as_int(s.get('evs_real'), 0),
        'n_contrasts': _as_int(s.get('ncon_orig'), 0),
        'n_ftests': max(_as_int(s.get('nftests_orig'), 0),
                        _as_int(s.get('nftests_real'), 0)),
        'ev_names': template_ev_names(text),
        'custom_dir': template_custom_dir(text),
        'outputdir': _unquote(s.get('outputdir', '')),
        'options': opts,
        'level': _as_int(s.get('level'), 1),
    }


# --- block splitting -------------------------------------------------------

def _split_blocks(text):
    """Split into blank-line-separated blocks, preserving each block verbatim."""
    return re.split(r'\n[ \t]*\n', text)


def _block_kind(block):
    """Classify a block as ``'ev'``, ``'con'`` or ``'keep'``."""
    keys = [m.group(1) for m in
            (_SET_RE.match(ln) for ln in block.splitlines()) if m]
    if not keys:
        return 'keep'
    if all(_EV_KEY_RE.match(k) for k in keys):
        return 'ev'
    if all(_CON_KEY_RE.match(k) for k in keys):
        return 'con'
    return 'keep'


def _rewrite_counts(block, counts):
    """Rewrite the values of any counter keys appearing in ``block``."""
    out = []
    for line in block.splitlines():
        m = _SET_RE.match(line)
        if m and m.group(1) in counts:
            out.append(f'set fmri({m.group(1)}) {counts[m.group(1)]}')
        else:
            out.append(line)
    return '\n'.join(out)


# --- generation ------------------------------------------------------------

def _fmt_weight(v):
    """Format a contrast weight the way FEAT writes them (``0``, ``1.0``, ``-0.5``)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        v = 0.0
    if v == 0:
        return '0'
    s = f'{v:g}'
    if '.' not in s and 'e' not in s and 'E' not in s:
        s += '.0'
    return s


_SHAPE_COMMENT = (
    "# 0 : Square\n"
    "# 1 : Sinusoid\n"
    "# 2 : Custom (1 entry per volume)\n"
    "# 3 : Custom (3 column format)\n"
    "# 4 : Interaction\n"
    "# 10 : Empty (all zeros)"
)

_CONVOLVE_COMMENT = (
    "# 0 : None\n"
    "# 1 : Gaussian\n"
    "# 2 : Gamma\n"
    "# 3 : Double-Gamma HRF\n"
    "# 4 : Gamma basis functions\n"
    "# 5 : Sine basis functions\n"
    "# 6 : FIR basis functions"
)


def _ev_blocks(ev_names, custom_files, opts):
    """Generate every block of the EV section, in FEAT's own order."""
    n = len(ev_names)
    blocks = []
    for i, name in enumerate(ev_names, start=1):
        custom = custom_files[i - 1] if i - 1 < len(custom_files) else ''
        blocks.append(f'# EV {i} title\nset fmri(evtitle{i}) "{name}"')
        blocks.append(f'# Basic waveform shape (EV {i})\n{_SHAPE_COMMENT}\n'
                      f'set fmri(shape{i}) {opts.shape}')
        blocks.append(f'# Convolution (EV {i})\n{_CONVOLVE_COMMENT}\n'
                      f'set fmri(convolve{i}) {opts.convolve}')
        blocks.append(f'# Convolve phase (EV {i})\n'
                      f'set fmri(convolve_phase{i}) {opts.convolve_phase}')
        blocks.append(f'# Apply temporal filtering (EV {i})\n'
                      f'set fmri(tempfilt_yn{i}) {opts.tempfilt_yn}')
        blocks.append(f'# Add temporal derivative (EV {i})\n'
                      f'set fmri(deriv_yn{i}) {opts.deriv_yn}')
        blocks.append(f'# Custom EV file (EV {i})\nset fmri(custom{i}) "{custom}"')
        # FEAT writes ortho against EV 0 (the mean) through EV n.
        for j in range(0, n + 1):
            blocks.append(f'# Orthogonalise EV {i} wrt EV {j}\n'
                          f'set fmri(ortho{i}.{j}) 0')
    return blocks


def _real_columns(n_evs, opts):
    """Real-design column index (1-based) of each orig EV."""
    cols, col = [], 1
    for _ in range(n_evs):
        cols.append(col)
        col += opts.real_per_ev
    return cols


def _contrast_blocks(contrasts, n_evs, opts, zerothresh=0):
    """Generate every block of the contrast section, in FEAT's own order."""
    n_con = len(contrasts)
    n_real = n_evs * opts.real_per_ev
    real_col = _real_columns(n_evs, opts)

    blocks = ['# Contrast & F-tests mode\n'
              '# real : control real EVs\n'
              '# orig : control original EVs\n'
              'set fmri(con_mode_old) orig\n'
              'set fmri(con_mode) orig']

    # Real contrasts: the orig weight lands on the EV's own column, the
    # temporal-derivative column stays 0.
    for c, con in enumerate(contrasts, start=1):
        real = [0.0] * n_real
        for e, w in enumerate(con.weights[:n_evs]):
            real[real_col[e] - 1] = w
        blocks.append(f'# Display images for contrast_real {c}\n'
                      f'set fmri(conpic_real.{c}) 1')
        blocks.append(f'# Title for contrast_real {c}\n'
                      f'set fmri(conname_real.{c}) "{con.name}"')
        for e in range(1, n_real + 1):
            blocks.append(f'# Real contrast_real vector {c} element {e}\n'
                          f'set fmri(con_real{c}.{e}) {_fmt_weight(real[e - 1])}')

    for c, con in enumerate(contrasts, start=1):
        blocks.append(f'# Display images for contrast_orig {c}\n'
                      f'set fmri(conpic_orig.{c}) 1')
        blocks.append(f'# Title for contrast_orig {c}\n'
                      f'set fmri(conname_orig.{c}) "{con.name}"')
        for e in range(1, n_evs + 1):
            w = con.weights[e - 1] if e - 1 < len(con.weights) else 0.0
            blocks.append(f'# Real contrast_orig vector {c} element {e}\n'
                          f'set fmri(con_orig{c}.{e}) {_fmt_weight(w)}')

    blocks.append('# Contrast masking - use >0 instead of thresholding?\n'
                  f'set fmri(conmask_zerothresh_yn) {zerothresh}')
    for i in range(1, n_con + 1):
        for j in range(1, n_con + 1):
            if i == j:
                continue
            blocks.append(f'# Mask real contrast/F-test {i} with real contrast/F-test {j}?\n'
                          f'set fmri(conmask{i}_{j}) 0')
    blocks.append('# Do contrast masking at all?\nset fmri(conmask1_1) 0')
    return blocks


# --- the rewrite ------------------------------------------------------------

def build_fsf(template_text, ev_names, custom_files, contrasts, opts=None):
    """Return a new ``.fsf`` text: the template with a new EV/contrast model.

    Parameters
    ----------
    template_text : str
        Contents of the supplied design file. Everything outside the EV and
        contrast sections is carried over unchanged.
    ev_names : list[str]
        One condition name per EV, in order — normally the config's ``stim_types``.
    custom_files : list[str]
        Timing file (3-column format) for each EV, same order and length.
    contrasts : list[Contrast]
        Contrasts in *orig* space; each ``weights`` list is one weight per EV.
    opts : EVOptions, optional
        Per-EV settings. Defaults to those of the template's EV 1.

    Raises
    ------
    FsfError
        If the template has no EV section, or the arguments disagree in length.
    """
    if not ev_names:
        raise FsfError("no conditions — nothing to write.")
    if len(custom_files) != len(ev_names):
        raise FsfError(f"{len(ev_names)} conditions but {len(custom_files)} "
                       f"timing files.")
    for con in contrasts:
        if len(con.weights) != len(ev_names):
            raise FsfError(f"contrast '{con.name}' has {len(con.weights)} weights "
                           f"but there are {len(ev_names)} conditions.")

    if opts is None:
        opts = ev_options_from_template(template_text)

    n_ev = len(ev_names)
    counts = {
        'evs_orig': n_ev,
        'evs_real': n_ev * opts.real_per_ev,
        'evs_vox': 0,
        'ncon_orig': len(contrasts),
        'ncon_real': len(contrasts),
        'nftests_orig': 0,
        'nftests_real': 0,
    }
    zerothresh = _as_int(parse_settings(template_text).get('conmask_zerothresh_yn'), 0)

    blocks = _split_blocks(template_text)
    kinds = [_block_kind(b) for b in blocks]
    if 'ev' not in kinds:
        raise FsfError("template has no EV section — is this a first-level design?")

    first_ev = kinds.index('ev')
    first_con = kinds.index('con') if 'con' in kinds else None

    new_ev = _ev_blocks(ev_names, custom_files, opts)
    new_con = _contrast_blocks(contrasts, n_ev, opts, zerothresh)

    out = []
    for idx, (block, kind) in enumerate(zip(blocks, kinds)):
        if kind == 'ev':
            if idx == first_ev:
                out.extend(new_ev)
                # A template with no contrast section at all still needs one.
                if first_con is None:
                    out.extend(new_con)
            continue
        if kind == 'con':
            if idx == first_con:
                out.extend(new_con)
            continue
        out.append(_rewrite_counts(block, counts))

    return '\n\n'.join(out).rstrip('\n') + '\n'


def write_fsf(path, text):
    """Write ``text`` to ``path`` with Unix line endings (FEAT runs on Linux)."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(text)
    return path


# --- helpers the UI leans on ------------------------------------------------

def custom_files_for(custom_dir, ev_names, ext='.txt'):
    """Timing-file path per condition: ``{custom_dir}/{name}.txt``.

    Matches the convention ``rsa_utils.calculate_beta_maps`` uses when it fills
    ``set fmri(custom{i})`` per subject/session/run, so the generated design is a
    drop-in template for the pipeline.
    """
    d = (custom_dir or '').rstrip('/\\')
    return [f'{d}/{name}{ext}' if d else f'{name}{ext}' for name in ev_names]


def identity_contrasts(ev_names):
    """One contrast per condition, weight 1 on itself — FEAT's usual starting point."""
    out = []
    for i, name in enumerate(ev_names):
        w = [0.0] * len(ev_names)
        w[i] = 1.0
        out.append(Contrast(name=name, weights=w))
    return out


def mean_contrasts_by_group(ev_names, groups, normalise=True):
    """One contrast per distinct group value, averaging that group's conditions.

    ``groups`` is a per-condition list of group labels (e.g. the config's
    ``label`` or ``specie_shown`` field). Conditions with an empty label are
    skipped. With ``normalise`` the weights are ``1/n`` so the contrast is a
    mean rather than a sum.
    """
    order, members = [], {}
    for i, g in enumerate(groups):
        if g in (None, '', 'nan'):
            continue
        g = str(g)
        if g not in members:
            members[g] = []
            order.append(g)
        members[g].append(i)

    out = []
    for g in order:
        idxs = members[g]
        w = [0.0] * len(ev_names)
        val = 1.0 / len(idxs) if normalise else 1.0
        for i in idxs:
            w[i] = val
        out.append(Contrast(name=g, weights=w))
    return out
