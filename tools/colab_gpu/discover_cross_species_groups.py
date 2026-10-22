"""Discover completed step-16.2 cohorts in an existing workstation/Drive tree.

Only metadata is read here. Numerical work stays in cross_species_rsa.
"""
import json
from pathlib import Path
import warnings

import h5py


def discover_groups(root):
    """Return keyword dictionaries for cross_species_rsa.group_cross_similarity.

    Discover every dataset/model/class/direction/radius combination. Include
    only completed participant results, never group files or snapshots that
    are still incomplete. No configs, masks or original distance maps needed.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f'Step-16.2 results directory does not exist: {root}')
    groups = {}
    pattern = '*/results/RSA_cross-specie/*/class_*/*-sub-*/*.HDF5'
    for path in sorted(root.glob(pattern)):
        dataset, _, _, model, class_folder, subject, _ = path.relative_to(root).parts
        with h5py.File(path, 'r') as handle:
            if not handle.attrs.get('complete', False):
                warnings.warn(f'Skipping incomplete step-16.2 result: {path}')
                continue
            source = handle.attrs.get('source_specie')
            target = handle.attrs.get('target_specie')
            participant = handle.attrs.get('source_participant')
            if {source, target} != {'D', 'H'} or participant is None:
                raise ValueError(f'Expected a participant-to-target-group step-16.2 file: {path}')
            participant = int(participant)
            if subject != f'{source}-sub-{participant:02d}':
                raise ValueError(f'Participant folder disagrees with saved metadata: {path}')
            identity = json.loads(handle.attrs.get('identity', '{}'))
            masks = identity.get('settings', {}).get('masks', {})
            mask_type = handle[source].attrs.get('mask_type', masks.get(source))
            target_mask_type = handle[target].attrs.get('mask_type', masks.get(target))
            if mask_type is None:
                raise ValueError(f'Source mask name is missing from saved metadata: {path}')
            radius = float(handle.attrs['radius'])
            target_radius = float(handle.attrs['target_radius'])
            expected_name = f'{mask_type}-r-{radius:g}_mahalanobis_kendall_vs_{target}-r-{target_radius:g}.HDF5'
            if path.name != expected_name:
                raise ValueError(f'Filename disagrees with saved step-16.2 settings: {path}')
            # Different target cohorts/masks cannot share an output filename.
            # Validate their identity within one group instead of silently
            # scheduling multiple analyses that would overwrite one another.
            key = (dataset, model, class_folder[6:], source, target, radius, target_radius, mask_type)
            target_ids = sorted(int(p) for p in handle[f'{target}/participants'][:])
        if key not in groups:
            groups[key] = dict(root=root, dataset=dataset, model=model,
                rsa_class=class_folder[6:], specie=source, target_specie=target,
                radius=radius, target_radius=target_radius, mask_type=mask_type,
                target_mask_type=target_mask_type, participants=[], target_participants=target_ids)
        group = groups[key]
        if group['target_mask_type'] != target_mask_type or group['target_participants'] != target_ids:
            raise ValueError(f'Conflicting target cohort/mask within one analysis: {path}')
        if participant in group['participants']:
            raise ValueError(f'Duplicate source participant: {path}')
        group['participants'].append(participant)
    if not groups:
        raise FileNotFoundError(f'No complete step-16.2 participant HDF5 files found under {root}')
    analyses = list(groups.values())
    for analysis in analyses:
        analysis['participants'].sort()
    return analyses
