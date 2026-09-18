"""Synthetic NIfTI checks for searchlight steps 15, 15.3, 15.4 and 15.5."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pandas as pd

import rsa_utils as rsa


class RegressionPipelineTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.mask = self.root / 'mask.nii.gz'
        self.save_map(self.mask, [1, 0])
        self.units = {1: [{'session': 1, 'run_N': 1}, {'session': 1, 'run_N': 2}],
                      2: [{'session': 2, 'run_N': 1}]}
        self.common = dict(datafolder=str(self.root), dataset='test',
                           session_and_run_all_dict=self.units,
                           regression_model='controls', specie='H', model='basic',
                           task='test', radius=4, dis_method='correlation',
                           rsa_models_list=['target'], mask=str(self.mask))
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)

    @staticmethod
    def save_map(path, values, affine=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(np.array(values, dtype=float).reshape(2, 1, 1),
                                np.eye(4) if affine is None else affine), str(path))

    def run_folder(self, sub, session, run, rnd=False):
        return (self.root / 'test/results' /
                ('RSA_regression_rnd' if rnd else 'RSA_regression') /
                f'basic/controls/target/H-sub-{sub:02d}/'
                f'ses-{session:02d}_task-test_run-{run:02d}')

    def group_path(self, stat='mean', rnd_index=None):
        root = 'RSA_regression' if rnd_index is None else 'RSA_regression_rnd'
        suffix = '' if rnd_index is None else f'_{rnd_index:05d}'
        return (self.root / f'test/results/{root}/basic/controls/target/mean/'
                f'H-r-4_correlation_beta_{stat}{suffix}.nii.gz')

    def write_betas(self, rnd=False):
        for (sub, session, run), value in zip([(1, 1, 1), (1, 1, 2), (2, 2, 1)],
                                             [1., 3., 8.]):
            folder = self.run_folder(sub, session, run, rnd)
            for index in (range(2) if rnd else [None]):
                paths = rsa._regression_map_paths(str(folder), 4, 'correlation', index)
                self.save_map(Path(paths['beta_map_path']),
                              [value + (10 * index if rnd else 0), 500])
            # A huge t-map must never enter the beta average.
            self.save_map(folder / 'r-4_correlation_t_map.nii.gz', [1000, 1000])

    def test_observed_mean_std_mask_and_resume(self):
        self.write_betas()
        self.assertTrue(rsa.calculate_group_regression_maps(**self.common))
        np.testing.assert_allclose(nib.load(self.group_path()).get_fdata().ravel(), [4, 0])
        np.testing.assert_allclose(nib.load(self.group_path('std')).get_fdata().ravel(),
                                   [np.std([1, 3, 8]), 0])
        with patch.object(rsa, 'nifti_mean_stream', side_effect=AssertionError('not cached')):
            self.assertTrue(rsa.calculate_group_regression_maps(**self.common))

    def test_group_permutations_select_one_per_run_and_honor_reps(self):
        self.write_betas(rnd=True)
        for sub, entries in self.units.items():
            for entry in entries:
                folder = self.run_folder(sub, entry['session'], entry['run_N'], True)
                self.save_map(folder / 'r-4_correlation_beta_map_0002.nii.gz', [999, 999])
        with patch.object(rsa.random, 'choice', side_effect=lambda pool: pool[-1]):
            self.assertTrue(rsa.calculate_group_regression_maps(
                **self.common, rnd=True, reps=2, reps_group=3))
        for index in range(3):
            np.testing.assert_allclose(nib.load(self.group_path(rnd_index=index)).get_fdata().ravel(),
                                       [14, 0])
            receipt = json.loads(self.group_path(rnd_index=index).with_suffix('').with_suffix('.json').read_text())
            self.assertEqual(len(receipt['file_list']), 3)
            self.assertTrue(all(p.endswith('_0001.nii.gz') for p in receipt['file_list']))

    def test_missing_coverage_and_partial_cache_refresh(self):
        folder = self.run_folder(1, 1, 1)
        self.save_map(folder / 'r-4_correlation_beta_map.nii.gz', [1, 0])
        self.assertFalse(rsa.calculate_group_regression_maps(**self.common))
        self.assertFalse(self.group_path().exists())
        self.assertTrue(rsa.calculate_group_regression_maps(
            **self.common, min_percentage_available=0.3))
        self.write_betas()
        self.assertTrue(rsa.calculate_group_regression_maps(**self.common))
        self.assertEqual(nib.load(self.group_path()).get_fdata()[0, 0, 0], 4)
        missing_target = dict(self.common, rsa_models_list=['target', 'absent'])
        self.assertFalse(rsa.calculate_group_regression_maps(**missing_target))

    def test_grid_mismatch_is_rejected(self):
        self.write_betas()
        affine = np.eye(4)
        affine[0, 3] = 20
        self.save_map(self.run_folder(2, 2, 1) / 'r-4_correlation_beta_map.nii.gz',
                      [8, 0], affine)
        with self.assertRaises(rsa.SpaceMismatchError):
            rsa.calculate_group_regression_maps(**self.common)

    def write_models(self):
        folder = self.root / 'test/rsa_models'
        (folder / 'regression_models').mkdir(parents=True)
        (folder / 'regression_models/controls.csv').write_text('visual\n')
        vectors = {'target': [0, 1, 0, 1, 0, 1],
                   'visual-run-1': [1, 2, 5, 4, 3, 7],
                   'visual-run-2': [8, 1, 4, 3, 5, 2]}
        for name, vector in vectors.items():
            matrix = np.zeros((4, 4))
            matrix[np.triu_indices(4, 1)] = vector
            matrix += matrix.T
            pd.DataFrame(matrix, index=list('ABCD'), columns=list('ABCD')).to_csv(folder / f'{name}.csv')
        return vectors

    def test_permuted_fits_keep_controls_and_neural_data_and_resume(self):
        vectors = self.write_models()
        neural = np.array([0.2, 1.5, -0.1, 2.4, 1.2, 0.3]).reshape(1, 1, 1, 6)
        neural = np.concatenate([neural, neural], axis=0)
        designs = []
        solver = rsa.perform_multiple_regression_rsa

        def record(data, design, *args, **kwargs):
            np.testing.assert_array_equal(data, neural)
            designs.append(design.copy())
            return solver(data, design, *args, **kwargs)

        args = dict(self.common, participants=[1, 2], verbose=False)
        with patch.object(rsa, 'load_meta_similarity_map', return_value=neural) as load, \
                patch.object(rsa, 'perform_multiple_regression_rsa', side_effect=record), \
                patch.object(rsa.random, 'shuffle', side_effect=lambda labels: labels.reverse()):
            self.assertTrue(rsa.calculate_multiple_regression_rsa(**args, rnd=True, reps=2))
            self.assertEqual(load.call_count, 3)  # once per run, not per permutation
        permuted = np.array(vectors['target'])[[5, 4, 2, 3, 1, 0]]
        self.assertFalse(np.array_equal(permuted, vectors['target']))
        for fit, run in zip(designs, [1, 1, 2, 2, 1, 1]):
            np.testing.assert_array_equal(fit[:, 0], 1)
            np.testing.assert_array_equal(fit[:, 1], permuted)
            np.testing.assert_array_equal(fit[:, 2], vectors[f'visual-run-{run}'])
        self.assertFalse(self.run_folder(1, 1, 1).exists())
        paths = rsa._regression_map_paths(str(self.run_folder(1, 1, 1, True)), 4, 'correlation', 0)
        receipt = json.loads(Path(paths['sidecar_path']).read_text())
        self.assertEqual(receipt['target_vector'], permuted.tolist())
        expected = solver(neural, designs[0], np.array([True, False]).reshape(2, 1, 1))[0]
        np.testing.assert_allclose(nib.load(paths['beta_map_path']).get_fdata(), expected, rtol=1e-6)
        with patch.object(rsa, 'load_meta_similarity_map', side_effect=AssertionError('not cached')):
            self.assertTrue(rsa.calculate_multiple_regression_rsa(**args, rnd=True, reps=2))
        Path(paths['t_map_path']).unlink()
        with patch.object(rsa, 'load_meta_similarity_map', return_value=neural) as load:
            self.assertTrue(rsa.calculate_multiple_regression_rsa(**args, rnd=True, reps=2))
            self.assertEqual(load.call_count, 1)
        self.assertTrue(Path(paths['t_map_path']).exists())
        with patch.object(rsa, 'load_meta_similarity_map', return_value=neural) as load:
            self.assertTrue(rsa.calculate_multiple_regression_rsa(**args))
            self.assertEqual(load.call_count, 3)
        self.assertTrue((self.run_folder(1, 1, 1) / 'r-4_correlation_beta_map.nii.gz').exists())


if __name__ == '__main__':
    unittest.main()
