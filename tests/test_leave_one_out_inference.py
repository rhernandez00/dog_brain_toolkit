"""Synthetic checks for the step-11 group inference and report continuation."""
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
import searchlight


class LeaveOneOutInferenceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.mask = self.root / 'mask.nii.gz'
        mask = np.ones((3, 3, 3))
        mask[0, 0, 0] = 0
        self.save(self.mask, mask)
        self.kw = dict(datafolder=str(self.root), dataset='tiny', specie='H',
                       model='basic', task='tiny_mean_task', radius=3,
                       rsa_class='all', dis_method='mahalanobis',
                       rsa_method='kendall', mah_fold='run-wise',
                       mask=str(self.mask), mask_type='mask', reps_group=10,
                       z_threshold=2., cluster_threshold=.05)
        args = {k: self.kw[k] for k in ('datafolder', 'dataset', 'specie', 'model',
                'task', 'radius', 'rsa_class', 'dis_method', 'rsa_method',
                'mah_fold', 'mask_type')}
        self.real = Path(rsa._loo_group_path(**args))
        self.rnd = Path(rsa._loo_group_path(**args, rnd=True))
        self.stem = self.real.name
        rng = np.random.default_rng(457)
        self.maps = []
        for i in range(10):
            data = rng.normal(size=(3, 3, 3))
            data[2, 2, 2] = 7  # zero variance yields z=0
            data[0, 0, 0] = np.nan  # ignored outside the analysis mask
            self.maps.append(data)
            self.save(self.rnd.parent / 'mean' / f'{self.stem}_mean_{i:05d}.nii.gz', data)
        self.save(str(self.real) + '_mean.nii.gz', np.full((3, 3, 3), 20.))
        Path(str(self.real) + '_mean.json').write_text(json.dumps(
            dict(participants_included=[1, 2], rsa_class='class_all')))
        mirror = patch('publication_report.mirror_results')
        self.mirror = mirror.start()
        self.addCleanup(mirror.stop)
        quiet = contextlib.redirect_stdout(io.StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)

    @staticmethod
    def save(path, data, affine=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(np.asarray(data, dtype=float),
                                np.eye(4) if affine is None else affine), path)

    def run_step(self, step, **kwargs):
        return rsa.calculate_leave_one_out_inference(step, **dict(self.kw, **kwargs))

    def test_numerics_clusters_reports_and_input_isolation(self):
        self.assertTrue(self.run_step('11.6'))
        dist = self.rnd.parent / 'dist'
        mean = nib.load(dist / f'{self.stem}_mean.nii.gz').get_fdata()
        std = nib.load(dist / f'{self.stem}_std.nii.gz').get_fdata()
        np.testing.assert_allclose(mean[1:], np.mean(self.maps, axis=0)[1:], atol=1e-6)
        np.testing.assert_allclose(std[1:], np.std(self.maps, axis=0)[1:], atol=1e-6)
        self.assertEqual(mean[0, 0, 0], 0)
        # An unrelated leftover must not contribute to the cluster null.
        self.save(self.rnd.parent / 'mean' / f'{self.stem}_z_99999.nii.gz',
                  np.full((3, 3, 3), 100.))
        self.assertTrue(self.run_step('11.7'))
        z = nib.load(self.real.parent / 'mean' / f'{self.stem}_z.nii.gz').get_fdata()
        self.assertEqual(z[2, 2, 2], 0)
        self.assertEqual(z[0, 0, 0], 0)
        np.testing.assert_allclose(z[1, 1, 1], (20 - mean[1, 1, 1]) / std[1, 1, 1], rtol=1e-6)
        self.assertTrue(self.run_step('11.8'))
        cluster_path = self.real.parent / 'dist' / f'{self.stem}_dist.npy'
        entry = np.load(cluster_path, allow_pickle=True).item()['z2.0']
        self.assertEqual(entry['number_of_images'], 10)
        for path, sizes in zip(entry['file_list'], entry['cluster_sizes']):
            _, expected = rsa._count_on_3d(nib.load(path).get_fdata(), 2., False, connectivity=3)
            self.assertEqual(sizes, list(expected))
        self.assertTrue(self.run_step('11.9'))
        corrected = self.real.parent / 'mean' / f'{self.stem}_zt2.0_corrected.nii.gz'
        self.assertEqual(np.count_nonzero(nib.load(corrected).get_fdata()), 25)
        self.assertTrue(self.run_step('11.101'))
        csv = corrected.with_name(f'{self.stem}_zt2.0.csv')
        self.assertEqual(len(pd.read_csv(csv)), 1)
        for suffix in ('_publication.tex', '_results.txt', '_report.json'):
            self.assertTrue(csv.with_name(csv.stem + suffix).is_file())
        self.assertIn('RSA_same', self.mirror.call_args.args[0][0][1])
        with self.assertRaises(FileNotFoundError):
            self.run_step('11.7', task='different')
        with self.assertRaises(FileNotFoundError):
            self.run_step('11.7', mah_fold='stim-wise')

    def test_missing_permutations_changed_parameters_and_nonfinite_inputs(self):
        path = self.rnd.parent / 'mean' / f'{self.stem}_mean_00009.nii.gz'
        path.unlink()
        self.assertFalse(self.run_step('11.6'))
        self.assertTrue(self.run_step('11.6', min_percentage_available=.9))
        with self.assertRaisesRegex(ValueError, 'parameters changed'):
            self.run_step('11.7', reps_group=11)
        with self.assertRaisesRegex(ValueError, 'coverage is insufficient'):
            self.run_step('11.7')
        self.assertTrue(self.run_step('11.7', min_percentage_available=.9))
        bad = np.ones((3, 3, 3))
        bad[1, 1, 1] = np.nan
        self.save(path, bad)
        self.assertTrue(self.run_step('11.6'))
        std = self.rnd.parent / 'dist' / f'{self.stem}_std.nii.gz'
        self.assertEqual(nib.load(std).get_fdata()[1, 1, 1], 0)

    def test_nonfinite_voxels_are_excluded_from_real_and_entire_null(self):
        # Both sporadic and persistent undefined correlations occur upstream.
        for i, values in enumerate(self.maps):
            data = values.copy()
            data[1, 0, 0] = np.nan
            if i == 3:
                data[1, 0, 1] = np.nan
                data[1, 0, 2] = np.inf
            self.save(self.rnd.parent / 'mean' / f'{self.stem}_mean_{i:05d}.nii.gz', data)
        real = np.full((3, 3, 3), 20.)
        real[1, 1, 0] = np.nan
        self.save(str(self.real) + '_mean.nii.gz', real)
        for step in ('11.6', '11.7', '11.8', '11.9', '11.10'):
            self.assertTrue(self.run_step(step))
        receipt = json.loads((self.rnd.parent / 'dist' / f'{self.stem}_distribution.json').read_text())
        self.assertEqual(receipt['n_excluded_nonfinite_voxels'], 3)
        z_paths = list((self.rnd.parent / 'mean').glob('*_z_*.nii.gz'))
        z_paths.append(self.real.parent / 'mean' / f'{self.stem}_z.nii.gz')
        self.assertEqual(len(z_paths), 11)
        for path in z_paths:
            z = nib.load(path).get_fdata()
            self.assertTrue(np.isfinite(z).all())
            for voxel in ((1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 1, 0)):
                self.assertEqual(z[voxel], 0)
        np.testing.assert_allclose(
            nib.load(z_paths[-1]).get_fdata()[1, 1, 1],
            (20 - np.mean(self.maps, axis=0)[1, 1, 1]) / np.std(self.maps, axis=0)[1, 1, 1],
            rtol=1e-6)
        corrected = self.real.parent / 'mean' / f'{self.stem}_zt2.0_corrected.nii.gz'
        self.assertEqual(np.count_nonzero(nib.load(corrected).get_fdata()), 21)

    def test_empty_result_and_grid_validation(self):
        self.save(str(self.real) + '_mean.nii.gz', np.zeros((3, 3, 3)))
        for step in ('11.6', '11.7', '11.8', '11.9', '11.10'):
            self.assertTrue(self.run_step(step))
        csv = self.real.parent / 'mean' / f'{self.stem}_zt2.0.csv'
        self.assertTrue(pd.read_csv(csv).empty)
        self.save(self.rnd.parent / 'mean' / f'{self.stem}_mean_00000.nii.gz',
                  np.ones((3, 3, 3)), affine=np.diag([2, 2, 2, 1]))
        with self.assertRaises(ValueError):
            self.run_step('11.6')

    def test_cli_report_ids_do_not_collide_with_11_1(self):
        for value in ('11.10', '11.101'):
            with patch('sys.argv', ['searchlight.py', '--steps_to_run',
                                   '11.6', '11.7', '11.8', '11.9', value]):
                self.assertEqual(searchlight.parse_arguments().steps_to_run,
                                 [11.6, 11.7, 11.8, 11.9, '11.10'])
        self.assertEqual(searchlight._parse_step('11.1'), 11.1)
        self.assertEqual(searchlight._parse_step('15.10'), '15.10')


if __name__ == '__main__':
    unittest.main()
