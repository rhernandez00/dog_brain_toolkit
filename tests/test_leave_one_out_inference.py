"""Synthetic checks for the step-11 group inference and report continuation."""
import contextlib
import io
import json
import shutil
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
            _, expected = rsa._count_on_3d(nib.load(self.root / 'tiny' / path).get_fdata(), 2., False, connectivity=3)
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

    def test_receipts_survive_dataset_relocation(self):
        for step in ('11.6', '11.7', '11.8', '11.9'):
            self.assertTrue(self.run_step(step))
        receipt_path = self.rnd.parent / 'dist' / f'{self.stem}_distribution.json'
        receipt = json.loads(receipt_path.read_text())
        self.assertEqual(receipt['receipt_version'], 2)
        self.assertFalse(any('\\' in p or Path(p).is_absolute() for p in receipt['file_list']))
        with tempfile.TemporaryDirectory() as destination:
            moved = Path(destination) / 'moved'
            shutil.copytree(self.root, moved)
            self.assertTrue(self.run_step('11.10', datafolder=str(moved),
                                          mask=str(moved / self.mask.name)))
        receipt['receipt_version'] = 1
        receipt_path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, 'receipt_version'):
            self.run_step('11.7')

    def test_external_mask_never_requires_dataset_relative_path(self):
        # Simulate Windows relpath failing for a C: mask against a P: dataset.
        original_relpath = rsa.os.path.relpath

        def cross_drive_relpath(path, start):
            if str(path) == str(self.mask):
                raise ValueError("path is on mount 'C:', start on mount 'P:'")
            return original_relpath(path, start)

        with patch.object(rsa.os.path, 'relpath', side_effect=cross_drive_relpath):
            for step in ('11.6', '11.7', '11.8', '11.9', '11.10'):
                self.assertTrue(self.run_step(step))
        with tempfile.TemporaryDirectory() as destination:
            external_mask = Path(destination) / 'other_name.nii.gz'
            shutil.copy2(self.mask, external_mask)
            self.assertTrue(self.run_step('11.10', mask=str(external_mask)))
            changed = np.ones((3, 3, 3))
            self.save(external_mask, changed)
            with self.assertRaisesRegex(ValueError, 'parameters changed.*mask'):
                self.run_step('11.10', mask=str(external_mask))

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

    def test_similarity_remains_positive_tail(self):
        self.save(str(self.real) + '_mean.nii.gz', np.full((3, 3, 3), -20.))
        for step in ('11.6', '11.7', '11.8', '11.9', '11.10'):
            self.assertTrue(self.run_step(step))
        corrected = self.real.parent / 'mean' / f'{self.stem}_zt2.0_corrected.nii.gz'
        self.assertFalse(nib.load(corrected).get_fdata().any())
        receipt = json.loads(corrected.with_name(corrected.name[:-7] + '.json').read_text())
        self.assertEqual(receipt['tail'], 'positive')

    def test_cli_report_ids_do_not_collide_with_11_1(self):
        for value in ('11.10', '11.101'):
            with patch('sys.argv', ['searchlight.py', '--steps_to_run',
                                   '11.6', '11.7', '11.8', '11.9', value]):
                self.assertEqual(searchlight.parse_arguments().steps_to_run,
                                 [11.6, 11.7, 11.8, 11.9, '11.10'])
        self.assertEqual(searchlight._parse_step('11.1'), 11.1)
        self.assertEqual(searchlight._parse_step('15.10'), '15.10')


class ClassDifferenceInferenceTests(unittest.TestCase):
    def setUp(self):
        fixture = LeaveOneOutInferenceTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.kw = dict(fixture.kw, target_class='dog')
        args = {key: self.kw[key] for key in (
            'datafolder', 'dataset', 'specie', 'model', 'task', 'radius',
            'rsa_class', 'target_class', 'dis_method', 'rsa_method', 'mah_fold', 'mask_type')}
        self.real = Path(rsa._loo_group_path(**args))
        self.rnd = Path(rsa._loo_group_path(**args, rnd=True))
        self.stem = self.real.name
        self.real_mean = self.real.parent / 'mean' / f'{self.stem}_mean.nii.gz'
        fixture.save(self.real_mean, np.full((3, 3, 3), 20.))
        self.real_mean.with_name(f'{self.stem}_mean.json').write_text(json.dumps(
            dict(participants_included=[1, 2], rsa_class='all', target_class='dog')))
        for i, data in enumerate(fixture.maps):
            fixture.save(self.rnd.parent / 'mean' / f'{self.stem}_mean_{i:05d}.nii.gz', data)

    def run_step(self, step, **kwargs):
        return rsa.calculate_leave_one_out_inference(step, **dict(self.kw, **kwargs))

    def test_complete_difference_pipeline_uses_its_own_null_and_exports(self):
        self.assertTrue(self.run_step('11.16'))
        dist = self.rnd.parent / 'dist'
        mean = nib.load(dist / f'{self.stem}_mean.nii.gz').get_fdata()
        std = nib.load(dist / f'{self.stem}_std.nii.gz').get_fdata()
        np.testing.assert_allclose(mean[1:], np.mean(self.fixture.maps, axis=0)[1:], atol=1e-6)
        np.testing.assert_allclose(std[1:], np.std(self.fixture.maps, axis=0)[1:], atol=1e-6)
        # Existing class-only and unrelated permutation maps must never enter.
        self.fixture.save(self.rnd.parent / 'mean' / f'{self.stem}_mean_99999.nii.gz',
                          np.full((3, 3, 3), -1000.))
        self.fixture.save(self.rnd.parent / 'mean' / f'{self.stem}_z_99999.nii.gz',
                          np.full((3, 3, 3), 1000.))
        for step in ('11.17', '11.18', '11.19', '11.20'):
            self.assertTrue(self.run_step(step))
        z = nib.load(self.real.parent / 'mean' / f'{self.stem}_z.nii.gz').get_fdata()
        self.assertAlmostEqual(z[1, 1, 1], (20 - mean[1, 1, 1]) / std[1, 1, 1], places=5)
        self.assertEqual(z[2, 2, 2], 0)  # zero null variance
        entry = np.load(self.real.parent / 'dist' / f'{self.stem}_dist.npy', allow_pickle=True).item()['z2.0']
        self.assertEqual(entry['number_of_images'], 10)
        self.assertEqual(entry['target_class'], 'dog')
        corrected = self.real.parent / 'mean' / f'{self.stem}_zt2.0_corrected.nii.gz'
        self.assertEqual(np.count_nonzero(nib.load(corrected).get_fdata()), 25)
        csv = corrected.with_name(f'{self.stem}_zt2.0.csv')
        self.assertEqual(len(pd.read_csv(csv)), 1)
        report = json.loads(csv.with_name(csv.stem + '_report.json').read_text())
        self.assertEqual(report['rsa_model'], 'all - dog')
        self.assertEqual(report['correction']['tail'], 'two-sided')
        for suffix in ('_publication.tex', '_results.txt', '_report.json'):
            self.assertTrue(csv.with_name(csv.stem + suffix).is_file())
        self.assertIn('class_dog', self.fixture.mirror.call_args.args[0][0][1])

    def test_both_tails_share_maximum_null_and_keep_touching_clusters_separate(self):
        # A single permutation has touching clusters of sizes 3 and 4 with
        # opposite signs. Its maximum must be 4, not their merged extent 7.
        positive = [(0, 0, 1), (0, 0, 2), (0, 1, 0)]
        negative = [(1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 1, 0)]
        maps = []
        for i in range(10):
            data = np.full((3, 3, 3), (-1.) ** i)
            extreme = 3. if i == 0 else -1. / 3.
            for voxel in positive:
                data[voxel] = extreme
            for voxel in negative:
                data[voxel] = -extreme
            maps.append(data)
            self.fixture.save(self.rnd.parent / 'mean' / f'{self.stem}_mean_{i:05d}.nii.gz', data)
        desired_z = np.zeros((3, 3, 3))
        desired_z[0] = 4.
        desired_z[1] = -4.
        desired_z[0, 0, 1] = 6.
        desired_z[1, 0, 0] = -8.
        self.fixture.save(self.real_mean, np.mean(maps, axis=0) + desired_z * np.std(maps, axis=0))
        for step in ('11.16', '11.17', '11.18', '11.19', '11.20'):
            self.assertTrue(self.run_step(step))
        cluster_path = self.real.parent / 'dist' / f'{self.stem}_dist.npy'
        entry = np.load(cluster_path, allow_pickle=True).item()['z2.0']
        self.assertEqual(entry['cluster_sizes'], [[3, 4]] + [[] for _ in range(9)])
        self.assertEqual(rsa.get_minimal_cluster_size(str(cluster_path), 2., .05), 5)
        corrected = self.real.parent / 'mean' / f'{self.stem}_zt2.0_corrected.nii.gz'
        data = nib.load(corrected).get_fdata()
        self.assertEqual(np.count_nonzero(data > 0), 8)
        self.assertEqual(np.count_nonzero(data < 0), 9)
        receipt = json.loads(corrected.with_name(corrected.name[:-7] + '.json').read_text())
        self.assertEqual((receipt['n_clusters'], receipt['n_voxels']), (2, 17))
        csv = corrected.with_name(f'{self.stem}_zt2.0.csv')
        table = pd.read_csv(csv)
        np.testing.assert_allclose(table.subpeak_Z, [6., -8.])
        report = json.loads(csv.with_name(csv.stem + '_report.json').read_text())
        self.assertIn('|Z| > 2', report['results_text'])
        self.assertIn('minimum (peak', report['results_text'])
        self.assertEqual(report['clusters'][1]['peaks'][0]['z'], -8.)
        # Legacy positive-only nulls and correction receipts require rebuilding.
        legacy = dict(entry, tail='positive')
        np.save(cluster_path, {'z2.0': legacy})
        for step in ('11.19', '11.20'):
            with self.assertRaisesRegex(ValueError, 'Cluster null parameters changed'):
                self.run_step(step)
        self.assertTrue(self.run_step('11.18'))
        receipt['tail'] = 'positive'
        corrected.with_name(corrected.name[:-7] + '.json').write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, 'Correction parameters changed'):
            self.run_step('11.20')

    def test_negative_only_result_and_strict_thresholds(self):
        for step in ('11.16', '11.17'):
            self.assertTrue(self.run_step(step))
        # Test the cluster stages directly with known signed z-maps.
        for i in range(10):
            self.fixture.save(self.rnd.parent / 'mean' / f'{self.stem}_z_{i:05d}.nii.gz',
                              np.zeros((3, 3, 3)))
        z = np.zeros((3, 3, 3))
        z[1, 1, 1] = -5.
        z[0, 1, 1] = -2.  # exact cutoffs are excluded in both tails
        z[2, 1, 1] = 2.
        self.fixture.save(self.real.parent / 'mean' / f'{self.stem}_z.nii.gz', z)
        for step in ('11.18', '11.19', '11.20'):
            self.assertTrue(self.run_step(step))
        csv = self.real.parent / 'mean' / f'{self.stem}_zt2.0.csv'
        table = pd.read_csv(csv)
        self.assertEqual(len(table), 1)
        self.assertEqual(table.subpeak_Z.iloc[0], -5.)
        self.assertEqual(table.cluster_size_vox.iloc[0], 1)

    def test_missing_coverage_and_changed_inputs_fail(self):
        missing = self.rnd.parent / 'mean' / f'{self.stem}_mean_00009.nii.gz'
        missing.unlink()
        self.assertFalse(self.run_step('11.16'))
        self.assertTrue(self.run_step('11.16', min_percentage_available=.9))
        with self.assertRaisesRegex(ValueError, 'coverage is insufficient'):
            self.run_step('11.17')
        with self.assertRaisesRegex(ValueError, 'parameters changed'):
            self.run_step('11.17', reps_group=11)
        self.assertTrue(self.run_step('11.17', min_percentage_available=.9))
        source = self.rnd.parent / 'mean' / f'{self.stem}_mean_00000.nii.gz'
        self.fixture.save(source, np.ones((3, 3, 3)))
        with self.assertRaisesRegex(ValueError, 'inputs changed'):
            self.run_step('11.18', min_percentage_available=.9)
        with self.assertRaises(FileNotFoundError):
            self.run_step('11.17', target_class='other')

    def test_nonfinite_voxels_and_empty_reports(self):
        path = self.rnd.parent / 'mean' / f'{self.stem}_mean_00000.nii.gz'
        values = self.fixture.maps[0].copy()
        values[1, 0, 0] = np.nan
        values[1, 0, 1] = np.inf
        self.fixture.save(path, values)
        self.fixture.save(self.real_mean, np.zeros((3, 3, 3)))
        for step in ('11.16', '11.17', '11.18', '11.19', '11.20'):
            self.assertTrue(self.run_step(step))
        real_z = self.real.parent / 'mean' / f'{self.stem}_z.nii.gz'
        paths = [real_z] + list((self.rnd.parent / 'mean').glob('*_z_*.nii.gz'))
        for path in paths:
            data = nib.load(path).get_fdata()
            self.assertTrue(np.isfinite(data).all())
            self.assertEqual(data[1, 0, 0], 0)
            self.assertEqual(data[1, 0, 1], 0)
        csv = self.real.parent / 'mean' / f'{self.stem}_zt2.0.csv'
        self.assertTrue(pd.read_csv(csv).empty)
        shifted = np.eye(4)
        shifted[0, 3] = 10
        self.fixture.save(self.rnd.parent / 'mean' / f'{self.stem}_mean_00000.nii.gz',
                          np.ones((3, 3, 3)), affine=shifted)
        with self.assertRaises(rsa.SpaceMismatchError):
            self.run_step('11.16')

    def test_cli_preserves_report_step_and_requires_classes(self):
        with patch('sys.argv', ['searchlight.py', '--steps_to_run', '11.16', '11.17',
                                '11.18', '11.19', '11.20', '--rsa_class', 'all',
                                '--target_class', 'dog']):
            self.assertEqual(searchlight.parse_arguments().steps_to_run,
                             [11.16, 11.17, 11.18, 11.19, '11.20'])
        self.assertEqual(searchlight._parse_step('11.2'), 11.2)
        with patch('sys.argv', ['searchlight.py', '--steps_to_run', '11.20', '--rsa_class', 'all']):
            with self.assertRaisesRegex(ValueError, '--target_class'):
                searchlight.main()


if __name__ == '__main__':
    unittest.main()
