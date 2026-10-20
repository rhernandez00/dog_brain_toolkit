"""Numerical and file-layout regression checks for steps 11.2/11.3."""
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

import rsa_utils as rsa


class LeaveOneOutTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        config = self.root / 'EmoB/config_files'
        config.mkdir(parents=True)
        (config / 'H_basic.yaml').write_text('model: basic\ntask: EmoB\n')
        classes = self.root / 'EmoB/rsa_models/by_class'
        classes.mkdir(parents=True)
        self.pairs = [('A', 'B'), ('A', 'C'), ('B', 'C')]
        pd.DataFrame(self.pairs, columns=['cat1', 'cat2']).to_csv(classes / 'class_all.csv', index=False)
        self.mask = self.root / 'mask.nii.gz'
        self.save(self.mask, [1., 0.])
        self.units = {1: [dict(session=1, run_N=1), dict(session=2, run_N=1),
                          dict(session=1, run_N=2)],
                      2: [dict(session=1, run_N=1), dict(session=1, run_N=2)],
                      3: [dict(session=1, run_N=1), dict(session=1, run_N=2)]}
        self.vectors = {1: np.array([0., 1., 4.]),
                        2: np.array([3., 2., 0.]),
                        3: np.array([0., 4., 1.])}
        self.kw = dict(datafolder=str(self.root), dataset='EmoB',
                       session_and_run_all_dict=self.units, participants=[1, 2, 3],
                       specie='H', mask=str(self.mask), model='basic', task='EmoB',
                       radius=3, rsa_class='all', mah_fold='run-wise', mask_type='mask')

    def save(self, path, values, affine=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(np.array(values, dtype=float).reshape(2, 1, 1),
                               np.eye(4) if affine is None else affine), path)

    def make_inputs(self, method):
        self.kw['dis_method'] = method
        for sub, vector in self.vectors.items():
            entries = [(None, None)] if method == 'mahalanobis' else [
                (e['session'], e['run_N']) for e in self.units[sub]]
            for session, run in entries:
                folder = self.root / f'EmoB/results/RSA/basic/H-sub-{sub:02d}'
                if run is not None:
                    folder /= f'ses-{session:02d}_task-EmoB_run-{run:02d}'
                for (a, b), value in zip(self.pairs, vector):
                    # Reverse pair order to exercise the existing input fallback.
                    self.save(folder / f'r-3_{method}_{b}_{a}.nii.gz', [value, np.nan])

    def test_mahalanobis_self_exclusion_and_group_arithmetic_mean(self):
        self.make_inputs('mahalanobis')
        for method, correlate in [('kendall', kendalltau), ('spearman', spearmanr), ('pearson', pearsonr)]:
            self.kw['rsa_method'] = method
            paths = rsa.calculate_leave_one_out_cross_participant_similarity(**self.kw)
            self.assertEqual(len(paths), 3)  # Never repeat pooled maps for runs.
            expected = []
            for sub, path in zip(self.vectors, paths):
                peers = np.mean([v for other, v in self.vectors.items() if other != sub], axis=0)
                expected.append(correlate(self.vectors[sub], peers)[0])
                np.testing.assert_allclose(nib.load(path).get_fdata().ravel(), [expected[-1], 0.])
                self.assertIn('RSA_same/basic/class_all/leave-one-out/run-wise/H-sub-', Path(path).as_posix())
                self.assertTrue(path.endswith(f'r-3_mahalanobis_{method}_task-EmoB.nii.gz'))
            mean, std = rsa.calculate_mean_leave_one_out_cross_participant_similarity_map(**self.kw)
            np.testing.assert_allclose(nib.load(mean).get_fdata().ravel(), [np.mean(expected), 0.])
            np.testing.assert_allclose(nib.load(std).get_fdata().ravel(), [np.std(expected), 0.])

    def test_correlation_equal_reference_weight_and_group_weight(self):
        self.make_inputs('correlation')
        self.kw['rsa_method'] = 'pearson'
        extra = np.array([5., 0., 2.])
        folder = self.root / 'EmoB/results/RSA/basic/H-sub-01/ses-02_task-EmoB_run-01'
        for (a, b), value in zip(self.pairs, extra):
            self.save(folder / f'r-3_correlation_{b}_{a}.nii.gz', [value, 999.])
        paths = rsa.calculate_leave_one_out_cross_participant_similarity(**self.kw)
        self.assertEqual(len(paths), 7)
        expected = []
        for sub in self.vectors:
            run_means = []
            for run in (1, 2):
                refs = {s: ((v + extra) / 2 if s == 1 and run == 1 else v)
                        for s, v in self.vectors.items()}
                peers = np.mean([v for s, v in refs.items() if s != sub], axis=0)
                targets = [self.vectors[sub], extra] if sub == 1 and run == 1 else [self.vectors[sub]]
                run_means.append(np.mean([pearsonr(v, peers)[0] for v in targets]))
            expected.append(np.mean(run_means))
        mean, std = rsa.calculate_mean_leave_one_out_cross_participant_similarity_map(**self.kw)
        self.assertAlmostEqual(nib.load(mean).get_fdata()[0, 0, 0], np.mean(expected))
        self.assertAlmostEqual(nib.load(std).get_fdata()[0, 0, 0], np.std(expected))
        # Resuming does not overwrite existing maps.
        before = [Path(p).stat().st_mtime_ns for p in paths]
        rsa.calculate_leave_one_out_cross_participant_similarity(**self.kw)
        self.assertEqual(before, [Path(p).stat().st_mtime_ns for p in paths])

    def test_missing_inputs_and_grid_mismatch_fail(self):
        self.make_inputs('mahalanobis')
        path = self.root / 'EmoB/results/RSA/basic/H-sub-02/r-3_mahalanobis_B_A.nii.gz'
        path.unlink()
        with self.assertWarns(UserWarning), self.assertRaises(FileNotFoundError):
            rsa.calculate_leave_one_out_cross_participant_similarity(**self.kw)
        affine = np.eye(4)
        affine[0, 3] = 10.
        self.save(path, [3., 0.], affine)
        with self.assertRaises(rsa.SpaceMismatchError):
            rsa.calculate_leave_one_out_cross_participant_similarity(**self.kw)

    def test_missing_group_map_obeys_completeness_threshold(self):
        self.make_inputs('mahalanobis')
        paths = rsa.calculate_leave_one_out_cross_participant_similarity(**self.kw)
        Path(paths[0]).unlink()
        with self.assertWarnsRegex(UserWarning, 'incomplete'), self.assertRaises(ValueError):
            rsa.calculate_mean_leave_one_out_cross_participant_similarity_map(
                **self.kw, min_percentage_available=1)
        with self.assertWarnsRegex(UserWarning, 'incomplete'):
            mean, _ = rsa.calculate_mean_leave_one_out_cross_participant_similarity_map(**self.kw)
        expected = np.mean([nib.load(p).get_fdata() for p in paths[1:]], axis=0)
        np.testing.assert_allclose(nib.load(mean).get_fdata(), expected)

    def test_constant_rdm_is_undefined_inside_mask(self):
        self.vectors[1] = np.ones(3)
        self.make_inputs('mahalanobis')
        paths = rsa.calculate_leave_one_out_cross_participant_similarity(**self.kw)
        self.assertTrue(np.isnan(nib.load(paths[0]).get_fdata()[0, 0, 0]))
        self.assertEqual(nib.load(paths[0]).get_fdata()[1, 0, 0], 0.)

    def test_absent_run_with_class_pairs(self):
        self.units[3] = [dict(session=1, run_N=1)]
        self.make_inputs('correlation')
        with self.assertWarnsRegex(UserWarning, 'Run 2: excluding H-sub-03'):
            paths = rsa.calculate_leave_one_out_cross_participant_similarity(**self.kw)
        self.assertEqual(len(paths), 6)
        for path in paths:
            if 'run-02' in path:
                self.assertAlmostEqual(nib.load(path).get_fdata()[0, 0, 0],
                                       kendalltau(self.vectors[1], self.vectors[2])[0])

    def test_class_rows_are_authoritative_and_rsa_model_is_ignored(self):
        for method in ('mahalanobis', 'correlation'):
            self.make_inputs(method)
            class_path = self.root / 'EmoB/rsa_models/by_class/class_subset.csv'
            # This set deliberately omits A/C, although all three categories occur.
            selected = [0, 2]
            pd.DataFrame([self.pairs[i] for i in selected], columns=['cat1', 'cat2']).to_csv(
                class_path, index=False)
            self.kw.update(rsa_class='subset', rsa_model='nonexistent_model')
            with patch.object(rsa, 'read_model_dict', side_effect=AssertionError('Must not load RSA model')):
                paths = rsa.calculate_leave_one_out_cross_participant_similarity(**self.kw)
                mean, _ = rsa.calculate_mean_leave_one_out_cross_participant_similarity_map(**self.kw)
            expected = []
            for sub, vector in self.vectors.items():
                peers = np.mean([v[selected] for s, v in self.vectors.items() if s != sub], axis=0)
                value = kendalltau(vector[selected], peers)[0]
                expected.append(value)
                for path in paths:
                    if f'H-sub-{sub:02d}' in path:
                        self.assertAlmostEqual(nib.load(path).get_fdata()[0, 0, 0], value)
                        self.assertIn('/class_subset/', Path(path).as_posix())
            self.assertAlmostEqual(nib.load(mean).get_fdata()[0, 0, 0], np.mean(expected))
            self.kw['rsa_class'] = 'class_subset'
            self.assertEqual(paths, rsa.calculate_leave_one_out_cross_participant_similarity(**self.kw))

    def test_rsa_model_cannot_replace_missing_class(self):
        self.kw.update(rsa_class=None, rsa_model='source')
        for function in (rsa.calculate_leave_one_out_cross_participant_similarity,
                         rsa.calculate_mean_leave_one_out_cross_participant_similarity_map):
            with self.assertRaisesRegex(ValueError, 'require rsa_class'):
                function(**self.kw)

    def test_class_requires_cat1_and_cat2(self):
        self.make_inputs('mahalanobis')
        pd.DataFrame({'wrong': ['A', 'B']}).to_csv(
            self.root / 'EmoB/rsa_models/by_class/class_all.csv', index=False)
        with self.assertRaisesRegex(ValueError, 'cat1 and cat2'):
            rsa.calculate_leave_one_out_cross_participant_similarity(**self.kw)

    def test_group_rejects_affine_mismatch(self):
        self.make_inputs('mahalanobis')
        paths = rsa.calculate_leave_one_out_cross_participant_similarity(**self.kw)
        affine = np.eye(4)
        affine[1, 3] = 5.
        self.save(Path(paths[0]), [1., 0.], affine)
        with self.assertRaises(rsa.SpaceMismatchError):
            rsa.calculate_mean_leave_one_out_cross_participant_similarity_map(**self.kw)


if __name__ == '__main__':
    unittest.main()
