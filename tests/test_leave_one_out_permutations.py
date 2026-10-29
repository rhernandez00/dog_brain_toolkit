"""Numerical checks for the participant and group leave-one-out null maps."""
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import test_leave_one_out_similarity as fixtures
import rsa_utils as rsa


class LeaveOneOutPermutationTests(unittest.TestCase):
    setUp = fixtures.LeaveOneOutTests.setUp
    save = fixtures.LeaveOneOutTests.save
    make_inputs = fixtures.LeaveOneOutTests.make_inputs

    def test_randomized_creation_order_for_participant_and_group(self):
        self.make_inputs('mahalanobis')
        with patch.object(rsa.random, 'shuffle', side_effect=lambda items: items.reverse()), \
                patch.object(rsa.nib, 'save', wraps=nib.save) as save:
            rsa.calculate_leave_one_out_cross_participant_similarity(**self.kw, rnd=True, reps=3)
        written = [Path(call.args[1]).name[-11:-7] for call in save.call_args_list]
        self.assertEqual(written, ['0002', '0001', '0000'] * 3)
        first_path = Path(save.call_args_list[0].args[1])
        self.assertEqual(first_path.parent, self.root / (
            'EmoB/results/RSA_same_rnd/basic/class_all/leave-one-out/run-wise/H-sub-01'))
        self.assertEqual(first_path.name, 'mask-r-3_mahalanobis_kendall_0002.nii.gz')
        with patch.object(rsa.random, 'shuffle', side_effect=lambda items: items.reverse()), \
                patch.object(rsa.nib, 'save', wraps=nib.save) as save:
            rsa.calculate_mean_leave_one_out_cross_participant_similarity_map_rnd(
                **self.kw, reps=3, reps_group=3)
        written = [Path(call.args[1]).name for call in save.call_args_list]
        self.assertEqual([name[-12:-7] for name in written if '_mean_' in name],
                         ['00002', '00001', '00000'])

    def test_group_rechecks_files_created_after_initial_inventory(self):
        self.make_inputs('mahalanobis')
        rsa.calculate_leave_one_out_cross_participant_similarity(**self.kw, rnd=True, reps=1)

        def another_instance_finishes(entries):
            for mean, std, log in entries:
                self.save(Path(mean), [17, 0])
                self.save(Path(std), [0, 0])
                Path(log).write_text(json.dumps(dict(
                    participants_requested=[1, 2, 3], reps=1, perc_available=1)))

        with patch.object(rsa.random, 'shuffle', side_effect=another_instance_finishes), \
                patch.object(rsa.random, 'choice', side_effect=AssertionError('must skip completed draw')):
            outputs = rsa.calculate_mean_leave_one_out_cross_participant_similarity_map_rnd(
                **self.kw, reps=1, reps_group=2)
        self.assertEqual(len(outputs), 2)
        for mean, _ in outputs:
            self.assertEqual(nib.load(mean).get_fdata()[0, 0, 0], 17)

    def test_target_labels_shuffle_with_fixed_reference_and_resume(self):
        self.make_inputs('mahalanobis')
        self.kw['rsa_method'] = 'pearson'
        # A->C, B->B, C->A maps AB, AC, BC to BC, AC, AB.
        with patch.object(rsa.random, 'shuffle', side_effect=lambda labels: labels.reverse()) as shuffle:
            paths = rsa.calculate_leave_one_out_cross_participant_similarity(
                **self.kw, rnd=True, reps=2)
        self.assertEqual(shuffle.call_count, 9)  # Three file orders plus six label permutations.
        self.assertEqual(len(paths), 6)
        for sub, vector in self.vectors.items():
            peers = np.mean([v for s, v in self.vectors.items() if s != sub], axis=0)
            expected = pearsonr(vector[[2, 1, 0]], peers)[0]
            for path in paths:
                if f'H-sub-{sub:02d}' in path:
                    np.testing.assert_allclose(nib.load(path).get_fdata().ravel(), [expected, 0])
                    self.assertIn('RSA_same_rnd/basic/class_all/leave-one-out/run-wise/', Path(path).as_posix())
        times = [Path(p).stat().st_mtime_ns for p in paths]
        with patch.object(rsa.random, 'shuffle', side_effect=AssertionError('must reuse')):
            self.assertEqual(paths, rsa.calculate_leave_one_out_cross_participant_similarity(
                **self.kw, rnd=True, reps=2))
        self.assertEqual(times, [Path(p).stat().st_mtime_ns for p in paths])
        Path(paths[0]).unlink()
        with patch.object(rsa.random, 'shuffle') as shuffle:
            rsa.calculate_leave_one_out_cross_participant_similarity(**self.kw, rnd=True, reps=2)
        self.assertEqual(shuffle.call_count, 2)
        with patch.object(rsa.random, 'shuffle') as shuffle:
            rsa.calculate_leave_one_out_cross_participant_similarity(
                **self.kw, rnd=True, reps=2, replace_rnd_files=True)
        self.assertEqual(shuffle.call_count, 9)

    def test_sparse_class_permutation_uses_unselected_target_pairs(self):
        self.vectors[2] = np.array([8., 2., 0.])
        self.make_inputs('mahalanobis')
        self.kw.update(rsa_class='subset', rsa_method='pearson')
        pd.DataFrame([self.pairs[0], self.pairs[1]], columns=['cat1', 'cat2']).to_csv(
            self.root / 'EmoB/rsa_models/by_class/class_subset.csv', index=False)
        with patch.object(rsa.random, 'shuffle', side_effect=lambda labels: labels.reverse()):
            paths = rsa.calculate_leave_one_out_cross_participant_similarity(**self.kw, rnd=True, reps=1)
        for sub, path in zip(self.vectors, paths):
            peers = np.mean([v[:2] for s, v in self.vectors.items() if s != sub], axis=0)
            expected = pearsonr(self.vectors[sub][[2, 1]], peers)[0]
            self.assertAlmostEqual(nib.load(path).get_fdata()[0, 0, 0], expected)

    def test_group_draw_sampling_equal_weights_and_provenance(self):
        self.make_inputs('correlation')
        rsa.calculate_leave_one_out_cross_participant_similarity(**self.kw, rnd=True, reps=2)
        _, units, paths, _ = rsa._loo_layout(
            str(self.root), 'EmoB', self.units, [1, 2, 3], 'H', 'basic', 'EmoB', 3,
            'all', 'correlation', 'kendall', None, 'run-wise', 'mask', rnd=True)
        expected = []
        for sub in self.vectors:
            by_run = {}
            for session, run in units[sub]:
                value = sub * 10 + session * 3 + run
                by_run.setdefault(run, []).append(value)
                for i in range(2):
                    self.save(Path(paths[sub, session, run][:-7] + f'_{i:04d}.nii.gz'),
                              [value + i * 100, 0])
            expected.append(np.mean([np.mean(v) for v in by_run.values()]))
        with patch.object(rsa.random, 'choice', side_effect=lambda files: files[0]) as choose:
            outputs = rsa.calculate_mean_leave_one_out_cross_participant_similarity_map_rnd(
                **self.kw, reps=2, reps_group=3)
        self.assertEqual(choose.call_count, 21)
        for mean, std in outputs:
            self.assertIn('/mean/', Path(mean).as_posix())
            np.testing.assert_allclose(nib.load(mean).get_fdata().ravel(), [np.mean(expected), 0])
            np.testing.assert_allclose(nib.load(std).get_fdata().ravel(), [np.std(expected), 0])
            log = json.loads(Path(mean[:-7] + '.json').read_text())
            self.assertEqual(len(log['input_files']), 7)
            self.assertTrue(all(p.endswith('_0000.nii.gz') for p in log['input_files']))
        with patch.object(rsa.random, 'choice', side_effect=AssertionError('must reuse')):
            rsa.calculate_mean_leave_one_out_cross_participant_similarity_map_rnd(
                **self.kw, reps=2, reps_group=3)

    def test_group_missing_units_and_grid_mismatch(self):
        self.make_inputs('mahalanobis')
        paths = rsa.calculate_leave_one_out_cross_participant_similarity(**self.kw, rnd=True, reps=1)
        Path(paths[0]).unlink()
        with self.assertWarnsRegex(UserWarning, 'incomplete'), self.assertRaisesRegex(ValueError, 'insufficient'):
            rsa.calculate_mean_leave_one_out_cross_participant_similarity_map_rnd(
                **self.kw, reps=1, reps_group=1, min_percentage_available=1)
        with self.assertWarnsRegex(UserWarning, 'incomplete'):
            rsa.calculate_mean_leave_one_out_cross_participant_similarity_map_rnd(
                **self.kw, reps=1, reps_group=1)
        affine = np.eye(4)
        affine[0, 3] = 10
        self.save(Path(paths[1]), [1, 0], affine)
        with self.assertWarns(UserWarning), self.assertRaises(rsa.SpaceMismatchError):
            rsa.calculate_mean_leave_one_out_cross_participant_similarity_map_rnd(
                **self.kw, reps=1, reps_group=1, replace_rnd_files=True)

    def test_group_complete_draws_skip_inputs_and_partial_draws_resume(self):
        self.make_inputs('mahalanobis')
        rsa.calculate_leave_one_out_cross_participant_similarity(**self.kw, rnd=True, reps=2)
        kwargs = dict(self.kw, reps=2, reps_group=3)
        outputs = rsa.calculate_mean_leave_one_out_cross_participant_similarity_map_rnd(**kwargs)
        expected_root = (self.root / 'EmoB/results/RSA_same_rnd/basic/class_all'
                         / 'leave-one-out/run-wise/mean')
        self.assertEqual(Path(outputs[0][0]), expected_root / 'mask-H-r-3_mahalanobis_kendall_task-EmoB_mean_00000.nii.gz')
        before = {path: Path(path).stat().st_mtime_ns for pair in outputs for path in pair}
        with patch.object(rsa.nib, 'load', side_effect=AssertionError('must skip completed draws')), \
                patch.object(rsa.random, 'shuffle', side_effect=AssertionError('must not randomize completed draws')):
            self.assertEqual(outputs, rsa.calculate_mean_leave_one_out_cross_participant_similarity_map_rnd(**kwargs))
        Path(outputs[1][1]).unlink()
        with patch.object(rsa.nib, 'save', wraps=nib.save) as save:
            self.assertEqual(outputs, rsa.calculate_mean_leave_one_out_cross_participant_similarity_map_rnd(**kwargs))
        self.assertEqual({call.args[1] for call in save.call_args_list}, set(outputs[1]))
        for pair in (outputs[0], outputs[2]):
            for path in pair:
                self.assertEqual(Path(path).stat().st_mtime_ns, before[path])
        with patch.object(rsa.nib, 'save', wraps=nib.save) as save:
            rsa.calculate_mean_leave_one_out_cross_participant_similarity_map_rnd(**kwargs, replace_rnd_files=True)
        self.assertEqual(save.call_count, 6)

    def test_group_requires_permutations_and_valid_counts(self):
        with patch('builtins.print') as output, self.assertWarnsRegex(UserWarning, 'incomplete'), \
                self.assertRaisesRegex(ValueError, 'Run step 11.4'):
            rsa.calculate_mean_leave_one_out_cross_participant_similarity_map_rnd(
                **self.kw, reps=1, reps_group=1, verbose=True)
        printed = '\n'.join(str(call.args[0]) for call in output.call_args_list)
        self.assertIn('Expected first file:', printed)
        self.assertIn('mask-r-3_mahalanobis_kendall_0000.nii.gz', printed)
        self.assertIn('0/1 permutation files found', printed)
        for options in (dict(reps=0), dict(reps_group=0), dict(min_percentage_available=2)):
            with self.assertRaises(ValueError):
                rsa.calculate_mean_leave_one_out_cross_participant_similarity_map_rnd(**self.kw, **options)
