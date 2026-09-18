"""CPU-reference checks for the CUDA-capable regression kernels and ZIP workflow."""
import contextlib
import io
import itertools
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pandas as pd
import torch

import rsa_utils
from tools.colab_gpu import gpu_regression as gpu
from tools.colab_gpu import run_colab_regression as runner
from tools.create_regression_package import build_support
from tools import create_regression_package as packager


class KernelTests(unittest.TestCase):
    def test_finite_data_does_not_sort_voxel_rows(self):
        rng = np.random.default_rng(7)
        data = rng.normal(size=(20, 10))
        designs = np.array([np.column_stack([np.ones(10), rng.normal(size=(10, 2))])
                            for _ in range(4)])
        messages = []
        with patch.object(gpu.np, 'unique', side_effect=AssertionError('Unnecessary row sort')):
            fits = list(gpu.fit_designs(data, designs, 'cpu', 3, 2, progress=messages.append))
        self.assertEqual(len(fits), 4)
        self.assertTrue(any('4/4' in message for message in messages))

    def check_solver(self, device):
        rng = np.random.default_rng(11)
        data = rng.normal(size=(9, 10))
        data[0, :] = 3  # constant response
        data[1, 0] = np.nan
        data[2, 2:] = np.nan  # insufficient usable pairs
        designs = np.stack([np.column_stack([np.ones(10), rng.normal(size=(10, 2))]) for _ in range(5)])
        designs[1, 3, 1] = np.nan
        designs[2, 3, 1] = np.nan
        designs[3, :, 2] = designs[3, :, 1]  # rank deficient, CPU pseudoinverse semantics
        for i, beta, t, p, dof in gpu.fit_designs(data, designs, device, 3, 2):
            expected = rsa_utils.perform_multiple_regression_rsa(
                data.reshape(9, 1, 1, 10), designs[i], np.ones((9, 1, 1), bool))
            for actual, reference in zip([beta, t, p], expected[:3]):
                np.testing.assert_allclose(actual, reference.ravel(), atol=1e-10, rtol=1e-9)
            self.assertEqual(dof, expected[3])

    def test_solver_matches_cpu_including_missing_data(self):
        self.check_solver('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable on workstation')
    def test_solver_on_cuda(self):
        self.check_solver('cuda')

    def test_group_chunks_match_numpy(self):
        bank = np.arange(77, dtype=np.float32).reshape(7, 11)
        selections = np.array([[0, 2, 6], [1, 3, 5], [0, 4, 5]])
        for g, v, mean, std in gpu.group_moments(bank, selections, 'cpu', 4, 2):
            expected = bank[selections[g:g + len(mean)], v:v + mean.shape[1]].astype(float)
            np.testing.assert_allclose(mean, expected.mean(axis=1))
            np.testing.assert_allclose(std, expected.std(axis=1))


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.packages = self.root / 'packages'
        self.packages.mkdir()
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)
        self.responses = {}
        for sub in [1, 2]:
            m = dict(dataset='tiny', model='basic', specie='H', sub_N=sub, radius=1,
                     task='tiny', dis_method='correlation', mah_fold='stim-wise',
                     mask_type='mask', models=['target', 'visual'], categories=list('ABCD'),
                     stim_types=list('ABCD'), runs=[dict(session=1, run_N=r) for r in range(1, sub + 1)])
            with zipfile.ZipFile(self.packages / f'pkg_{sub}.zip', 'w') as zf:
                zf.writestr('manifest.json', json.dumps(m))
                zf.writestr('data/tiny/config_files/H_basic.yaml', 'participants: [1, 2]\n')
                zf.writestr('data/tiny/ROI/H/mask.nii.gz', self.nifti_bytes([1, 1, 0]))
                zf.writestr('data/tiny/rsa_models/target.csv', self.csv([0, 1, 0, 1, 0, 1]))
                for entry in m['runs']:
                    run = entry['run_N']
                    zf.writestr(f'data/tiny/rsa_models/visual-run-{run}.csv', self.csv([1, 2, 4, 3, 5, 7]))
                    values = np.random.default_rng(sub * 10 + run).normal(size=(2, 6))
                    self.responses[sub, run] = values
                    for i, (a, b) in enumerate(itertools.combinations('ABCD', 2)):
                        zf.writestr(f'data/tiny/results/RSA/basic/H-sub-{sub:02d}/'
                                    f'ses-01_task-tiny_run-{run:02d}/r-1_correlation_{a}_{b}.nii.gz',
                                    self.nifti_bytes([*values[:, i], 999]))
        self.support = build_support(self.packages, self.root / 'support', 'visual_3', ['visual'],
                                     dataset='tiny', species=['H'], model='basic')
        self.kw = dict(specie='H', dataset='tiny', model='basic', reps=2, reps_group=3,
                       device='cpu', voxel_batch=1, permutation_batch=2, group_batch=2,
                       work_root=self.root / 'work')

    @staticmethod
    def csv(values):
        matrix = np.zeros((4, 4))
        matrix[np.triu_indices(4, 1)] = values
        matrix += matrix.T
        return pd.DataFrame(matrix, index=list('ABCD'), columns=list('ABCD')).to_csv()

    def nifti_bytes(self, values):
        path = self.root / 'temp.nii.gz'
        nib.save(nib.Nifti1Image(np.array(values, dtype=float).reshape(3, 1, 1), np.eye(4)), path)
        return path.read_bytes()

    def test_all_steps_pipeline_paths_numerical_parity_and_resume(self):
        out = self.root / 'out'
        written = runner.run_regression(self.packages, out, self.support, **self.kw)
        self.assertEqual(len(written), 3)
        merged = self.root / 'merged'
        for archive in out.glob('*.zip'):
            runner.extract_safe(archive, merged)
        # Every saved participant fit agrees with the CPU for its exact shuffled target.
        for sub in (1, 2):
            for run in range(1, sub + 1):
                for family, indices in [('RSA_regression', [None]), ('RSA_regression_rnd', [0, 1])]:
                    folder = merged / f'tiny/results/{family}/basic/visual_3/target/H-sub-{sub:02d}/ses-01_task-tiny_run-{run:02d}'
                    for index in indices:
                        suffix = '' if index is None else f'_{index:04d}'
                        record = json.loads((folder / f'r-1_correlation_regression{suffix}.json').read_text())
                        design = np.column_stack([np.ones(6), record['target_vector'], [1, 2, 4, 3, 5, 7]])
                        expected = rsa_utils.perform_multiple_regression_rsa(
                            self.responses[sub, run].reshape(2, 1, 1, 6), design, np.ones((2, 1, 1), bool))
                        for name, reference in zip(['beta', 't', 'p'], expected[:3]):
                            actual = nib.load(folder / f'r-1_correlation_{name}_map{suffix}.nii.gz').get_fdata().ravel()
                            np.testing.assert_allclose(actual[:2], reference.ravel(), atol=2e-6, rtol=2e-6)
                            self.assertEqual(actual[2], 1 if name == 'p' else 0)
        for record_path in merged.glob('tiny/results/*/basic/visual_3/target/mean/*_mean*.json'):
            record = json.loads(record_path.read_text())
            source = np.stack([nib.load(merged / p).get_fdata() for p in record['file_list']])
            np.testing.assert_allclose(nib.load(str(record_path).replace('.json', '.nii.gz')).get_fdata(), source.mean(axis=0))
            np.testing.assert_allclose(nib.load(str(record_path).replace('_mean', '_std').replace('.json', '.nii.gz')).get_fdata(), source.std(axis=0), atol=1e-12)
        with patch.object(gpu, 'fit_designs', side_effect=AssertionError('Should resume')):
            self.assertEqual(runner.run_regression(self.packages, out, self.support, **self.kw), [])

    def test_separate_steps_preserve_earlier_outputs_and_group_only(self):
        out = self.root / 'out'
        runner.run_regression(self.packages, out, self.support, steps=[15], **self.kw)
        runner.run_regression(self.packages, out, self.support, steps=[15.4], **self.kw)
        written = runner.run_regression(self.packages, out, self.support, steps=[15.3, 15.5], **self.kw)
        self.assertEqual(len(written), 1)
        with self.assertRaisesRegex(FileNotFoundError, 'Missing or stale'):
            runner.run_regression(self.packages, out, self.support, steps=[15.5], **dict(self.kw, reps=3))

    def test_interrupted_participant_restores_completed_run_checkpoints(self):
        out = self.root / 'out'
        original = gpu.load_run_data
        calls = []
        def interrupt_second_run(root, manifest, entry, *args, **kwargs):
            calls.append((manifest['sub_N'], entry['run_N']))
            if manifest['sub_N'] == 2 and entry['run_N'] == 2:
                raise RuntimeError('Simulated runtime interruption')
            return original(root, manifest, entry, *args, **kwargs)
        with patch.object(gpu, 'load_run_data', side_effect=interrupt_second_run):
            with self.assertRaisesRegex(RuntimeError, 'Simulated runtime interruption'):
                runner.run_regression(self.packages, out, self.support, **self.kw)
        self.assertEqual(len(list((out / 'run_checkpoints').glob('*.zip'))), 2)
        with patch.object(gpu, 'load_run_data', wraps=original) as load:
            runner.run_regression(self.packages, out, self.support, **self.kw)
            self.assertEqual(load.call_count, 1)
            self.assertEqual(load.call_args.args[1]['sub_N'], 2)
            self.assertEqual(load.call_args.args[2]['run_N'], 2)

    def test_missing_participant_and_unsafe_archive_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Missing participant'):
            runner.discover_packages(self.packages, 'H', 'tiny', participants=[1, 2, 3], model='basic')
        archive = self.root / 'bad.zip'
        with zipfile.ZipFile(archive, 'w') as zf:
            zf.writestr('../escaped.txt', 'bad')
        with self.assertRaisesRegex(ValueError, 'Unsafe ZIP'):
            runner.extract_safe(archive, self.root / 'extract')

    def test_new_dataset_package_includes_betas_controls_code_and_notebook(self):
        source = self.root / 'source'
        runner.extract_safe(self.packages / 'pkg_1.zip', source, prefix='data/')
        dataset = source / 'data/tiny'
        (dataset / 'config_files/H_basic.yaml').write_text(
            'participants: [1]\ntask: tiny\nstim_types: [A, B, C, D]\n')
        (dataset / 'BIDS').mkdir()
        (dataset / 'BIDS/H_database-details.csv').write_text('sub_N,session,run_N\n1,1,1\n')
        beta_folder = dataset / 'results/GLM/basic/H-sub-01/ses-01_task-tiny_run-01'
        beta_folder.mkdir(parents=True)
        for i, stim in enumerate('ABCD'):
            (beta_folder / f'beta_{stim}.nii.gz').write_bytes(self.nifti_bytes([i + 1, 4 - i, 0]))
        output = self.root / 'fresh'
        argv = ['create_regression_package.py', '--dataset', 'tiny', '--specie', 'H',
                '--model', 'basic', '--models', 'target', '--controls', 'visual',
                '--radius', '1', '--mask_type', 'mask', '--out', str(output)]
        locations = (str(source / 'data'), str(self.root), 'python')
        with patch('sys.argv', argv), patch('scheduler.paths.get_paths', return_value=locations), \
                patch('create_package.get_paths', return_value=locations):
            packager.main()
        archive = next((output / 'packages').glob('pkg_*.zip'))
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
            self.assertEqual(sum('/beta_' in p for p in names), 4)
            self.assertIn('data/tiny/rsa_models/visual-run-1.csv', names)
        with zipfile.ZipFile(output / 'regression_support_tiny_visual_3.zip') as zf:
            self.assertIn('code/gpu_regression.py', zf.namelist())
            self.assertIn('colab_rsa_regression.ipynb', zf.namelist())
            self.assertIn('data/tiny/rsa_models/regression_models/visual_3.csv', zf.namelist())


if __name__ == '__main__':
    unittest.main()
