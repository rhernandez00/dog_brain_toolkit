"""Check the Colab continuation against the actual searchlight inference steps."""
import contextlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pandas as pd

import rsa_utils as rsa
from tools.colab_gpu import run_colab_regression_inference as inference
from tools.colab_gpu.run_colab_regression import extract_safe
from tools.unpack_results import _safe_member


class ColabInferenceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)
        self.support = self.root / 'support'
        self.mask = self.support / 'data/tiny/ROI/H/mask.nii.gz'
        self.save(self.mask, np.ones((3, 3, 3)))
        self.save(self.support / 'atlas.nii.gz', np.ones((3, 3, 3)))
        pd.DataFrame({'Number': [1], 'Region': ['Test region']}).to_csv(self.support / 'labels.csv', index=False)
        (self.support / 'inference_manifest.json').write_text(json.dumps(dict(
            dataset='tiny', species={'H': dict(labels='atlas.nii.gz', dictionary='labels.csv',
            template=None, apply_coords_transform=False)}, file_sha256={})))
        self.stem = 'mask-H-r-1_correlation_beta'
        self.real = Path('tiny/results/RSA_regression/basic/visual_3/target/mean')
        self.rnd = Path('tiny/results/RSA_regression_rnd/basic/visual_3/target/mean')
        self.inputs = self.root / 'inputs'
        rng = np.random.default_rng(456)
        for i in range(10):
            data = rng.normal(size=(3, 3, 3))
            data[2, 2, 2] = 7  # zero null variance must yield z=0
            self.save(self.inputs / self.rnd / f'{self.stem}_mean_{i:05d}.nii.gz', data)
        real = np.zeros((3, 3, 3))
        real[:2, :2, :2] = 20
        real[2, 2, 2] = 20
        self.save(self.inputs / self.real / f'{self.stem}_mean.nii.gz', real)
        self.results = self.root / 'results'
        self.results.mkdir()
        self.archive = self.results / 'result_regression_group_target_H.zip'
        with zipfile.ZipFile(self.archive, 'w') as zf:
            for path in self.inputs.rglob('*.nii.gz'):
                zf.write(path, path.relative_to(self.inputs).as_posix())
        self.kw = dict(dataset='tiny', model='basic', regression_model='visual_3', specie='H',
                       radius=1, mask_type='mask', dis_method='correlation', reps_group=10,
                       z_threshold=2., cluster_threshold=.05, min_dist_mm=8.)

    @staticmethod
    def save(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(data.astype(float), np.eye(4)), str(path))

    def test_outputs_match_cpu_pipeline_and_resume(self):
        out = self.root / 'out'
        files = inference.run_inference(self.results, out, self.support, **self.kw,
                                        device='cpu', work_root=self.root / 'work')
        self.assertEqual(len(files), 1)
        actual = self.root / 'actual'
        extract_safe(files[0], actual)
        expected = self.root / 'expected'
        extract_safe(self.archive, expected)
        atlas = nib.load(str(self.support / 'atlas.nii.gz'))
        for step in inference.STEPS:
            self.assertTrue(rsa.calculate_regression_inference(step, datafolder=str(expected),
                rsa_models_list=['target'], mask=str(self.mask), **self.kw,
                label_dict=pd.read_csv(self.support / 'labels.csv'),
                label_nii_data=atlas.get_fdata(), label_affine=atlas.affine))
        nifti_outputs = list(actual.rglob('*.nii.gz'))
        self.assertEqual(len(nifti_outputs), 14)  # two null moments, 11 z maps, corrected
        for path in nifti_outputs:
            reference = expected / path.relative_to(actual)
            np.testing.assert_allclose(nib.load(path).get_fdata(), nib.load(reference).get_fdata(),
                                       atol=1e-6, rtol=1e-6)
        csv = next(actual.rglob('*.csv'))
        pd.testing.assert_frame_equal(pd.read_csv(csv), pd.read_csv(expected / csv.relative_to(actual)))
        self.assertTrue((pd.read_csv(csv).region == 'Test region').all())
        dist = next(actual.rglob('*.npy'))
        self.assertEqual(np.load(dist, allow_pickle=True).item()['z2.0']['cluster_sizes'],
                         np.load(expected / dist.relative_to(actual), allow_pickle=True).item()['z2.0']['cluster_sizes'])
        self.assertIsNotNone(_safe_member(csv.relative_to(actual).as_posix()))
        with patch.object(rsa, 'calculate_regression_inference', side_effect=AssertionError('cached')):
            self.assertEqual(inference.run_inference(self.results, out, self.support, **self.kw,
                device='cpu', work_root=self.root / 'work'), [])

    def test_missing_group_and_wrong_analysis_are_explicit(self):
        empty = self.root / 'empty'
        empty.mkdir()
        args = dict(dataset='tiny', model='basic', regression_model='visual_3', specie='H',
                    mask_type='mask', radius=1, dis_method='correlation', reps_group=10)
        with self.assertRaisesRegex(FileNotFoundError, 'finish steps 15.3/15.5'):
            inference.discover_groups(empty, **args)
        with self.assertRaisesRegex(ValueError, 'required group mean maps missing'):
            inference.discover_groups(self.results, **dict(args, radius=3))
        with self.assertRaisesRegex(ValueError, 'required group mean maps missing'):
            inference.discover_groups(self.results, **dict(args, reps_group=11))

    def test_empty_report_and_omit_permutation_z_export(self):
        self.save(self.inputs / self.real / f'{self.stem}_mean.nii.gz', np.zeros((3, 3, 3)))
        with zipfile.ZipFile(self.archive, 'w') as zf:
            for path in self.inputs.rglob('*.nii.gz'):
                zf.write(path, path.relative_to(self.inputs).as_posix())
        files = inference.run_inference(self.results, self.root / 'out', self.support, **self.kw,
            device='cpu', work_root=self.root / 'work', write_permutation_z=False)
        with zipfile.ZipFile(files[0]) as zf:
            self.assertFalse(any('_beta_z_' in name for name in zf.namelist()))
            csv = next(name for name in zf.namelist() if name.endswith('.csv'))
            with zf.open(csv) as handle:
                self.assertTrue(pd.read_csv(handle).empty)


if __name__ == '__main__':
    unittest.main()
