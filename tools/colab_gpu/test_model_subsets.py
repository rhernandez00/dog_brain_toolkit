"""CPU regression checks for Colab correlation models with subset stimuli."""
import tempfile
import unittest
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from tools.colab_gpu import gpu_rsa as gpu


class ModelSubsetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.models = self.root / 'data/EmoB/rsa_models'
        self.models.mkdir(parents=True)
        self.categories = ['A-1', 'SA-1', 'H-1', 'C-1']
        self.selected = ['A-1', 'H-1', 'C-1']
        self.matrix = np.array([[0., 1., 3.], [1., 0., 2.], [3., 2., 0.]])

    def write_model(self, name, labels, matrix):
        path = self.models / (name + '.csv')
        pd.DataFrame(matrix, index=labels, columns=labels).to_csv(path)
        return str(path)

    def test_subset_pairs_and_permutations_exclude_other_stimuli(self):
        # CSV order differs from the package order, as in real EmoB models.
        order = [2, 0, 1]
        path = self.write_model('anger-strict', [self.selected[i] for i in order],
                                self.matrix[np.ix_(order, order)])
        vectors, columns = gpu.correlation_model_vectors(path, self.categories, 7, 42)
        np.testing.assert_array_equal(columns, [1, 2, 5])
        expected = gpu.build_model_vectors(self.matrix, 7, 42)
        np.testing.assert_array_equal(vectors, expected)
        self.assertTrue(np.isfinite(vectors).all())

    def test_full_model_preserves_existing_vectors(self):
        matrix = np.arange(16, dtype=float).reshape(4, 4)
        matrix = matrix + matrix.T
        np.fill_diagonal(matrix, 0)
        path = self.write_model('full__all', self.categories, matrix)
        vectors, columns = gpu.correlation_model_vectors(path, self.categories, 5, 10)
        np.testing.assert_array_equal(columns, np.arange(6))
        np.testing.assert_array_equal(vectors, gpu.build_model_vectors(matrix, 5, 10))

    def test_unknown_stimulus_is_rejected(self):
        path = self.write_model('wrong', ['typo', 'H-1', 'C-1'], self.matrix)
        with self.assertRaisesRegex(ValueError, 'absent from the package'):
            gpu.correlation_model_vectors(path, self.categories, 1, 0)

    def test_per_run_comparison_and_zip_names(self):
        self.write_model('anger-strict', self.selected, self.matrix)
        self.write_model('visual1-run-1', self.selected, self.matrix)
        self.write_model('visual1-run-2', self.selected, -self.matrix)
        manifest = dict(dataset='EmoB', model='basic-block', specie='H', sub_N=1,
                        radius=4, rsa_method='kendall', reps=2, mask_type='mask',
                        task='EmoB', categories=self.categories, dis_method='correlation',
                        run_dependent_models=['visual1'])
        # Only columns 1, 2, 5 belong to the model; excluded data are extreme.
        data = torch.tensor([[999., 1., 3., -999., 888., 2.]], dtype=torch.float64)
        meta = dict(shape=(1, 1, 1), affine=np.eye(4), runs=[
            dict(session=1, run_N=n, data=data, mask_flat=np.array([0])) for n in (1, 2)])
        for name, expected in [('anger-strict', [1., 1.]), ('visual1', [1., -1.])]:
            gpu.run_model(str(self.root), manifest, name, meta=meta,
                          device=torch.device('cpu'), seed=42, verbose=False)
            for run_n, value in enumerate(expected, 1):
                path = self.root / ('data/EmoB/results/RSA/basic-block/' + name) / (
                    f'H-sub-01/ses-01_task-EmoB_run-{run_n:02d}/mask-r-4_correlation_kendall.nii.gz')
                self.assertAlmostEqual(nib.load(path).get_fdata().item(), value)
            output = gpu.zip_model_result(str(self.root), manifest, name, str(self.root / 'out'))
            self.assertEqual(Path(output).name, f'result_{name}_H-sub-01.zip')
            with zipfile.ZipFile(output) as zf:
                self.assertEqual(len(zf.namelist()), 6)
                self.assertTrue(all(p.replace('\\', '/').startswith('EmoB/results/')
                                    for p in zf.namelist()))


if __name__ == '__main__':
    unittest.main()
