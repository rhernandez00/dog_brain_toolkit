"""Tests for unreadable permutation z maps in step 8."""

import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

import rsa_utils


class ClusterSizeDistributionTests(unittest.TestCase):
    def test_unreadable_map_is_skipped_and_logged(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root) / 'EmoB'
            mean_dir = base / 'results/RSA_rnd/basic-block/LcSSG_mah/mean'
            mean_dir.mkdir(parents=True)
            config_dir = base / 'config_files'
            config_dir.mkdir()
            (config_dir / 'H_basic-block.yaml').write_text('{}')
            stem = 'H-r-4_mahalanobis_kendall'
            good_path = mean_dir / f'{stem}_z_00000.nii.gz'
            bad_path = mean_dir / f'{stem}_z_00001.nii.gz'
            nib.save(nib.Nifti1Image(np.full((3, 3, 3), 2, dtype=np.float32),
                                     np.eye(4)), str(good_path))
            rng = np.random.default_rng(42)
            nib.save(nib.Nifti1Image(rng.standard_normal((32, 32, 32)).astype(np.float32),
                                     np.eye(4)), str(bad_path))
            bad_bytes = bad_path.read_bytes()[:bad_path.stat().st_size // 2]
            bad_path.write_bytes(bad_bytes)

            self.assertTrue(rsa_utils.calculate_cluster_size_distribution(
                root, 'EmoB', 'basic-block', 'LcSSG_mah', 4, 'H',
                'mahalanobis', 'kendall', z_threshold=1))

            dist = base / f'results/RSA/basic-block/LcSSG_mah/dist/{stem}_dist.npy'
            entry = np.load(dist, allow_pickle=True).item()['z1']
            self.assertEqual(entry['number_of_images'], 1)
            self.assertEqual(list(entry['cluster_sizes'][0]), [27])
            log = dist.with_name(dist.stem + '_log.txt').read_text()
            self.assertIn(f'Processed files: {[str(good_path)]}', log)
            self.assertIn(f'Skipped file: {bad_path}', log)
            self.assertEqual(bad_path.read_bytes(), bad_bytes)


if __name__ == '__main__':
    unittest.main()
