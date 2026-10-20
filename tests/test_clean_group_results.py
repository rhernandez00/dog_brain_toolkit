import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from tools.clean_group_results import main, plan_cleanup


class CleanGroupResultsTests(unittest.TestCase):
    def test_preview_apply_and_preservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            dataset = data / 'Example'
            base = 'results/{tree}/basic/emotion/{leaf}/'
            removed = [
                base.format(tree='RSA', leaf='mean') + 'mask-H-r-4_mahalanobis_kendall_mean.json',
                base.format(tree='RSA', leaf='mean') + 'H-r-3_mahalanobis_kendall_mean.nii.gz',
                base.format(tree='RSA', leaf='mean') + 'mask-H-r-4_mahalanobis_kendall_zt3.1_corrected.json',
                base.format(tree='RSA', leaf='mean') + 'mask-H-r-4_mahalanobis_kendall_zt2.3_publication.docx',
                base.format(tree='RSA', leaf='dist') + 'H-r-4_mahalanobis_kendall_dist.npy',
                base.format(tree='RSA', leaf='dist') + 'H-r-4_mahalanobis_kendall_dist_log.txt',
                base.format(tree='RSA_rnd', leaf='mean') + 'H-r-4_mahalanobis_kendall_mean_01000.nii.gz',
                base.format(tree='RSA_rnd', leaf='mean') + 'H-r-4_mahalanobis_kendall_z_log.txt',
                'results/RSA_rnd/basic/H-emotion_mean.nii.gz',
                'results/RSA_rnd/basic/H-emotion_std.nii.gz',
                'results/RSA_rnd/basic/H-emotion_mean_log.txt',
            ]
            kept = [
                base.format(tree='RSA', leaf='H-sub-01') + 'r-4_mahalanobis_kendall.nii.gz',
                base.format(tree='RSA_rnd', leaf='H-sub-01') + 'r-4_mahalanobis_kendall_0001.nii.gz',
                base.format(tree='RSA', leaf='mean') + 'mask-D-r-4_mahalanobis_kendall_mean.nii.gz',
                base.format(tree='RSA', leaf='mean') + 'mask-H-r-4_correlation_kendall_mean.nii.gz',
                'results/RSA_rnd/basic/H-emotion-other_mean.nii.gz',
                'results/RSA/basic/other/mean/H-r-4_mahalanobis_kendall_mean.nii.gz',
                'results/RSA_regression/basic/emotion/mean/H-r-4_mahalanobis_kendall_mean.nii.gz',
            ]
            for name in removed + kept:
                path = dataset / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('test')
            argv = ['--datafolder', str(data), '--dataset', 'Example', '--model', 'basic',
                    '--rsa_model', 'emotion', '--specie', 'H']
            with contextlib.redirect_stdout(io.StringIO()):
                main(argv)
                self.assertTrue(all((dataset / name).exists() for name in removed + kept))
                main(argv + ['--apply'])
            self.assertTrue(all(not (dataset / name).exists() for name in removed))
            self.assertTrue(all((dataset / name).exists() for name in kept))
            self.assertEqual(plan_cleanup(dataset, 'basic', ['emotion'], ['H'],
                                          'mahalanobis', 'kendall'), [])

    def test_mirrors_and_markers_are_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / 'data' / 'Example'
            dataset.mkdir(parents=True)
            mirror = root / 'mirror'
            copied = mirror / 'Example/current-results/RSA/H/mask/H_emotion_zt3.1.csv'
            copied.parent.mkdir(parents=True)
            copied.write_text('test')
            marker = root / 'markers'
            marker.mkdir()
            for step in (2, 3, 4, 4.5, 5, 7.6, 10, 11):
                (marker / f'{step}.done').write_text('test')
            args = (dataset, 'basic', ['emotion'], ['H'], 'mahalanobis', 'kendall')
            self.assertEqual(plan_cleanup(*args), [])
            planned = plan_cleanup(*args, mirror_root=mirror, marker_dir=marker)
            self.assertEqual(set(planned), {copied, marker / '3.done', marker / '5.done', marker / '10.done'})


if __name__ == '__main__':
    unittest.main()
