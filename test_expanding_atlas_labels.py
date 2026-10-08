"""Region lookup shared by the step 10 and 15.10 cluster reports."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from rsa_utils import extract_clusters_and_peaks


class ExpandingAtlasLabelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.stat_path = Path(self.temp.name) / 'corrected.nii.gz'
        stat = np.zeros((9, 9, 9))
        stat[4, 4, 4] = 5
        nib.save(nib.Nifti1Image(stat, np.eye(4)), str(self.stat_path))
        self.atlas = np.zeros((5, 5, 5))
        self.affine = np.diag([2, 2, 2, 1])
        self.names = pd.DataFrame({'Number': [1, 2], 'Region': ['A', 'B']})

    def region(self):
        with contextlib.redirect_stdout(io.StringIO()):
            clusters = extract_clusters_and_peaks(
                str(self.stat_path), label_dict=self.names,
                label_nii_data=self.atlas, label_affine=self.affine)
        return clusters[0]['peaks'][0]['region']

    def test_expands_to_first_sphere_and_uses_voxel_majority(self):
        # Peak is at 4 mm in each axis, or atlas voxel (2, 2, 2).
        # The first named voxels are all 4 mm away; B occupies two of them.
        self.atlas[4, 2, 2] = 1
        self.atlas[2, 4, 2] = 2
        self.atlas[2, 2, 4] = 2
        self.assertEqual(self.region(), 'B')

    def test_keeps_direct_label(self):
        self.atlas[2, 2, 2] = 1
        self.atlas[4, 2, 2] = 2
        self.assertEqual(self.region(), 'A')

    def test_no_named_voxels_remains_unknown(self):
        self.atlas[3, 2, 2] = 99
        self.assertEqual(self.region(), 'Unknown')

    def test_expands_past_atlas_edge(self):
        self.affine[0, 3] = 20
        self.atlas[0, 2, 2] = 2
        self.assertEqual(self.region(), 'B')


if __name__ == '__main__':
    unittest.main()
