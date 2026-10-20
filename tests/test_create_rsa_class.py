"""Check class labels and complete pair sets against step-1 conventions."""
import itertools
from pathlib import Path
import tempfile
import unittest

import pandas as pd
import yaml

from rsa_utils import create_rsa_class


class CreateClassTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.config = self.root / 'EmoC/config_files/H_basic-block.yaml'
        self.config.parent.mkdir(parents=True)
        self.stimuli = [f'{category}{i}' for category in
                       ['DogP', 'DogH', 'DogA', 'DogF', 'DogN',
                        'HumP', 'HumH', 'HumA', 'HumF', 'HumN'] for i in range(1, 5)]
        self.write_config(dict(dataset='EmoC', stim_types=self.stimuli))

    def write_config(self, config):
        self.config.write_text(yaml.safe_dump(config))

    def read_pairs(self, path):
        table = pd.read_csv(path)
        self.assertEqual(list(table), ['cat1', 'cat2'])
        pairs = list(table.itertuples(index=False, name=None))
        self.assertTrue(all(a != b for a, b in pairs))
        self.assertEqual(len(pairs), len({frozenset(pair) for pair in pairs}))
        return pairs

    def test_all_mahalanobis_has_all_45_category_pairs(self):
        path = create_rsa_class(self.config, 'all_mahalanobis')
        self.assertEqual(Path(path), self.root / 'EmoC/rsa_models/by_class/class_all_mahalanobis.csv')
        pairs = self.read_pairs(path)
        labels = list(dict.fromkeys(s[:-1] for s in self.stimuli))
        self.assertEqual(pairs, list(itertools.combinations(labels, 2)))
        self.assertEqual(len(pairs), 45)

    def test_correlation_and_run_wise_have_780_raw_stimulus_pairs(self):
        for name, method, fold in [('all_correlation', 'correlation', 'stim-wise'),
                                    ('all_runwise', 'mahalanobis', 'run-wise')]:
            pairs = self.read_pairs(create_rsa_class(self.config, name, method, mah_fold=fold))
            self.assertEqual(len(pairs), 780)
            self.assertEqual(pairs, list(itertools.combinations(self.stimuli, 2)))

    def test_emob_category_collapse_and_subset(self):
        self.write_config(dict(dataset='EmoB', stim_types=['face-1', 'face-2', 'body-1', 'voice-1']))
        pairs = self.read_pairs(create_rsa_class(self.config, 'class_face_body', stimuli=['face', 'body']))
        self.assertEqual(pairs, [('face', 'body')])
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            create_rsa_class(self.config, 'wrong', stimuli=['face-1', 'body-1'])

    def test_no_overwrite_and_optional_output_directory(self):
        path = create_rsa_class(self.config, 'all', output_dir=self.root / 'custom')
        initial = Path(path).read_bytes()
        with self.assertRaises(FileExistsError):
            create_rsa_class(self.config, 'class_all', output_dir=self.root / 'custom')
        self.assertEqual(Path(path).read_bytes(), initial)
        create_rsa_class(self.config, 'all', output_dir=self.root / 'custom',
                         stimuli=['DogP', 'HumP'], replace_file=True)
        self.assertEqual(self.read_pairs(path), [('DogP', 'HumP')])

    def test_within_run_union_does_not_invent_cross_run_comparisons(self):
        self.write_config(dict(dataset='EmoC', runs=[1, 2],
            stim_types=['A1', 'A2', 'B1', 'B2', 'C1', 'C2'],
            model_dict={
                'run01': {s: {} for s in ['A1', 'A2', 'B1', 'B2']},
                'run02': {s: {} for s in ['A1', 'A2', 'C1', 'C2']}}))
        pairs = self.read_pairs(create_rsa_class(self.config, 'within_run',
                                                 mah_fold='stim-wise-all-runs'))
        self.assertEqual(pairs, [('A', 'B'), ('A', 'C')])

    def test_invalid_method_fold_and_stimulus_list(self):
        for kwargs in [dict(dis_method='unknown'), dict(mah_fold='unknown'),
                       dict(stimuli='DogP'), dict(stimuli=['DogP', 'DogP'])]:
            with self.assertRaises(ValueError):
                create_rsa_class(self.config, 'invalid', **kwargs)
        self.write_config(dict(dataset='EmoC', stim_types=['DogP1', 'DogP1']))
        with self.assertRaisesRegex(ValueError, 'unique'):
            create_rsa_class(self.config, 'invalid')


if __name__ == '__main__':
    unittest.main()
