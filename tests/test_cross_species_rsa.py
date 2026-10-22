"""Numerical, I/O, portable-package and restart tests for step 16.2."""
import copy
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import h5py
import nibabel as nib
import numpy as np
from scipy.stats import kendalltau
import yaml

import cross_species_rsa as rsa
from tools.create_cross_species_package import create_package
from tools.colab_gpu.run_colab_cross_species import extract_package, run_colab


class KendallTests(unittest.TestCase):
    def test_pipeline_dispatches_step_in_main_loop(self):
        path = Path(rsa.__file__).parent / 'searchlight.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        step_loop = next(node for node in main.body if isinstance(node, ast.For)
                         and isinstance(node.target, ast.Name) and node.target.id == 'step')
        calls = [node for node in ast.walk(step_loop) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)
                 and node.func.attr == 'calculate_voxelwise_cross_similarity']
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].func.value.id, 'rsa_utils')
        self.assertFalse(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                             and node.func.id == 'run_from_pipeline' for node in ast.walk(main)))

    def test_colab_notebook_python_cells(self):
        path = Path(rsa.__file__).parent / 'tools/colab_gpu/colab_rsa_cross_species.ipynb'
        notebook = json.loads(path.read_text(encoding='utf-8'))
        for index, cell in enumerate(notebook['cells']):
            source = ''.join(cell['source'])
            if cell['cell_type'] == 'code' and not source.startswith('%'):
                compile(source, f'colab-cell-{index}', 'exec')

    def test_role_named_cli_arguments(self):
        import argparse
        parser = argparse.ArgumentParser()
        rsa.add_arguments(parser)
        args = parser.parse_args(['--mask', 'source.nii.gz', '--target_mask', 'target.nii.gz',
                                  '--target_participants', '2', '--source_batch', '3', '--target_batch', '7'])
        self.assertEqual((args.mask, args.target_mask), ('source.nii.gz', 'target.nii.gz'))
        self.assertEqual((args.source_batch, args.target_batch), (3, 7))
        self.assertFalse(any(name.startswith(('dog_', 'human_')) for name in vars(args)))

    def test_reference_and_tiles(self):
        rng = np.random.default_rng(7)
        dog = rng.integers(-3, 4, (8, 9)).astype(float)
        human = rng.integers(-3, 4, (7, 9)).astype(float)
        dog[0] = 1
        dog[1, 0] = np.nan
        human[0] = -dog[2]
        human[1, 1] = np.inf
        actual = rsa.kendall_block(dog, human)
        expected = np.full(actual.shape, np.nan)
        for d, x in enumerate(dog):
            for h, y in enumerate(human):
                if np.isfinite(x).all() and np.isfinite(y).all():
                    expected[d, h] = kendalltau(x, y, variant='b').statistic
        np.testing.assert_allclose(actual, expected, atol=4e-8, equal_nan=True)
        self.assertEqual(actual[2, 0], -1)
        tiled = np.vstack([np.hstack([rsa.kendall_block(dog[d:d+2], human[h:h+3])
                                     for h in range(0, len(human), 3)])
                           for d in range(0, len(dog), 2)])
        np.testing.assert_array_equal(actual, tiled)

    def test_cuda_matches_cpu(self):
        # Probe in isolation: optional CPU-only torch and Anaconda MKL builds
        # can load conflicting OpenMP libraries into a scipy test process.
        probe = subprocess.run([sys.executable, '-c',
                                "import cross_species_rsa as r; print(r.pick_device('auto'))"],
                               capture_output=True, text=True)
        if probe.returncode != 0 or probe.stdout.strip() != 'cuda':
            self.skipTest('CUDA runtime unavailable')
        rng = np.random.default_rng(2)
        a, b = rng.integers(-4, 5, (11, 45)), rng.integers(-4, 5, (17, 45))
        np.testing.assert_array_equal(rsa.kendall_block(a, b, 'cuda'), rsa.kendall_block(a, b))


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.dataset = self.base / 'EmoC'
        classdir = self.dataset / 'rsa_models/by_class'
        classdir.mkdir(parents=True)
        self.csv = classdir / 'class_subset.csv'
        self.csv.write_text('cat1,cat2\nDogP,HumH\nDogP,DogH\nDogH,HumH\n')
        self.affine = np.diag([2., 3., 4., 1.])
        self.values = {}
        rng = np.random.default_rng(9)
        for species, shape in [('D', (2, 2, 1)), ('H', (3, 2, 1))]:
            config = self.dataset / f'config_files/{species}_basic-block.yaml'
            config.parent.mkdir(exist_ok=True)
            config.write_text(yaml.safe_dump(dict(participants=[1, 2], model='basic-block', task='EmoC')))
            mask = self.dataset / f'ROI/{species}/b_GreyMatter2mmB.nii.gz'
            mask.parent.mkdir(parents=True)
            mask_values = np.ones(shape, dtype=np.uint8)
            mask_values[-1, -1, -1] = 0
            nib.save(nib.Nifti1Image(mask_values, self.affine), mask)
            for sub in [1, 2]:
                maps = self.dataset / f'results/RSA/basic-block/{species}-sub-{sub:02d}'
                maps.mkdir(parents=True)
                values = rng.normal(size=(int(mask_values.sum()), 3))
                self.values[species, sub] = values
                for column, (a, b) in enumerate(rsa.read_pairs(self.csv)):
                    data = np.zeros(shape)
                    data[mask_values.astype(bool)] = values[:, column]
                    # All selected pairs are intentionally stored in reversed order.
                    radius = 5 if species == 'D' else 4
                    nib.save(nib.Nifti1Image(data, self.affine), maps / f'r-{radius}_mahalanobis_{b}_{a}.nii.gz')
        self.manifest = rsa.discover_inputs(self.base, rsa_class='subset', radius=5, target_radius=4)
        self.single = next(rsa.participant_manifests(self.manifest))[1]

    def test_step16_3_mean_and_validation(self):
        outputs = rsa.run_comparisons(self.manifest, self.base, device='cpu')
        # Include an undefined entry to verify the stated NaN policy.
        with h5py.File(outputs[0], 'r+') as handle:
            handle['correlations'][0, 0] = np.nan
            # Original step-16.2 v2 files had identity masks but no group-level
            # mask_type attributes, and used the historical batch names.
            for species in ('D', 'H'):
                del handle[species].attrs['mask_type']
            handle.attrs['dog_batch'] = handle.attrs.pop('source_batch')
            handle.attrs['human_batch'] = handle.attrs.pop('target_batch')
        output = rsa.group_cross_similarity(
            self.base, 'EmoC', 'basic-block', 'subset', radius=5,
            participants=[1, 2], source_batch=2, target_batch=2)
        with h5py.File(outputs[0]) as a, h5py.File(outputs[1]) as b:
            expected = (a['correlations'][:].astype(float) + b['correlations'][:]) / 2
        with rsa.CorrelationReader(output) as reader:
            np.testing.assert_allclose(reader.file['correlations'][:], expected, equal_nan=True)
            self.assertNotIn('dog_batch', reader.file.attrs)
            self.assertNotIn('human_batch', reader.file.attrs)
            self.assertEqual(reader.file.attrs['source_batch'], 2)
            image = reader.target_map(reader.file['D/ijk'][1])
            coords = reader.file['H/ijk'][:]
            np.testing.assert_allclose(image.get_fdata()[tuple(coords.T)], expected[1])
        maps = rsa.summarize_group_cross_similarity(output, source_batch=2, target_batch=2)
        self.assertEqual(len(maps), 21)
        with rsa.CorrelationReader(output) as reader:
            expected = reader.file['correlations'][:].astype(float)
            coords = reader.file['D/ijk'][:]
            outside = np.ones(tuple(reader.file['D/shape'][:]), dtype=bool)
            outside[tuple(coords.T)] = False
            mean_img = nib.load(maps[0])
            np.testing.assert_allclose(mean_img.get_fdata()[tuple(coords.T)],
                                       np.nanmean(expected, axis=1), atol=1e-7)
            np.testing.assert_allclose(mean_img.affine, reader.file['D/affine'][:])
            self.assertTrue(np.isnan(mean_img.get_fdata()[outside]).all())
            thresholds = np.linspace(np.nanmin(expected), np.nanmax(expected), 20)
            for threshold, map_path in zip(thresholds, maps[1:]):
                actual = nib.load(map_path).get_fdata()[tuple(coords.T)]
                np.testing.assert_array_equal(actual, (expected > threshold).sum(axis=1))
            self.assertEqual(maps[0].parent.name, 'mean')
        with self.assertRaises(FileNotFoundError):
            rsa.group_cross_similarity(self.base, 'EmoC', 'basic-block', 'subset',
                                       radius=5, participants=[1, 3])
        with h5py.File(outputs[1], 'r+') as handle:
            handle['H/affine'][0, 3] += 1
        with self.assertRaisesRegex(ValueError, 'H/affine'):
            rsa.group_cross_similarity(self.base, 'EmoC', 'basic-block', 'subset', radius=5)

    def test_group_weighting_and_roundtrip(self):
        for species in ('D', 'H'):
            _, _, values, _ = rsa.group_rdms(self.manifest, species)
            np.testing.assert_allclose(values, (self.values[species, 1] + self.values[species, 2]) / 2)
        first, second = self.base / 'first.h5', self.base / 'second.h5'
        rsa.run_analysis(self.single, first, device='cpu', source_batch=2, target_batch=3)
        rsa.run_analysis(self.single, second, device='cpu', source_batch=1, target_batch=1)
        with rsa.CorrelationReader(first) as reader, h5py.File(second) as other:
            matrix = reader.file['correlations'][:]
            # Only nonzero mask centres are allocated: full grids have 4 and 6 voxels.
            self.assertEqual(matrix.shape, (3, 5))
            self.assertEqual(len(reader.file['D/ijk']), 3)
            self.assertEqual(len(reader.file['H/ijk']), 5)
            self.assertFalse((reader.file['D/ijk'][:] == [1, 1, 0]).all(axis=1).any())
            self.assertFalse((reader.file['H/ijk'][:] == [2, 1, 0]).all(axis=1).any())
            np.testing.assert_array_equal(reader.file['D/group_rdms'][:], self.values['D', 1])
            np.testing.assert_allclose(reader.file['H/group_rdms'][:],
                                       (self.values['H', 1] + self.values['H', 2]) / 2)
            self.assertEqual(reader.file.attrs['source_participant'], 1)
            self.assertEqual(reader.file.attrs['radius'], 5)
            self.assertEqual(reader.file.attrs['target_radius'], 4)
            np.testing.assert_array_equal(matrix, other['correlations'][:])
            self.assertEqual(reader.file['correlations'].compression, 'gzip')
            expected = rsa.kendall_block(reader.file['D/group_rdms'][:], reader.file['H/group_rdms'][:])
            np.testing.assert_array_equal(matrix, expected)
            ijk = reader.file['D/ijk'][0]
            image = reader.human_map(ijk)
            np.testing.assert_array_equal(image.affine, self.affine)
            np.testing.assert_allclose(reader.file['H/xyz_mm'][:], nib.affines.apply_affine(self.affine, reader.file['H/ijk'][:]))
            hijk = reader.file['H/ijk'][:]
            np.testing.assert_array_equal(image.get_fdata()[tuple(hijk.T)], matrix[0])
            self.assertTrue(np.isnan(image.get_fdata()[-1, -1, -1]))
            assembled = np.empty_like(matrix)
            for ds, hs, values in reader.iter_blocks(1, 2):
                assembled[ds, hs] = values
            np.testing.assert_array_equal(assembled, matrix)
            with self.assertRaisesRegex(ValueError, 'outside'):
                reader.human_map([100, 0, 0])
            with self.assertRaisesRegex(ValueError, 'integer'):
                reader.human_map([0.2, 0, 0])

    def test_role_specific_radius_defaults(self):
        # The fixture has H=4 and D=5: H's omitted source radius should resolve
        # to 4, and an explicit target radius must remain independent.
        manifest = rsa.discover_inputs(self.base, rsa_class='subset',
            specie='H', target_specie='D', target_radius=5,
            participants=[2], target_participants=[1])
        self.assertEqual((manifest['radius'], manifest['target_radius']), (4, 5))
        # Group-only steps need no data or config files when resolving names.
        import rsa_utils
        with mock.patch.object(rsa, 'summarize_group_cross_similarity', return_value=['saved']) as summary:
            rsa_utils.calculate_group_level_cross_similarity_maps(self.base, 'EmoC',
                specie='H', target_specie='D', rsa_class='subset')
        self.assertEqual(summary.call_args.args[0].name,
            'b_GreyMatter2mmB-r-4_mahalanobis_kendall_vs_D-r-3.HDF5')

    def test_rsa_utils_public_entry_point(self):
        import rsa_utils
        output_root = self.base / 'public-api'
        with mock.patch.object(rsa, 'group_rdms', wraps=rsa.group_rdms) as loader:
            paths = rsa_utils.calculate_voxelwise_cross_similarity(
                datafolder=self.base, dataset='EmoC', model='basic-block',
                specie='D', target_specie='H', radius=5, target_radius=4,
                rsa_class='subset', participants=[1, 2], target_participants=[2],
                device='cpu', source_batch=1, target_batch=2, output_root=output_root)
        self.assertEqual([call.args[1] for call in loader.call_args_list], ['H', 'D', 'D'])
        self.assertEqual(len(paths), 2)
        for participant, path in enumerate(paths, 1):
            with rsa.CorrelationReader(path) as reader:
                expected = rsa.kendall_block(self.values['D', participant], self.values['H', 2])
                np.testing.assert_array_equal(reader.file['correlations'][:], expected)
        self.assertEqual(len([p for p in output_root.rglob('*') if p.is_file()]), 4)
        with self.assertRaisesRegex(ValueError, 'stim-wise'):
            rsa_utils.calculate_voxelwise_cross_similarity(self.base, 'EmoC', mah_fold='run-wise')

    def test_resume_and_input_change(self):
        output = self.base / 'resume.h5'
        def interrupt(path):
            # HDF5 must be closed and readable at the callback boundary.
            with h5py.File(path) as handle:
                self.assertTrue(handle['completed_blocks'][0].all())
            raise InterruptedError('synthetic disconnect')
        with self.assertRaises(InterruptedError):
            rsa.run_analysis(self.single, output, device='cpu', source_batch=1, target_batch=2, checkpoint_callback=interrupt)
        self.assertFalse(output.exists())
        with self.assertRaises(ValueError):
            rsa.CorrelationReader(str(output) + '.partial')
        original_csv = self.csv.read_text()
        self.csv.write_text(original_csv + '\n')
        with self.assertRaisesRegex(ValueError, 'Incompatible'):
            rsa.run_analysis(self.single, output, device='cpu')
        self.csv.write_text(original_csv)
        # Historical tile metadata used species names, even for reverse runs.
        # Read it as row/column sizes while keeping all new interfaces neutral.
        with h5py.File(str(output) + '.partial', 'r+') as handle:
            handle.attrs['dog_batch'] = handle.attrs.pop('source_batch')
            handle.attrs['human_batch'] = handle.attrs.pop('target_batch')
        rsa.run_analysis(self.single, output, device='cpu', source_batch=7, target_batch=9)
        self.assertTrue(output.exists())
        self.assertFalse(Path(str(output) + '.partial').exists())
        # Reuse a complete file with matching inputs.
        self.assertEqual(rsa.run_analysis(self.single, output, device='cpu'), output)
        self.csv.write_text(self.csv.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'Incompatible'):
            rsa.run_analysis(self.single, output, device='cpu')

    def test_invalid_rows_and_missing_maps(self):
        path = Path(self.manifest['species']['D']['participants'][0]['maps'][0])
        image = nib.load(path)
        data = image.get_fdata()
        data[0, 0, 0] = np.nan
        nib.save(nib.Nifti1Image(data, image.affine), path)
        _, _, values, valid = rsa.group_rdms(self.manifest, 'D')
        self.assertFalse(valid[0])
        self.assertTrue(np.isnan(values[0]).all())
        for participant in self.manifest['species']['D']['participants']:
            for filename in participant['maps']:
                current = nib.load(filename)
                data = current.get_fdata()
                data[0, 1, 0] = 2
                nib.save(nib.Nifti1Image(data, current.affine), filename)
        _, _, values, valid = rsa.group_rdms(self.manifest, 'D')
        self.assertFalse(valid[1])  # constant group RDM, even though all inputs are finite
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            rsa.discover_inputs(self.base, rsa_class='subset', radius=5, target_radius=4)

    def test_duplicate_pairs_and_grid(self):
        self.csv.write_text('cat1,cat2\nDogH,HumH\nHumH,DogH\n')
        with self.assertRaisesRegex(ValueError, 'unique'):
            rsa.read_pairs(self.csv)
        self.csv.write_text('cat1,cat2\nDogP,HumH\nDogP,DogH\nDogH,HumH\n')
        path = Path(self.manifest['species']['H']['participants'][0]['maps'][0])
        image = nib.load(path)
        wrong_affine = image.affine.copy()
        wrong_affine[0, 3] = 4
        nib.save(nib.Nifti1Image(image.get_fdata(), wrong_affine), path)
        with self.assertRaisesRegex(ValueError, 'grid mismatch'):
            rsa.discover_inputs(self.base, rsa_class='subset', radius=5, target_radius=4)
        with self.assertRaisesRegex(ValueError, 'grid mismatch'):
            rsa.group_rdms(self.manifest, 'H')

    def test_package_colab_and_local_cli(self):
        package = create_package(self.manifest, self.base / 'package.zip')
        # Execute only the shipped code outside the repository, as Colab does.
        bootstrap = extract_package(package, self.base / 'bootstrap')
        standalone = self.base / 'standalone'
        isolated = subprocess.run([sys.executable, str(bootstrap / 'run_colab_cross_species.py'),
                                   '--package', str(package), '--output', str(standalone),
                                   '--staging', str(self.base / 'isolated'), '--device', 'cpu'],
                                  cwd=self.base, capture_output=True, text=True)
        self.assertEqual(isolated.returncode, 0, isolated.stdout + isolated.stderr)
        output = self.base / 'drive'
        outputs = run_colab(package, output, self.base / 'colab', device='cpu', source_batch=1,
                            target_batch=2, checkpoint_minutes=0)
        self.assertEqual(len(outputs), 2)
        self.assertEqual(sorted(p.suffix for p in output.rglob('*') if p.is_file()),
                         ['.HDF5', '.HDF5', '.log', '.log'])
        # A fresh runtime also restores completed outputs, without extra checkpoint files.
        run_colab(package, output, self.base / 'fresh-colab', device='cpu')
        # Real pipeline dispatch must avoid all atlas/model/single-species initialization.
        repo = Path(rsa.__file__).parent
        local = self.base / 'cli'
        result = subprocess.run([sys.executable, str(repo / 'searchlight.py'), '--steps_to_run', '16.2',
                                 '--datafolder', str(self.base), '--dataset', 'EmoC', '--model', 'basic-block',
                                 '--rsa_class', 'subset', '--rsa_model', 'deliberately_absent',
                                 '--specie', 'D', '--target_specie', 'H', '--radius', '5', '--target_radius', '4',
                                 '--cross_species_device', 'cpu',
                                 '--cross_species_output', str(local)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for participant in (1, 2):
            relative = rsa.relative_output_path(self.manifest, participant)
            with h5py.File(local / relative) as a, h5py.File(output / relative) as b, h5py.File(standalone / relative) as c:
                np.testing.assert_array_equal(a['correlations'][:], b['correlations'][:])
                np.testing.assert_array_equal(a['correlations'][:], c['correlations'][:])
                self.assertEqual(a.attrs['signature'], b.attrs['signature'])
                np.testing.assert_array_equal(a['D/group_rdms'][:], self.values['D', participant])

    def test_output_path_cohort_and_reverse_direction(self):
        expected = ('EmoC/results/RSA_cross-specie/basic-block/class_subset/D-sub-01/'
                    'b_GreyMatter2mmB-r-5_mahalanobis_kendall_vs_H-r-4.HDF5')
        self.assertEqual(rsa.relative_output_path(self.manifest, 1).as_posix(), expected)
        with mock.patch.object(rsa, 'group_rdms', wraps=rsa.group_rdms) as loader:
            outputs = rsa.run_comparisons(self.manifest, self.base / 'outputs', device='cpu')
        self.assertEqual([call.args[1] for call in loader.call_args_list], ['H', 'D', 'D'])
        files = [p for p in (self.base / 'outputs').rglob('*') if p.is_file()]
        self.assertEqual(len(files), 4)
        self.assertTrue(all(p.suffix in ('.HDF5', '.log') for p in files))
        for path in outputs:
            self.assertIn('target_specie', path.with_suffix('.log').read_text())
        reverse = rsa.discover_inputs(self.base, rsa_class='subset', radius=4,
                                      specie='H', target_specie='D', target_radius=5,
                                      participants=[2], target_participants=[1, 2])
        path = rsa.run_comparisons(reverse, self.base / 'reverse', device='cpu')[0]
        self.assertTrue(path.name.endswith('_vs_D-r-5.HDF5'))
        with rsa.CorrelationReader(path) as reader:
            self.assertEqual(reader.file['correlations'].shape, (5, 3))
            np.testing.assert_array_equal(reader.file['H/group_rdms'][:], self.values['H', 2])
            target_image = reader.target_map(reader.file['H/ijk'][0])
            self.assertEqual(target_image.shape, (2, 2, 1))
            with self.assertRaisesRegex(ValueError, 'D-to-H'):
                reader.human_map([0, 0, 0])

    def test_source_target_selections_and_summaries_both_directions(self):
        repo = Path(rsa.__file__).parent
        for source, target, radius, target_radius in [('D', 'H', 5, 4), ('H', 'D', 4, 5)]:
            with self.subTest(source=source):
                source_mask = self.base / f'source_{source}.nii.gz'
                target_mask = self.base / f'target_{target}.nii.gz'
                nib.save(nib.load(self.dataset / f'ROI/{source}/b_GreyMatter2mmB.nii.gz'), source_mask)
                nib.save(nib.load(self.dataset / f'ROI/{target}/b_GreyMatter2mmB.nii.gz'), target_mask)
                root = self.base / f'roles_{source}'
                command = [sys.executable, str(repo / 'searchlight.py'),
                           '--steps_to_run', '16.2', '16.3', '16.4',
                           '--datafolder', str(self.base), '--dataset', 'EmoC',
                           '--rsa_class', 'subset', '--model', 'basic-block',
                           '--specie', source, '--target_specie', target,
                           '--radius', str(radius), '--target_radius', str(target_radius),
                           '--participants', '2', '--target_participants', '1',
                           '--mask', str(source_mask), '--target_mask', str(target_mask),
                           '--source_batch', '2', '--target_batch', '3',
                           '--cross_species_device', 'cpu', '--cross_species_output', str(root)]
                process = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                name = f'source_{source}-r-{radius}_mahalanobis_kendall_vs_{target}-r-{target_radius}.HDF5'
                folder = root / 'EmoC/results/RSA_cross-specie/basic-block/class_subset'
                with rsa.CorrelationReader(folder / f'{source}-sub-02' / name) as reader:
                    self.assertEqual(reader.file[f'{source}/participants'][:].tolist(), [2])
                    self.assertEqual(reader.file[f'{target}/participants'][:].tolist(), [1])
                    self.assertEqual(reader.file.attrs['source_batch'], 2)
                    self.assertEqual(reader.file.attrs['target_batch'], 3)
                    expected = rsa.kendall_block(self.values[source, 2], self.values[target, 1])
                    np.testing.assert_array_equal(reader.file['correlations'][:], expected)
                group_path = folder / f'{source}-group' / name
                with rsa.CorrelationReader(group_path) as reader:
                    np.testing.assert_array_equal(reader.file['correlations'][:], expected)
                    coords = reader.file[f'{source}/ijk'][:]
                    shape = tuple(reader.file[f'{source}/shape'][:])
                stem = Path(name).stem
                maps = list((folder / 'mean').glob('*.nii.gz'))
                self.assertEqual(len(maps), 21)
                average = nib.load(folder / 'mean' / f'{stem}_mean.nii.gz')
                self.assertEqual(average.shape, shape)
                np.testing.assert_allclose(average.get_fdata()[tuple(coords.T)], expected.mean(axis=1), atol=1e-7)
                metadata = json.loads((folder / 'mean' / f'{stem}_summary.json').read_text())
                self.assertEqual(metadata['source_specie'], source)
                self.assertEqual(metadata['target_specie'], target)
                self.assertEqual(metadata['number_target_voxels'], expected.shape[1])
                for index, threshold in enumerate(metadata['thresholds'], 1):
                    counts = nib.load(folder / 'mean' / f'{stem}_threshold-{index:02d}_count.nii.gz')
                    np.testing.assert_array_equal(counts.get_fdata()[tuple(coords.T)],
                                                  (expected.astype(float) > threshold).sum(axis=1))
                # Both mask and cohort mismatches must fail against saved metadata.
                with self.assertRaisesRegex(ValueError, 'mask selection'):
                    rsa.group_cross_similarity(root, 'EmoC', 'basic-block', 'subset',
                        source, target, radius, target_radius, mask=source_mask,
                        target_mask=self.base / 'wrong.nii.gz')
                with self.assertRaisesRegex(ValueError, 'Target participant'):
                    rsa.group_cross_similarity(root, 'EmoC', 'basic-block', 'subset',
                        source, target, radius, target_radius, mask=source_mask,
                        target_mask=target_mask, target_participants=[2])

    def test_colab_partial_snapshot_resume(self):
        package = create_package(self.single, self.base / 'single.zip')
        real_kernel = rsa.kendall_block
        count = 0
        def interrupted_kernel(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 4:
                raise InterruptedError('synthetic GPU interruption')
            return real_kernel(*args, **kwargs)
        output = self.base / 'drive'
        with mock.patch.object(rsa, 'kendall_block', side_effect=interrupted_kernel):
            with self.assertRaises(InterruptedError):
                run_colab(package, output, self.base / 'interrupted', device='cpu',
                          source_batch=1, target_batch=2, checkpoint_minutes=0)
        path = output / rsa.relative_output_path(self.single, 1)
        with h5py.File(path) as handle:
            self.assertFalse(handle.attrs['complete'])
            self.assertTrue(handle['completed_blocks'][0].all())
        run_colab(package, output, self.base / 'restarted', device='cpu')
        with rsa.CorrelationReader(path) as reader:
            expected = rsa.kendall_block(self.values['D', 1],
                                         (self.values['H', 1] + self.values['H', 2]) / 2)
            np.testing.assert_array_equal(reader.file['correlations'][:], expected)
        self.assertEqual(len([p for p in output.rglob('*') if p.is_file()]), 2)

    def test_manifest_validation(self):
        bad = copy.deepcopy(self.single)
        bad['mah_fold'] = 'run-wise'
        with self.assertRaisesRegex(ValueError, 'stim-wise'):
            rsa.run_analysis(bad, self.base / 'bad.h5')
        with self.assertRaisesRegex(ValueError, 'positive'):
            rsa.run_analysis(self.manifest, self.base / 'bad.h5', source_batch=0)


if __name__ == '__main__':
    unittest.main()
