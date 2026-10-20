"""Create a comparison class from the GLM config's method-specific labels."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsa_utils import create_rsa_class


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Path to the GLM YAML config')
    parser.add_argument('--rsa_class', required=True, help='Class name, e.g. all_mahalanobis')
    parser.add_argument('--dis_method', choices=['mahalanobis', 'correlation'], default='mahalanobis')
    parser.add_argument('--mah_fold', default='stim-wise', choices=[
        'stim-wise', 'run-wise', 'stim-wise-multiple-folds', 'stim-wise-all-runs'])
    parser.add_argument('--stimuli', nargs='+', help='Optional subset of available labels')
    parser.add_argument('--output_dir', help='Override rsa_models/by_class output directory')
    parser.add_argument('--replace_file', action='store_true')
    args = parser.parse_args()
    path = create_rsa_class(args.config, args.rsa_class, args.dis_method,
                            mah_fold=args.mah_fold, stimuli=args.stimuli,
                            output_dir=args.output_dir, replace_file=args.replace_file)
    print(f'Saved class: {path}')


if __name__ == '__main__':
    main()
