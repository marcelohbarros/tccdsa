import argparse

import preset as p

def parse_args():
    parser = argparse.ArgumentParser(description="Read a CSV file and print its contents.")
    parser.add_argument('--data_path', type=str, default='tccdsa/datasets', help='Path to the CSV file to open (default: datasets/tomcat.csv)')
    parser.add_argument('--repetitions', type=int, default=10, help='Number of times to repeat each experiment (default: 10)')
    parser.add_argument('--preset', type=str, nargs='+', choices=p.PreSet.all_names(), default=[], help='If set, use a predefined configuration (default: all presets)')
    parser.add_argument('--verbose', action='store_true', help='If set, print verbose output (default: False)')

    args = parser.parse_args()

    presets = p.PreSet.from_names(args.preset)

    return args.data_path, args.repetitions, presets, args.verbose

data_path, repetitions, presets, verbose = parse_args()
