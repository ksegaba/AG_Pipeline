#!/usr/bin/env python3

"""Aggregate AutoGluon performance results into one TSV file."""

import argparse, sys, os
import pandas as pd

def parse_args():
    # Argument parser
    parser = argparse.ArgumentParser(
        description="Aggregate AutoGluon performance results into one TSV file.")
    
    # Required input
    req_group = parser.add_argument_group('Required input')
    req_group.add_argument(
        "-work_dir", help="Path to the working directory containing AutoGluon results.",
        required=True, type=str)
    req_group.add_argument(
        "-out_file", help="Path to the output TSV file for aggregated results.",
        required=True, type=str)
    
    # Help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    
    return parser.parse_args() # Read arguments from the command line


if __name__ == "__main__":
    args = parse_args()
    work_dir = args.work_dir
    out_file = args.out_file

    agg_results = [] # Aggregated performance results dictionary
    # agg_imp = {} # Aggregated feature importance dictionary
    
    # Iterate over each folder in the working directory
    for folder in os.listdir(work_dir):
        folder_path = os.path.join(work_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        
        # Read in the performance results from the folder
        result_file = os.path.join(folder_path, f"{folder}_RESULTS.csv")
        try:
            assert os.path.exists(result_file), ""
        except AssertionError:
            print(f"Results file not found: {result_file}")
            continue
        
        result_df = pd.read_csv(result_file, index_col=0)
        result_df.insert(0, "Tag", folder)
        agg_results.append(result_df)
        
        # Read in the feature importance files
        # ... Not sure if it makes to aggregate these.

    # Concatenate all results into a single DataFrame
    agg_res_df = pd.concat(agg_results, axis=0, ignore_index=True)
    agg_res_df.to_csv(out_file, sep="\t", index=False)