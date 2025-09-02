#!/usr/bin/env python3
"""
vcf2img.py
  Generate barcode heatmaps from VCF and defined sample-pair CSV.

Usage example:
  python vcf2img.py --vcf(-v) [sample.vcf] --pair(-p) [sample_pair.csv] --output(-o) [output_path]
"""

import importlib
import sys
import allel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import os
import argparse


def check_modules():

    required = {
        "allel": "allel",
        "pandas": "pandas",
        "numpy": "numpy",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
    }

    missing = []
    for module, pip_name in required.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(pip_name)

    if missing:
        print(" Following dependencies are missing: ")
        print("  pip install " + " ".join(missing))
        sys.exit(1)

def main(vcf_path: str, csv_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # 1. reading VCF
    print('Reading VCF...')
    callset = allel.read_vcf(
        vcf_path,
        fields=['samples', 'calldata/GT', 'variants/CHROM', 'variants/POS']
    )
    gt = allel.GenotypeArray(callset['calldata/GT'])
    samples = callset['samples']
    chromosomes = callset['variants/CHROM']
    print('Reading is done...')

    # Preprocessing chromosomes
    chrom_counts = {str(c): np.sum(chromosomes == str(c)) for c in range(1, 23)}
    max_length = max(chrom_counts.values())

    pairs_df = pd.read_csv(csv_path)

    # 2. Matching and encoding
    code_map = {'00': 0, '01': 1, '10': 2, '11': 3}

    for _, row in pairs_df.iterrows():
        sample1, sample2, pair_label = row['Sample1'], row['Sample2'], row['label']

        filename = f"{sample1}_{sample2}-{pair_label}.png".replace(" ", "_").replace("/", "_")
        idx1 = list(samples).index(sample1)
        idx2 = list(samples).index(sample2)

        chrom_matrix = []
        for chrom in sorted(chrom_counts.keys(), key=int):
            mask = (chromosomes == chrom)
            chr_gt = gt[mask]

            chr_codes = []
            for pos_idx in range(len(chr_gt)):
                gt1 = chr_gt[pos_idx, idx1]
                gt2 = chr_gt[pos_idx, idx2]

                code = [
                    '1' if gt1[0] == gt2[0] else '0',
                    '1' if gt1[1] == gt2[1] else '0'
                ]
                merged_code = ''.join(code)
                chr_codes.append(code_map[merged_code])

            padded = chr_codes + [np.nan] * (max_length - len(chr_codes))
            chrom_matrix.append(padded)

        # Creating color matrix
        matrix_df = pd.DataFrame(
            chrom_matrix,
            index=[f'chr{c}' for c in range(1, 23)],
            columns=[f'pos_{i + 1}' for i in range(max_length)]
        )
        transposed_df = matrix_df.T

        # Color encoding
        cmap = mcolors.ListedColormap([
            '#FF0000', '#00FF00', '#0000FF', '#FFFF00'
        ])
        norm = mcolors.Normalize(vmin=0, vmax=3)

        # Generating barcode heatmap
        plt.figure(figsize=(2, 100))
        sns.heatmap(
            transposed_df,
            cmap=cmap,
            norm=norm,
            linewidths=0,
            cbar=False,
            xticklabels=False,
            yticklabels=False,
            mask=transposed_df.isna()
        )
        out_file = os.path.join(out_dir, filename)
        plt.savefig(out_file, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Generating: {out_file}")


if __name__ == '__main__':
    check_modules()
    parser = argparse.ArgumentParser(description="Generate barcode heatmaps from VCF and sample pair CSV.")
    parser.add_argument('-v', '--vcf', required=True, help='Input VCF file path')
    parser.add_argument('-p', '--pair', required=True, help='Input CSV file with defined sample pairs')
    parser.add_argument('-o', '--out', default='./input', help='Output directory (default: ./input)')
    args = parser.parse_args()

    main(args.vcf, args.pair, args.out)