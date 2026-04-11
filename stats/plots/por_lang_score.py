import argparse
import os
import matplotlib.pyplot as plt
import json
import numpy as np
from matplotlib.ticker import PercentFormatter


def format_count(x, pos=None):
    """Format large numbers with K/M suffixes."""
    if x >= 1_000_000:
        return f'{x / 1_000_000:.1f}M'
    elif x >= 1_000:
        return f'{x / 1_000:.1f}K'
    return str(int(x))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Portuguese language score distribution.")
    parser.add_argument("--input-file", required=True,
                        help="Path to the *_por_scores.json file.")
    parser.add_argument("--output-dir", default="pngs",
                        help="Directory to save output plots (default: pngs).")
    args = parser.parse_args()

    # Load the Portuguese language scores data
    with open(args.input_file, 'r') as f:
        scores_data = json.load(f)

    # Convert string keys to float and round to 2 decimal places
    # Group identical rounded scores by summing their counts
    rounded_scores = {}
    for score_str, count in scores_data.items():
        rounded_score = round(float(score_str), 3)
        rounded_scores[rounded_score] = rounded_scores.get(rounded_score, 0) + count

    # Sort by score value
    sorted_scores = dict(sorted(rounded_scores.items()))

    # Extract data for plotting
    score_values = list(sorted_scores.keys())
    doc_counts = list(sorted_scores.values())

    # Calculate total documents
    total_docs = sum(doc_counts)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Create the line plot without markers
    ax.plot(score_values, doc_counts, linestyle='-', color='blue', linewidth=1.5)

    # Set y-axis to logarithmic scale to make lower values more distinguishable
    ax.set_yscale('log')

    # Format axes
    ax.set_xlabel('Portuguese Language Score', fontsize=12)
    ax.set_ylabel('Number of Documents (log scale)', fontsize=12)
    ax.set_title(f'Rejected Portuguese Language Score Distribution (Total: {format_count(total_docs)})', fontsize=14,
                 fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_count))

    # Add grid for better readability (compatible with log scale)
    ax.grid(True, which='both', linestyle='--', alpha=0.7)

    plt.tight_layout()
    os.makedirs(args.output_dir, exist_ok=True)
    plt.savefig(f'{args.output_dir}/portuguese_language_score_distribution.png', bbox_inches='tight')
    plt.show()
