import argparse
import matplotlib.pyplot as plt
import json
import numpy as np


def format_count(x, pos=None):
    """Format large numbers with K/M suffixes."""
    if x >= 1_000_000:
        return f'{x/1_000_000:.1f}M'
    elif x >= 1_000:
        return f'{x/1_000:.1f}K'
    return str(int(x))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot language filter distribution.")
    parser.add_argument("--input-file", required=True,
                        help="Path to the *_languages.json file.")
    parser.add_argument("--output-dir", default="pngs",
                        help="Directory to save output plots (default: pngs).")
    args = parser.parse_args()

    # Load the language data
    with open(args.input_file, 'r') as f:
        language_data = json.load(f)

    # Sort languages by count (descending)
    sorted_languages = sorted(language_data.items(), key=lambda x: x[1], reverse=True)

    # Take top N languages for better visualization
    TOP_N = 40
    top_languages = sorted_languages[:TOP_N]
    other_count = sum(count for _, count in sorted_languages[TOP_N:])
    total_count = sum(count for _, count in sorted_languages)

    # Extract language codes and counts
    langs = [lang for lang, _ in top_languages]
    counts = [count for _, count in top_languages]

    # Add the "Others" category
    langs.append("Others")
    counts.append(other_count)

    # Create horizontal bar chart (better for text labels)
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot horizontal bars with log scale
    bars = ax.barh(langs, counts, color='skyblue', edgecolor='navy', log=True)

    # Add count labels to the bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        label = format_count(counts[i])
        ax.text(width, bar.get_y() + bar.get_height()/2,
                label, ha='left', va='center',
                fontweight='bold', color='navy', fontsize=10)

    # Format the axes
    ax.set_xlabel('Number of Documents (log scale)')
    ax.set_ylabel('Language Code')
    ax.set_title(f'Number of Documents Filtered by Language (Total: {format_count(total_count)})', fontsize=14, fontweight='bold')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_count))
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    # Add padding for count labels
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)

    import os
    os.makedirs(args.output_dir, exist_ok=True)
    plt.savefig(f'{args.output_dir}/language_distribution.png', bbox_inches='tight')
    plt.show()
