# Plot histogram with the filter reasons for each filtering stage
import os
import argparse
import matplotlib.pyplot as plt
import json
import numpy as np
from matplotlib.colors import CSS4_COLORS

# Choose a set of distinct colors for different files
COLORS = ['royalblue', 'crimson', 'forestgreen', 'darkorange', 'purple']


def format_label(label):
    """Format the filter reason labels to be more readable."""
    label = label.replace('_', ' ')
    return label.capitalize()


def format_count(x, pos=None):
    """Format large numbers with K/M suffixes."""
    if x >= 1_000_000:
        return f'{x / 1_000_000:.1f}M'
    elif x >= 1_000:
        return f'{x / 1_000:.1f}K'
    return str(int(x))


def plot_filter_histogram(all_data, stage, output_dir):
    """Plot histogram for a single filtering stage with data from multiple files."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Get unique reasons across all files
    all_reasons = set()
    for file_data in all_data.values():
        if stage in file_data:
            all_reasons.update([k for k in file_data[stage].keys() if k != 'total'])

    all_reasons = sorted(list(all_reasons))

    # Count valid files for this stage
    valid_files = [f for f, data in all_data.items() if stage in data]
    num_files = len(valid_files)

    # Calculate bar width based on number of files
    bar_width = 0.8 / num_files if num_files > 0 else 0.8

    # Plot bars for each file side by side
    for i, (filename, file_data) in enumerate(all_data.items()):
        if stage not in file_data:
            continue

        data = file_data[stage]
        total = data['total']

        # Prepare data for plotting
        y_values = []
        percentages = []

        for reason in all_reasons:
            count = data.get(reason, 0)
            y_values.append(count)
            percentages.append(100 * count / total if total else 0)

        # Calculate the x positions with offset for side-by-side bars
        x_positions = np.arange(len(all_reasons)) - (0.4 - bar_width / 2) + i * bar_width

        # Plot bars side by side
        bars = ax.bar(x_positions, y_values, width=bar_width,
                      color=COLORS[i % len(COLORS)],
                      label=filename.split("_")[0] + " - " + format_count(total))

        # Add percentage labels - for both >0 and 0 counts
        for j, (bar, perc, y_val) in enumerate(zip(bars, percentages, y_values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.,
                    height,  # 0 for zero bars
                    f'{perc:.1f}%',
                    ha='center', va='bottom',
                    fontsize=8, color=COLORS[i % len(COLORS)],
                    fontweight='bold')

    # Format x-axis labels
    ax.set_xticks(range(len(all_reasons)))
    ax.set_xticklabels([format_label(label) for label in all_reasons],
                       rotation=45, ha='right')

    # Format y-axis
    ax.set_ylabel('Count')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_count))

    # Set title
    title_stage = stage.replace('_', ' ').title()
    ax.set_title(f'Filter Reasons for {title_stage}')

    # Add legend
    ax.legend()

    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(f'{output_dir}/filt_stats_{stage}.png', bbox_inches='tight')
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot filter reason histograms.")
    parser.add_argument("--input-dir", required=True,
                        help="Path to directory containing *_filter_reasons.json files.")
    parser.add_argument("--output-dir", default="pngs",
                        help="Directory to save output plots (default: pngs).")
    parser.add_argument("--files", nargs="+", required=True,
                        help="Base names of the filter reason JSON files (without _filter_reasons.json suffix).")
    args = parser.parse_args()

    # Load the filter reasons data for all files
    all_filter_data = {}
    for filename in args.files:
        file_path = os.path.join(args.input_dir, filename + "_filter_reasons.json")
        try:
            with open(file_path, 'r') as f:
                all_filter_data[filename] = json.load(f)
        except FileNotFoundError:
            print(f"Warning: File {file_path} not found")

    # Find all unique stages across all files
    all_stages = set()
    for file_data in all_filter_data.values():
        all_stages.update(file_data.keys())

    # Create a figure for each filtering stage
    for stage in all_stages:
        fig = plot_filter_histogram(all_filter_data, stage, args.output_dir)
        plt.close(fig)  # Close the figure to free memory

    print(f"Plots saved to {args.output_dir} directory")
