import argparse
import os
import orjson
import matplotlib.pyplot as plt
import numpy as np


def plot_domain_distribution(data, quality, output_path):
    """Plots the domain distribution for a given quality."""
    domain_distribution = data.get(quality, {}).get("domain_distribution", {})
    if not domain_distribution:
        print(f"No domain distribution data for {quality}.")
        return

    domains = list(domain_distribution.keys())
    counts = list(domain_distribution.values())

    # Sort domains by counts in descending order and take top N
    top_n = 50  # Plotting top 50 domains
    top_domains_indices = np.argsort(counts)[-top_n:]
    top_domains = [domains[i] for i in top_domains_indices]
    top_counts = [counts[i] for i in top_domains_indices]

    plt.figure(figsize=(12, 10))
    plt.barh(top_domains, top_counts, color='skyblue')
    plt.ylabel("Domains")
    plt.xlabel("Counts")
    if quality == 'total':
        plt.title(f"Domain Distribution ({quality})")
    else:
        plt.title(f"Domain Distribution ({quality} quality)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_path}/domain_distribution_{quality}.png")
    plt.close()
    print(f"Domain distribution plot saved for {quality}.")


def plot_token_length_distribution(data, quality, output_path):
    """Plots the token length distribution for a given quality."""
    token_distribution = data.get(quality, {}).get("token_distribution", {})
    if not token_distribution:
        print(f"No token distribution data for {quality}.")
        return

    # Token ranges and their corresponding counts
    token_ranges = list(token_distribution.keys())
    counts = list(token_distribution.values())

    # Filter token ranges and counts based on the threshold
    filtered_token_ranges = []
    filtered_counts = []
    for token_range, count in zip(token_ranges, counts):
        if count > 100 and token_range != "4k":
            filtered_token_ranges.append(token_range)
            filtered_counts.append(count)

    plt.figure(figsize=(10, 6))
    plt.bar(filtered_token_ranges, filtered_counts, color='lightgreen')
    plt.xlabel("Token Ranges")
    plt.ylabel("Counts")
    if quality == 'total':
        plt.title(f"Token Length Distribution ({quality})")
    else:
        plt.title(f"Token Length Distribution ({quality} quality)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_path}/token_length_distribution_{quality}.png")
    plt.close()
    print(f"Token length distribution plot saved for {quality}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot domain and token length distributions.")
    parser.add_argument("--input-file", required=True,
                        help="Path to the token_counts JSON file.")
    parser.add_argument("--output-dir", default="pngs",
                        help="Directory to save output plots (default: pngs).")
    args = parser.parse_args()

    # Load data from JSON file
    with open(args.input_file, 'r') as f:
        data = orjson.loads(f.read())

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Plot domain distribution for total and high quality
    plot_domain_distribution(data, 'total', args.output_dir)
    plot_domain_distribution(data, 'high', args.output_dir)

    # Plot token length distribution for total and high quality
    plot_token_length_distribution(data, 'total', args.output_dir)
    plot_token_length_distribution(data, 'high', args.output_dir)

    print("Plots generated successfully!")