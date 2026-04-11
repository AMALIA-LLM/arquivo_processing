import argparse
import orjson
import gzip
import os
from urllib.parse import urlparse


QUALITY_LABELS = ["medium", "high"]

def analyze_token_counts(data_folder):
    """
    Analyzes token counts from .jsonl.gz files in a directory,
    calculates the sum of token_count, and counts entries in different token count ranges.
    Writes the results to a JSON file.
    """

    total_entries = 0
    total_token_count = 0
    token_distribution = {
        "4k": 0,
        "8k": 0,
        "16k": 0,
        "32k": 0,
        "64k": 0,
        "128k": 0,
        "256k": 0,
    }
    domain_distribution = {}
    collections = {}

    for filename in os.listdir(data_folder):
        if filename.endswith(".jsonl.gz"):
            file_path = os.path.join(data_folder, filename)
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    try:
                        total_entries += 1
                        data = orjson.loads(line)
                        token_count = data.get("metadata", {}).get("token_count", 0)
                        total_token_count += token_count

                        if token_count >= 4000:
                            token_distribution["4k"] += 1
                        if token_count >= 8000:
                            token_distribution["8k"] += 1
                        if token_count >= 16000:
                            token_distribution["16k"] += 1
                        if token_count >= 32000:
                            token_distribution["32k"] += 1
                        if token_count >= 64000:
                            token_distribution["64k"] += 1
                        if token_count >= 128000:
                            token_distribution["128k"] += 1
                        if token_count >= 256000:
                            token_distribution["256k"] += 1

                        # Extract domain from URL
                        url = data.get("metadata", {}).get("url", None)
                        if url:
                            domain = urlparse(url).netloc
                            if domain:
                                domain_distribution[domain] = domain_distribution.get(domain, 0) + 1

                        collection_name = filename.split("_")[0]
                        collections[collection_name] = collections.get(collection_name, 0) + token_count

                    except Exception as e:
                        print(f"An error occurred: {e}")

    # Prepare data for JSON output
    token_counts = {
        "total_entries": total_entries,
        "total_token_count": total_token_count,
        "token_distribution": token_distribution,
        "domain_distribution": domain_distribution,
        "collections": collections
    }

    return token_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze token counts from quality-classified JSONL files."
    )
    parser.add_argument("--data-dir", required=True,
                        help="Path to the directory containing quality-classified subdirectories.")
    parser.add_argument("--output-file", required=True,
                        help="Path to the output JSON file for token count results.")
    args = parser.parse_args()

    output_data = {'total': {
        "total_entries": 0,
        "total_token_count": 0,
        "token_distribution": {},
        "domain_distribution": {},
        "collections": {}
    }, 'low': {}, 'medium': {}, 'high': {}}

    # Check if data directory exists, if not, skip processing
    if not os.path.exists(args.data_dir):
        print(f"Directory {args.data_dir} does not exist. Skipping processing.")
    else:
        for label in QUALITY_LABELS:
            print(f"Processing {label}...", flush=True)
            data_path = os.path.join(args.data_dir, label)
            output_data[label] = analyze_token_counts(data_path)

            # Calculate total counts
            output_data['total']['total_entries'] += output_data[label]['total_entries']
            output_data['total']['total_token_count'] += output_data[label]['total_token_count']
            for key in output_data[label]['token_distribution']:
                if key not in output_data['total']['token_distribution']:
                    output_data['total']['token_distribution'][key] = 0
                output_data['total']['token_distribution'][key] += output_data[label]['token_distribution'][key]
            for key in output_data[label]['domain_distribution']:
                if key not in output_data['total']['domain_distribution']:
                    output_data['total']['domain_distribution'][key] = 0
                output_data['total']['domain_distribution'][key] += output_data[label]['domain_distribution'][key]
            for key in output_data[label]['collections']:
                if key not in output_data['total']['collections']:
                    output_data['total']['collections'][key] = 0
                output_data['total']['collections'][key] += output_data[label]['collections'][key]

        print(f"Total status: {output_data['total']}")

    # Write results to a JSON file
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as outfile:
        outfile.write(orjson.dumps(output_data, option=orjson.OPT_INDENT_2).decode('utf-8'))

    print(f"Token count analysis completed. Results written to {args.output_file}")
