import os
import gzip
import json
import time
import argparse

REMOVED_DATA_PATH = "1_filter/removed"
LANGUAGE_FILTER_STEP = "1_lang_exclusion"
FILTERING_STEPS = ["0_latest_url", "2_gopher_rep", "3_fineweb_qual", "4_gopher_qual"]


def por_language_score(data_path, output_path, collections):
    """
    This function generates statistics for the language filter step for the Portuguese language.
    It saves the distribution of language scores for documents that were removed to later be plotted,
    counting the number of documents removed for each score.
    """

    scores = {}

    for collection in (collections if collections else os.listdir(data_path)):
        print(f"\tProcessing {collection}...", flush=True)
        filtering_step_path = os.path.join(data_path, collection, REMOVED_DATA_PATH, LANGUAGE_FILTER_STEP)
        if not os.path.exists(filtering_step_path):
            continue

        por_path = os.path.join(filtering_step_path, "por")
        if not os.path.exists(por_path):
            continue

        for file_name in os.listdir(por_path):
            if file_name.endswith(".jsonl.gz"):
                with gzip.open(os.path.join(por_path, file_name), 'rt', encoding='utf-8') as f:
                    for line in f:
                        doc = json.loads(line)
                        score = doc["metadata"]["language_score"]
                        if score not in scores:
                            scores[score] = 0
                        scores[score] += 1

    # Write JSON with the scores
    file_name = f"{collections[0]}_por_scores.json" if len(collections) == 1 else "por_scores.json"
    with open(os.path.join(output_path, file_name), 'w') as f:
        json.dump(scores, f, indent=4)


def count_language_filter(data_path, output_path, collections):
    """
    This function generates statistics for the language filter step.
    It counts the number of documents removed for each language and saves the results in a JSON file.
    """
    languages = {}

    for collection in (collections if collections else os.listdir(data_path)):
        print(f"\tProcessing {collection}...", flush=True)
        filtering_step_path = os.path.join(data_path, collection, REMOVED_DATA_PATH, LANGUAGE_FILTER_STEP)
        if not os.path.exists(filtering_step_path):
            continue

        # Counts the number of removed docs and reasons for the removal
        for language in os.listdir(filtering_step_path):
            language_path = os.path.join(filtering_step_path, language)
            if not os.path.exists(language_path):
                continue

            print(f"\t\t{language}...", flush=True)

            # Counts number of removed docs
            if language not in languages:
                languages[language] = 0

            # Iterate over all files in the directory
            for file_name in os.listdir(language_path):
                if file_name.endswith(".jsonl.gz"):
                    with gzip.open(os.path.join(language_path, file_name), 'rt', encoding='utf-8') as f:
                        for line in f:
                            doc = json.loads(line)
                            languages[language] += 1

    # Write JSON with the languages
    file_name = f"{collections[0]}_languages.json" if len(collections) == 1 else "languages.json"
    with open(os.path.join(output_path, file_name), 'w') as f:
        json.dump(languages, f, indent=4)


def filters_stats(data_path, output_path, collections):
    """
    This function generates statistics for the filtering steps applied to the dataset.
    It reads the number of documents in each filtering step and calculates the percentage of documents
    removed at each step.
    It also plots a bar chart for each filtering step showing the number of documents removes organized
    by filter_reason.
    """
    steps = {}

    for filtering_step in FILTERING_STEPS:
        print(f"\tProcessing {filtering_step}...", flush=True)
        filter_reasons = {"total": 0}

        for collection in (collections if collections else os.listdir(data_path)):
            filtering_step_path = os.path.join(data_path, collection, REMOVED_DATA_PATH, filtering_step)
            if not os.path.exists(filtering_step_path):
                continue

            print(f"\t\t{collection}...", flush=True)

            # Counts the number of removed docs and reasons for the removal
            num_docs = 0
            for file_name in os.listdir(filtering_step_path):
                if file_name.endswith(".jsonl.gz"):
                    with gzip.open(os.path.join(filtering_step_path, file_name), 'rt', encoding='utf-8') as f:
                        for line in f:
                            num_docs += 1
                            doc = json.loads(line)
                            reason = doc["metadata"]["filter_reason"]
                            if reason not in filter_reasons:
                                filter_reasons[reason] = 0
                            filter_reasons[reason] += 1

            filter_reasons["total"] += num_docs

        steps[filtering_step] = filter_reasons

    # Write JSON with the filter reasons
    file_name = f"{collections[0]}_filter_reasons.json" if len(collections) == 1 else "filter_reasons.json"
    with open(os.path.join(output_path, file_name), 'w') as f:
        json.dump(steps, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate filtering statistics from processed collections."
    )
    parser.add_argument("--data-path", required=True,
                        help="Path to the directory containing filtered collection data.")
    parser.add_argument("--output-path", required=True,
                        help="Path to the directory where output JSON stats will be written.")
    parser.add_argument("--collections", nargs="*", default=None,
                        help="List of collection names to process. If omitted, all subdirectories of --data-path are used.")
    args = parser.parse_args()

    collections = args.collections if args.collections else []

    os.makedirs(args.output_path, exist_ok=True)

    start_time = time.time()
    print("Stats script started.\n", flush=True)

    print("Generating filters stats...", flush=True)
    filters_stats(args.data_path, args.output_path, collections)
    print("Filters stats finished.\n", flush=True)

    print("Counting language filter...", flush=True)
    count_language_filter(args.data_path, args.output_path, collections)
    print("Counting language filter finished.\n", flush=True)

    print("Counting Portuguese language score...", flush=True)
    por_language_score(args.data_path, args.output_path, collections)
    print("Counting Portuguese language score finished.\n", flush=True)

    print(f"Stats script finished in {time.time() - start_time:.2f} seconds.", flush=True)