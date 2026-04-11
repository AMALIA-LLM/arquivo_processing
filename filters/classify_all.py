import os
import json
import gzip
from pathlib import Path
import torch
import time
import argparse
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import multiprocessing as mp

LABELS = ["low", "medium", "high"]


class QualityClassifier:
    def __init__(self, model_path, device="cuda"):
        self.LABEL_MAP = {0: "low", 1: "medium", 2: "high"}

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True
        )

    def get_model_outputs(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to(
            self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            edu_score = outputs.educational_score
            depth_score = outputs.depth_score
            quality_labels = torch.argmax(outputs.quality_logits, dim=1)
        return edu_score.cpu(), depth_score.cpu(), quality_labels.cpu()


def process_gpu_worker(node_id, local_gpus, gpu_id, files_chunk, output_dirs, results_dict, collection, model_path, batch_size=1000):
    """Worker function for processing files on a specific GPU"""

    entries = []
    texts = []

    file_counters = {label: 1 for label in LABELS}
    quality_counts = {label: 0 for label in LABELS}

    for input_file in tqdm(files_chunk, desc=f"Node {node_id} GPU {gpu_id} -> Reading files",
                           position=node_id * local_gpus + gpu_id, total=len(files_chunk)):
        with gzip.open(input_file, 'rt', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line.strip())
                entries.append(entry)
                texts.append(entry['text'])

    device = f"cuda:{gpu_id}"
    classifier = QualityClassifier(model_path=model_path, device=device)

    write_data = {label: [] for label in LABELS}
    write_files = {label: None for label in LABELS}

    for i in tqdm(range(0, len(entries), batch_size), desc=f"Node {node_id} GPU {gpu_id} -> Processing files batches",
                  position=node_id * local_gpus + gpu_id + node_id * local_gpus, total=len(entries) // batch_size):
        chunk = entries[i:i + batch_size]
        texts_chunk = texts[i:i + batch_size]
        data, files = process_buffer_batch(classifier, chunk, texts_chunk, output_dirs, file_counters, quality_counts,
                                           gpu_id, node_id, collection)
        del chunk
        del texts_chunk

        for label in classifier.LABEL_MAP.values():
            if label in data:
                write_data[label].extend(data[label])
                if files[label] is not None:
                    write_files[label] = files[label]

        torch.cuda.empty_cache()

    del classifier
    del entries
    del texts

    for label, data in tqdm(write_data.items(), desc=f"Node {node_id} GPU {gpu_id} -> Writing files",
                            position=node_id * local_gpus + gpu_id + node_id * local_gpus * 2,
                            total=len(write_data)):
        if data and write_files[label]:
            with gzip.open(write_files[label], 'wt', encoding='utf-8') as f:
                f.write('\n'.join(json.dumps(item, ensure_ascii=False) for item in data) + '\n')

    results_dict[f"{node_id}_{gpu_id}"] = quality_counts


def process_buffer_batch(classifier, entries, texts, output_dirs, file_counters, quality_counts, gpu_id, node_id, collection):
    """Process batch with asynchronous file writing"""
    try:
        edu_scores, depth_scores, quality_labels = classifier.get_model_outputs(texts)
        binary_edu_preds = (edu_scores >= 2.5).int()

        output_data = {label: [] for label in classifier.LABEL_MAP.values()}
        output_files = {label: None for label in classifier.LABEL_MAP.values()}

        for i, entry in enumerate(entries):
            quality_label = classifier.LABEL_MAP[quality_labels[i].item()]

            entry['metadata']['quality_label'] = quality_label
            entry['metadata']['edu_score'] = float(edu_scores[i])
            entry['metadata']['depth_score'] = float(depth_scores[i])
            entry['metadata']['binary_edu_score'] = int(binary_edu_preds[i])

            output_data[quality_label].append(entry)
            quality_counts[quality_label] += 1

            if output_files[quality_label] is None and output_data[quality_label]:
                filename = f"{collection}_n{node_id}_gpu{gpu_id}_{file_counters[quality_label]:05d}.jsonl.gz"
                output_files[quality_label] = os.path.join(output_dirs[quality_label], filename)
                file_counters[quality_label] += 1

        return output_data, output_files

    except Exception as e:
        print(f"Error processing batch: {e}", flush=True)
        return {}, {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quality-classify deduplicated JSONL collections using a sequence classification model."
    )

    # Data path arguments
    parser.add_argument(
        "--jsonl-dir", type=str, required=True,
        help="Path to the directory containing deduplicated collections (each collection is a subdirectory with **/*.jsonl.gz files)"
    )
    parser.add_argument(
        "--output-path", type=str, required=True,
        help="Path to the directory where quality-classified output will be written"
    )
    parser.add_argument(
        "--logs-dir", type=str, required=True,
        help="Path to the directory where per-collection classification logs will be written"
    )

    # Model argument
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path or HuggingFace Hub name of the quality classifier model"
    )

    # Optional tuning arguments
    parser.add_argument(
        "--batch-size", type=int, default=1000,
        help="Number of documents per inference batch per GPU (default: 1000)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run inference on, e.g. 'cuda' or 'cpu' (default: cuda)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    start_time = time.time()

    # Get SLURM environment variables
    node_id = int(os.environ.get('SLURM_NODEID', 0))
    local_gpus = int(os.environ.get('SLURM_GPUS_ON_NODE', 4))
    node_count = int(os.environ.get('SLURM_NNODES', 1))

    # Create output directories
    output_dirs = {}
    for label in LABELS:
        dir_path = os.path.join(args.output_path, label)
        os.makedirs(dir_path, exist_ok=True)
        output_dirs[label] = dir_path

    print(f"Node ID: {node_id}, Local GPUs: {local_gpus}, Node Count: {node_count}", flush=True)

    # Cycle through the collections
    for collection in os.listdir(args.jsonl_dir):
        start_col_time = time.time()

        collection_jsonl = os.path.join(args.jsonl_dir, collection)
        collections_logs_dir = os.path.join(args.logs_dir, collection)
        os.makedirs(collections_logs_dir, exist_ok=True)
        collections_logs_file = os.path.join(args.logs_dir, collection, "quality_counts.json")

        if os.path.exists(collections_logs_file):
            print(f"Skipping {collection} as it was already classified.", flush=True)
            continue

        print(f"Processing collection: {collection}...", flush=True)

        # Find all .jsonl.gz files
        input_files = list(Path(collection_jsonl).glob('**/*.jsonl.gz'))

        # Determine this node's portion of files
        node_files = input_files[node_id::node_count]

        # Distribute files among local GPUs
        files_per_gpu = [node_files[i::local_gpus] for i in range(local_gpus)]

        # Create a Manager to share results between processes
        with mp.Manager() as manager:
            results_dict = manager.dict()

            processes = []
            for local_gpu_id in range(local_gpus):
                if files_per_gpu[local_gpu_id]:
                    p = mp.Process(
                        target=process_gpu_worker,
                        args=(node_id, local_gpus, local_gpu_id, files_per_gpu[local_gpu_id],
                              output_dirs, results_dict, collection, args.model_path, args.batch_size)
                    )
                    p.start()
                    processes.append(p)

            for p in processes:
                p.join()

            # Aggregate results
            total_counts = {"Node": node_id, "Count": {"low": 0, "medium": 0, "high": 0}}
            for counts in results_dict.values():
                for label in total_counts["Count"]:
                    total_counts["Count"][label] += counts[label]

            print(f"\nNode {node_id} Total counts: {total_counts}", flush=True)

            # Save the per-collection counts to a JSON file
            with open(collections_logs_file, 'w') as f:
                json.dump(total_counts, f, indent=4)

            print(f"Collection {collection} processed in {time.time() - start_col_time:.2f} seconds.", flush=True)

    print("All collections processed.", flush=True)
    print(f"Total processing time: {time.time() - start_time:.2f} seconds.", flush=True)


if __name__ == "__main__":
    main()