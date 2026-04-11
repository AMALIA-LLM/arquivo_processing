from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.formatters import PIIFormatter, FTFYFormatter, SymbolLinesFormatter
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.pipeline.tokens import TokensCounter
from datatrove.utils.typeshelper import Languages
from datatrove.executor import MareNostrumExecutor
from datatrove.pipeline.post_scraping import PostScraper
from datatrove.pipeline.dedup import MinhashDedupSignature
from datatrove.pipeline.base import PipelineStep
from datatrove.data import DocumentsPipeline
from datatrove.utils.hashing import HashConfig
from datatrove.pipeline.filters import (
    FineWebQualityFilter,
    GopherQualityFilter,
    GopherRepetitionFilter,
    LanguageFilter,
)
from datatrove.pipeline.dedup.minhash import (
    MinhashConfig,
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupFilter,
)
import words
import argparse
import subprocess
import time
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter and deduplicate JSONL collections using datatrove on a SLURM cluster."
    )

    # Cluster / environment arguments (optional, only for SLURM-based executors)
    slurm = parser.add_argument_group("SLURM options (only for SLURM-based executors)")
    slurm.add_argument(
        "--qos", type=str, default=None,
        help="SLURM Quality of Service"
    )
    slurm.add_argument(
        "--partition", type=str, default=None,
        help="SLURM partition to submit jobs to"
    )
    slurm.add_argument(
        "--account", type=str, default=None,
        help="SLURM account/project to charge compute time to"
    )
    slurm.add_argument(
        "--venv-path", type=str, default=None,
        help="Path to the virtual environment activate script (e.g. /path/to/venv/bin/activate)"
    )

    # Data path arguments
    parser.add_argument(
        "--collections-path", type=str, required=True,
        help="Path to the directory containing the input collections"
    )
    parser.add_argument(
        "--output-path", type=str, required=True,
        help="Path to the directory where filtered/deduplicated output will be written"
    )
    parser.add_argument(
        "--logs-path", type=str, required=True,
        help="Path to the directory where logs will be written"
    )

    # Optional overrides for filtering job configuration
    # Defaults are derived from --qos if not provided
    parser.add_argument(
        "--filter-tasks", type=int, default=300,
        help="Number of filtering tasks (default: 300)"
    )
    parser.add_argument(
        "--filter-workers", type=int, default=128,
        help="Number of filtering workers/CPUs (default: 128)"
    )
    parser.add_argument(
        "--filter-cpus-per-task", type=int, default=6,
        help="CPUs per filtering task (default: 6)"
    )
    parser.add_argument(
        "--filter-time-limit", type=str, default="24:00:00",
        help="Time limit for filtering jobs in HH:MM:SS (default: 24:00:00)"
    )

    # Optional overrides for deduplication job configuration
    # Defaults are derived from --qos if not provided
    parser.add_argument(
        "--dedup-tasks", type=int, default=100,
        help="Number of deduplication tasks (default: 100)"
    )
    parser.add_argument(
        "--dedup-workers", type=int, default=100,
        help="Number of deduplication workers/CPUs (default: 100)"
    )
    parser.add_argument(
        "--dedup-cpus-per-task", type=int, default=8,
        help="CPUs per deduplication task (default: 8)"
    )

    return parser.parse_args()


def build_slurm_kwargs(cfg):
    """Build optional SLURM keyword arguments from parsed config."""
    kwargs = {}
    if cfg.qos is not None:
        kwargs["qos"] = cfg.qos
    if cfg.partition is not None:
        kwargs["partition"] = cfg.partition
    if cfg.venv_path is not None:
        kwargs["venv_path"] = cfg.venv_path
    if cfg.account is not None:
        kwargs["sbatch_args"] = {"account": cfg.account}
    return kwargs


# Rehydrater class to rehydrate documents based on their minhash cluster size
class Rehydrater(PipelineStep):
    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        import bisect
        upsampling_weights = {1: 1, 2: 2, 3: 3, 5: 5, 100: 8, 1000: 1}
        # Sorted keys
        limits = sorted(upsampling_weights.keys())

        for doc in data:
            upsampling_weight = upsampling_weights[
                limits[bisect.bisect_right(limits, doc.metadata["minhash_cluster_size"]) - 1]]
            doc.metadata["upsampling_weight"] = upsampling_weight
            yield doc
            # repeat each document upsampling_weight times
            # for _ in range(upsampling_weight):
            #     yield doc


def submit_filtering_job(collection, cfg):
    collection_job_id = collection.split("AWP")[0].lower() + collection.split("AWP")[
        1] if "AWP" in collection else collection[0:3].lower()

    job_name = f"{collection_job_id}-filter"

    # Cluster paths
    jsonl_dir = f"{cfg.collections_path}/{collection}"
    filtering_output_path = f"{cfg.output_path}/{collection}/1_filter"
    logs_dir = f"{cfg.logs_path}/{collection}/1_filter"

    slurm_kwargs = build_slurm_kwargs(cfg)

    # All the custom values were taken from the FineWeb-2 dataset filters for the Portuguese language
    data_filtering_executor = MareNostrumExecutor(
        job_name=job_name,
        pipeline=[
            JsonlReader(
                data_folder=f"{jsonl_dir}",
                glob_pattern="**/*.jsonl.gz",
            ),
            PostScraper(
                exclusion_writer=JsonlWriter(f"{filtering_output_path}/removed/0_post_scraper"),
                remove_dup_lines=True,
                remove_short_lines=True,
            ),
            LanguageFilter(
                languages=[Languages.portuguese__latn],
                language_threshold=0.799,
                backend="glotlid",
                exclusion_writer=JsonlWriter(
                    f"{filtering_output_path}/removed/1_lang_exclusion",
                    output_filename="${language}/${rank}.jsonl.gz"
                )
            ),
            GopherRepetitionFilter(
                exclusion_writer=JsonlWriter(f"{filtering_output_path}/removed/2_gopher_rep"),
                language=Languages.portuguese__latn,
                dup_para_frac=None,
                dup_line_char_frac=None,
                dup_para_char_frac=None,
                dup_line_frac=0.287,
                top_n_grams=((2, 0.371), (3, 0.191), (4, 0.163)),
                dup_n_grams=((5, 0.163), (6, 0.153), (7, 0.141), (8, 0.13),
                             (9, 0.119), (10, 0.108))
            ),
            FineWebQualityFilter(
                exclusion_writer=JsonlWriter(f"{filtering_output_path}/removed/3_fineweb_qual"),
                language=Languages.portuguese__latn,
                new_line_ratio=0.186,
                line_punct_thr=0.077,
                char_duplicates_ratio=0.1,
                short_line_thr=999  # Disabled this filter
            ),
            GopherQualityFilter(
                exclusion_writer=JsonlWriter(f"{filtering_output_path}/removed/4_gopher_qual"),
                language=Languages.portuguese__latn,
                min_avg_word_length=3,
                max_avg_word_length=13,
                min_doc_words=15,
                max_non_alpha_words_ratio=0.814,
                max_ellipsis_lines_ratio=0.3,
                stop_words=words.get_stop_words_fineweb2(),
                min_stop_words=2
            ),
            FTFYFormatter(),  # fix encoding issues. Important in a multilingual setting
            PIIFormatter(),  # remove PII (emails, ips, etc)
            SymbolLinesFormatter(symbols_to_remove=["|"], replace_char="\n"),  # fix trafilatura table artifacts
            TokensCounter(),
            JsonlWriter(f"{filtering_output_path}/output"),
        ],
        tasks=cfg.filter_tasks,
        workers=cfg.filter_workers,
        cpus_per_task=cfg.filter_cpus_per_task,
        randomize_start_duration=180,
        logging_dir=f"{logs_dir}",
        slurm_logs_folder=f"{logs_dir}/slurm_logs",
        time=cfg.filter_time_limit,
        **slurm_kwargs,
    )

    data_filtering_executor.run()

    return job_name


def submit_deduplication_job(collection, cfg):
    collection_job_id = collection.split("AWP")[0].lower() + collection.split("AWP")[
        1] if "AWP" in collection else collection[0:3].lower()

    filtered_jsonl_dir = f"{cfg.output_path}/{collection}/1_filter/output"
    dedup_output_path = f"{cfg.output_path}/{collection}/2_dedup"
    logs_dir = f"{cfg.logs_path}/{collection}/2_dedup"

    slurm_kwargs = build_slurm_kwargs(cfg)

    # Input reader for .jsonl.gz files
    INPUT_READER = JsonlReader(
        data_folder=filtered_jsonl_dir,
        glob_pattern="*.jsonl.gz"
    )

    # Configure Minhash
    minhash_config = MinhashConfig(
        hash_config=HashConfig(
            hash_fc="xxhash",
            precision=64,  # better precision -> fewer false positives (collisions)
        ),
        num_buckets=14,
        hashes_per_bucket=8,
        n_grams=5,
    )

    # Stage 1: Compute minhash signatures
    stage1 = MareNostrumExecutor(
        job_name=f"mh1-{collection_job_id}",
        pipeline=[
            INPUT_READER,
            MinhashDedupSignature(
                output_folder=f"{dedup_output_path}/steps/1_signatures",
                config=minhash_config,
                language=Languages.portuguese__latn
            ),
        ],
        tasks=cfg.dedup_tasks,
        workers=cfg.dedup_workers,
        cpus_per_task=cfg.dedup_cpus_per_task,
        time="10:00:00",
        logging_dir=f"{logs_dir}/1_signatures",
        slurm_logs_folder=f"{logs_dir}/1_signatures/slurm_logs",
        randomize_start_duration=180,
        **slurm_kwargs,
    )

    # Stage 2: Find matches between signatures in each bucket
    stage2 = MareNostrumExecutor(
        job_name=f"mh2-{collection_job_id}",
        pipeline=[
            MinhashDedupBuckets(
                input_folder=f"{dedup_output_path}/steps/1_signatures",
                output_folder=f"{dedup_output_path}/steps/2_buckets",
                config=minhash_config
            ),
        ],
        tasks=minhash_config.num_buckets,
        workers=cfg.dedup_workers,
        cpus_per_task=cfg.dedup_cpus_per_task,
        time="10:00:00",
        randomize_start_duration=180,
        logging_dir=f"{logs_dir}/2_buckets",
        slurm_logs_folder=f"{logs_dir}/2_buckets/slurm_logs",
        depends=stage1,
        **slurm_kwargs,
    )

    # Stage 3: Create clusters of duplicates
    stage3 = MareNostrumExecutor(
        job_name=f"mh3-{collection_job_id}",
        pipeline=[
            MinhashDedupCluster(
                input_folder=f"{dedup_output_path}/steps/2_buckets",
                output_folder=f"{dedup_output_path}/steps/3_clusters",
                config=minhash_config,
                save_cluster_size=True,
                save_cluster_id=True
            ),
        ],
        tasks=1,
        cpus_per_task=cfg.dedup_cpus_per_task,
        time="10:00:00",
        logging_dir=f"{logs_dir}/3_clusters",
        slurm_logs_folder=f"{logs_dir}/3_clusters/slurm_logs",
        depends=stage2,
        **slurm_kwargs,
    )

    # Stage 4: Remove duplicates
    stage4 = MareNostrumExecutor(
        job_name=f"mh4-{collection_job_id}",
        pipeline=[
            INPUT_READER,
            MinhashDedupFilter(
                input_folder=f"{dedup_output_path}/steps/3_clusters",
                exclusion_writer=JsonlWriter(f"{dedup_output_path}/removed"),
                load_cluster_ids=True,
                load_cluster_sizes=True
            ),
            Rehydrater(),
            TokensCounter(),
            JsonlWriter(output_folder=f"{dedup_output_path}/output"),
        ],
        tasks=cfg.dedup_tasks,
        workers=cfg.dedup_workers,
        cpus_per_task=cfg.dedup_cpus_per_task,
        time="10:00:00",
        logging_dir=f"{logs_dir}/final_output",
        slurm_logs_folder=f"{logs_dir}/final_output/slurm_logs",
        depends=stage3,
        **slurm_kwargs,
    )

    # Execute the pipeline
    stage4.run()

    return f"mh4-{collection_job_id}"


def process_collections(collections, cfg):
    for collection in collections:
        start_col = time.time()
        job_name = submit_filtering_job(collection, cfg)
        print(f"\nFiltering Job {job_name} submitted for {collection}", flush=True)

        # Wait for filtering job completion before submitting next
        time.sleep(60)
        while is_job_running(job_name):
            time.sleep(120)  # Check every 2 minutes

        print(f"Filtering Job {job_name} completed for {collection} in {time.time() - start_col:.2f} seconds.", flush=True)

        start_dedup = time.time()
        job_name = submit_deduplication_job(collection, cfg)
        print(f"Deduplication Job {job_name} submitted for {collection}", flush=True)

        # Wait for deduplication job completion before processing next collection
        time.sleep(60)
        while is_job_running(job_name):
            time.sleep(120)  # Check every 2 minutes

        print(f"Deduplication Job {job_name} completed for {collection} in {time.time() - start_dedup:.2f} seconds.", flush=True)
        print(f"Collection {collection} processed in {time.time() - start_col:.2f} seconds.", flush=True)


def is_job_running(job_name):
    """Check if SLURM job is still running"""
    result = subprocess.run(['squeue', '-n', job_name],
                            capture_output=True, text=True)
    return job_name[0:6] in result.stdout


if __name__ == "__main__":
    cfg = parse_args()

    start_time = time.time()
    collections = os.listdir(cfg.collections_path)

    # Filter out collections already processed
    for collection in collections:
        if os.path.exists(f"{cfg.output_path}/{collection}/1_filter/output"):
            print(f"Collection {collection} already processed, skipping.", flush=True)
            collections.remove(collection)

    if not collections:
        print("No collections to process.")
        exit()

    print(f"Found {len(collections)} collections to process:\n{collections}", flush=True)

    process_collections(collections, cfg)

    print(f"\nAll {len(collections)} collections processed in {time.time() - start_time:.2f} seconds.", flush=True)