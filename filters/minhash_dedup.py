import argparse

from datatrove.pipeline.dedup import MinhashDedupSignature
from datatrove.pipeline.base import PipelineStep
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.dedup.minhash import (
    MinhashConfig,
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupFilter,
)
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.hashing import HashConfig
from datatrove.utils.typeshelper import Languages
from datatrove.executor import MareNostrumExecutor


class Rehydrater(PipelineStep):
    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        import bisect
        upsampling_weights = {1: 1, 2: 2, 3: 3, 5: 5, 100: 8, 1000: 1}
        limits = sorted(upsampling_weights.keys())

        for doc in data:
            upsampling_weight = upsampling_weights[
                limits[bisect.bisect_right(limits, doc.metadata["minhash_cluster_size"]) - 1]]
            doc.metadata["upsampling_weight"] = upsampling_weight
            yield doc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Minhash deduplication pipeline for filtered JSONL data using datatrove."
    )

    # Required arguments
    parser.add_argument("--input-dir", required=True,
                        help="Path to the filtered JSONL input directory.")
    parser.add_argument("--output-dir", required=True,
                        help="Path to the deduplication output directory.")
    parser.add_argument("--logging-dir", required=True,
                        help="Path to the directory for pipeline logs.")

    # Executor arguments
    parser.add_argument("--job-name", default="mh-dedup",
                        help="Base name for executor jobs (default: mh-dedup).")
    parser.add_argument("--tasks", type=int, default=100,
                        help="Number of tasks to run (default: 100).")
    parser.add_argument("--workers", type=int, default=100,
                        help="Number of workers (default: 100).")
    parser.add_argument("--cpus-per-task", type=int, default=8,
                        help="Number of CPUs per task (default: 8).")
    parser.add_argument("--randomize-start-duration", type=int, default=180,
                        help="Max random delay (seconds) before task start (default: 180).")

    # Optional SLURM arguments (only needed when using a SLURM-based executor)
    slurm = parser.add_argument_group("SLURM options (only for SLURM-based executors)")
    slurm.add_argument("--time-limit", default=None,
                       help="SLURM time limit, e.g. '02:00:00'.")
    slurm.add_argument("--qos", default=None,
                       help="SLURM QOS value.")
    slurm.add_argument("--partition", default=None,
                       help="SLURM partition name.")
    slurm.add_argument("--venv-path", default=None,
                       help="Path to the Python virtual-env activate script.")
    slurm.add_argument("--account", default=None,
                       help="SLURM account for sbatch.")

    return parser.parse_args()


def build_slurm_kwargs(args):
    """Build optional SLURM keyword arguments from parsed args."""
    kwargs = {}
    if args.time_limit is not None:
        kwargs["time"] = args.time_limit
    if args.qos is not None:
        kwargs["qos"] = args.qos
    if args.partition is not None:
        kwargs["partition"] = args.partition
    if args.venv_path is not None:
        kwargs["venv_path"] = args.venv_path
    if args.account is not None:
        kwargs["sbatch_args"] = {"account": args.account}
    return kwargs


if __name__ == '__main__':
    args = parse_args()
    slurm_kwargs = build_slurm_kwargs(args)

    # Input reader for .jsonl.gz files
    input_reader = JsonlReader(
        data_folder=args.input_dir,
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
        job_name=f"{args.job_name}-1sig",
        pipeline=[
            input_reader,
            MinhashDedupSignature(
                output_folder=f"{args.output_dir}/steps/1_signatures",
                config=minhash_config,
                language=Languages.portuguese__latn
            ),
        ],
        tasks=args.tasks,
        workers=args.workers,
        cpus_per_task=args.cpus_per_task,
        logging_dir=f"{args.logging_dir}/1_signatures",
        slurm_logs_folder=f"{args.logging_dir}/1_signatures/slurm_logs",
        randomize_start_duration=args.randomize_start_duration,
        **slurm_kwargs,
    )

    # Stage 2: Find matches between signatures in each bucket
    stage2 = MareNostrumExecutor(
        job_name=f"{args.job_name}-2bkt",
        pipeline=[
            MinhashDedupBuckets(
                input_folder=f"{args.output_dir}/steps/1_signatures",
                output_folder=f"{args.output_dir}/steps/2_buckets",
                config=minhash_config,
            ),
        ],
        tasks=minhash_config.num_buckets,
        workers=args.workers,
        cpus_per_task=args.cpus_per_task,
        randomize_start_duration=args.randomize_start_duration,
        logging_dir=f"{args.logging_dir}/2_buckets",
        slurm_logs_folder=f"{args.logging_dir}/2_buckets/slurm_logs",
        depends=stage1,
        **slurm_kwargs,
    )

    # Stage 3: Create clusters of duplicates
    stage3 = MareNostrumExecutor(
        job_name=f"{args.job_name}-3cls",
        pipeline=[
            MinhashDedupCluster(
                input_folder=f"{args.output_dir}/steps/2_buckets",
                output_folder=f"{args.output_dir}/steps/3_clusters",
                config=minhash_config,
                save_cluster_size=True,
                save_cluster_id=True
            ),
        ],
        tasks=1,
        cpus_per_task=args.cpus_per_task,
        logging_dir=f"{args.logging_dir}/3_clusters",
        slurm_logs_folder=f"{args.logging_dir}/3_clusters/slurm_logs",
        depends=stage2,
        **slurm_kwargs,
    )

    # Stage 4: Remove duplicates
    stage4 = MareNostrumExecutor(
        job_name=f"{args.job_name}-4flt",
        pipeline=[
            input_reader,
            MinhashDedupFilter(
                input_folder=f"{args.output_dir}/steps/3_clusters",
                exclusion_writer=JsonlWriter(f"{args.output_dir}/removed"),
                load_cluster_ids=True,
                load_cluster_sizes=True
            ),
            Rehydrater(),
            TokensCounter(),
            JsonlWriter(output_folder=f"{args.output_dir}/output"),
        ],
        tasks=args.tasks,
        workers=args.workers,
        cpus_per_task=args.cpus_per_task,
        logging_dir=f"{args.logging_dir}/final_output",
        slurm_logs_folder=f"{args.logging_dir}/final_output/slurm_logs",
        depends=stage3,
        **slurm_kwargs,
    )

    # Execute the pipeline
    stage4.run()
