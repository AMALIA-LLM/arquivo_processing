import argparse

from datatrove.executor import SlurmPipelineExecutor, LocalPipelineExecutor
from datatrove.pipeline.readers import WarcReader
from datatrove.pipeline.extractors import TrafilaturaMetadata, Trafilatura
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.pipeline.filters import LambdaFilter
from datatrove.utils.logging import logger
from datatrove.pipeline.extractors.base import BaseExtractor


class EnhancedWarcReader(WarcReader):
    """ Custom WarcReader that logs errors and skips files that raise exceptions instead of stopping the task """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def read_file(self, filepath: str):
        from datatrove.utils.logging import logger
        try:
            yield from super().read_file(filepath)
        except Exception as e:
            logger.error(f"%%%%% Error processing file {filepath}: {e} %%%%%")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process WARC files: extract text with Trafilatura and filter Portuguese content."
    )

    # Required arguments
    parser.add_argument("--warc-dir", required=True,
                        help="Path to the directory containing WARC files.")
    parser.add_argument("--output-dir", required=True,
                        help="Path to the output directory for extracted JSONL files.")
    parser.add_argument("--logging-dir", required=True,
                        help="Path to the directory for pipeline logs.")

    # Executor arguments
    parser.add_argument("--job-name", default="warc-process",
                        help="Name for the executor job (default: warc-process).")
    parser.add_argument("--glob-pattern", default="*/*.arc.gz",
                        help="Glob pattern for WARC files (default: '*/*.arc.gz').")
    parser.add_argument("--tasks", type=int, default=1,
                        help="Number of tasks to run (default: 1).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of workers (default: 1).")
    parser.add_argument("--cpus-per-task", type=int, default=2,
                        help="Number of CPUs per task (default: 2).")
    parser.add_argument("--randomize-start-duration", type=int, default=60,
                        help="Max random delay (seconds) before task start (default: 60).")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Max number of documents to read per task (-1 for no limit, default: -1).")

    # Optional SLURM arguments (only needed when using a SLURM-based executor)
    slurm = parser.add_argument_group("SLURM options (only for SLURM-based executors)")
    slurm.add_argument("--time-limit", default=None,
                       help="SLURM time limit, e.g. '72:00:00'.")
    slurm.add_argument("--partition", default=None,
                       help="SLURM partition name.")
    slurm.add_argument("--mem-per-cpu-gb", type=int, default=None,
                       help="Memory per CPU in GB.")
    slurm.add_argument("--venv-path", default=None,
                       help="Path to the Python virtual-env activate script.")
    slurm.add_argument("--condaenv", default=None,
                       help="Name of the conda environment to activate.")
    slurm.add_argument("--account", default=None,
                       help="SLURM account for sbatch.")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    pipeline = [
        WarcReader(
            data_folder=args.warc_dir,
            glob_pattern=args.glob_pattern,
            limit=args.limit if args.limit > 0 else None,
        ),
        LambdaFilter(lambda doc: (doc_url := doc.metadata.get('url')) is not None and
                                 ('.pt/' in doc_url or '/pt/' in doc_url or '/pt-' in doc_url or
                                  '/pt_' in doc_url or doc_url.endswith('/pt') or doc_url.endswith('.pt'))
                                 and ('/pt-br' not in doc_url and '/pt_br/' not in doc_url)  # Exclude Brazilian Portuguese
                     ),
        TrafilaturaMetadata(favour_precision=True, timeout=30),
        JsonlWriter(args.output_dir),
    ]

    # Build optional SLURM keyword arguments
    executor_kwargs = {}
    if args.time_limit is not None:
        executor_kwargs["time"] = args.time_limit
    if args.partition is not None:
        executor_kwargs["partition"] = args.partition
    if args.mem_per_cpu_gb is not None:
        executor_kwargs["mem_per_cpu_gb"] = args.mem_per_cpu_gb
    if args.venv_path is not None:
        executor_kwargs["venv_path"] = args.venv_path
    if args.condaenv is not None:
        executor_kwargs["condaenv"] = args.condaenv
    if args.account is not None:
        executor_kwargs["sbatch_args"] = {"account": args.account}

    warc_processing_executor = SlurmPipelineExecutor(
        job_name=args.job_name,
        pipeline=pipeline,
        logging_dir=args.logging_dir,
        tasks=args.tasks,
        workers=args.workers,
        cpus_per_task=args.cpus_per_task,
        randomize_start_duration=args.randomize_start_duration,
        **executor_kwargs,
    )

    warc_processing_executor.run()
