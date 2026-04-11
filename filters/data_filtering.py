import argparse

from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.formatters import PIIFormatter, FTFYFormatter, SymbolLinesFormatter
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.filters import (
    FineWebQualityFilter,
    GopherQualityFilter,
    GopherRepetitionFilter,
    LanguageFilter,
)
from datatrove.utils.typeshelper import Languages
from datatrove.executor import MareNostrumExecutor
from datatrove.pipeline.post_scraping import PostScraper
import words


def build_pipeline(input_dir, output_dir):
    """Build the data filtering pipeline.

    All the custom filter values were taken from the FineWeb-2 dataset filters
    for the Portuguese language.
    """
    return [
        JsonlReader(
            data_folder=input_dir,
            glob_pattern="**/*.jsonl.gz",
        ),
        PostScraper(
            exclusion_writer=JsonlWriter(f"{output_dir}/removed/0_post_scraper"),
            remove_dup_lines=True,
            remove_short_lines=True,
        ),
        LanguageFilter(
            languages=[Languages.portuguese__latn],
            language_threshold=0.799,
            backend="glotlid",
            exclusion_writer=JsonlWriter(
                f"{output_dir}/removed/1_lang_exclusion",
                output_filename="${language}/${rank}.jsonl.gz"
            )
        ),
        GopherRepetitionFilter(
            exclusion_writer=JsonlWriter(f"{output_dir}/removed/2_gopher_rep"),
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
            exclusion_writer=JsonlWriter(f"{output_dir}/removed/3_fineweb_qual"),
            language=Languages.portuguese__latn,
            new_line_ratio=0.186,
            line_punct_thr=0.077,
            char_duplicates_ratio=0.1,
            short_line_thr=999  # Disabled this filter
        ),
        GopherQualityFilter(
            exclusion_writer=JsonlWriter(f"{output_dir}/removed/4_gopher_qual"),
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
        JsonlWriter(f"{output_dir}/output"),
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Data filtering pipeline for Portuguese web text using datatrove."
    )

    # Required arguments
    parser.add_argument("--input-dir", required=True,
                        help="Path to the input directory containing JSONL files.")
    parser.add_argument("--output-dir", required=True,
                        help="Path to the output directory for filtered data.")
    parser.add_argument("--logging-dir", required=True,
                        help="Path to the directory for pipeline logs.")

    # Executor arguments
    parser.add_argument("--job-name", default="data-filter",
                        help="Name for the executor job (default: data-filter).")
    parser.add_argument("--tasks", type=int, default=50,
                        help="Number of tasks to run (default: 50).")
    parser.add_argument("--workers", type=int, default=16,
                        help="Number of workers / CPUs to use (default: 16).")
    parser.add_argument("--cpus-per-task", type=int, default=2,
                        help="Number of CPUs per task (default: 2).")
    parser.add_argument("--randomize-start-duration", type=int, default=180,
                        help="Max random delay (seconds) before task start (default: 180).")

    # Optional SLURM arguments (only needed when using a SLURM-based executor)
    slurm = parser.add_argument_group("SLURM options (only for SLURM-based executors)")
    slurm.add_argument("--time-limit", default=None,
                       help="SLURM time limit, e.g. '72:00:00'.")
    slurm.add_argument("--qos", default=None,
                        help="SLURM QOS value.")
    slurm.add_argument("--partition", default=None,
                        help="SLURM partition name.")
    slurm.add_argument("--venv-path", default=None,
                        help="Path to the Python virtual-env activate script.")
    slurm.add_argument("--account", default=None,
                        help="SLURM account for sbatch.")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    pipeline = build_pipeline(args.input_dir, args.output_dir)

    # Build optional SLURM keyword arguments
    executor_kwargs = {}
    if args.time_limit is not None:
        executor_kwargs["time"] = args.time_limit
    if args.qos is not None:
        executor_kwargs["qos"] = args.qos
    if args.partition is not None:
        executor_kwargs["partition"] = args.partition
    if args.venv_path is not None:
        executor_kwargs["venv_path"] = args.venv_path
    if args.account is not None:
        executor_kwargs["sbatch_args"] = {"account": args.account}

    slurm_logs = f"{args.logging_dir}/slurm_logs"

    data_filtering_executor = MareNostrumExecutor(
        job_name=args.job_name,
        pipeline=pipeline,
        tasks=args.tasks,
        workers=args.workers,
        cpus_per_task=args.cpus_per_task,
        randomize_start_duration=args.randomize_start_duration,
        logging_dir=args.logging_dir,
        slurm_logs_folder=slurm_logs,
        **executor_kwargs,
    )

    data_filtering_executor.run()
