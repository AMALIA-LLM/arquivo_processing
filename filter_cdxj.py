import argparse
import orjson
import os
import subprocess
import time
import threading
from tqdm import tqdm
from multiprocessing import Manager
from utils.utils import hash_value

BATCH_SIZE = 200


def get_pt_indexes(filtered_lines):
    """
    Get the initial and end indexes of the lines with pt domains.
    If the file has scattered pt domains, the end index will be the last line with a pt domain.
    """
    start_idx = -1
    end_idx = -1

    for idx, line in enumerate(filtered_lines):
        parts = line.split(" ", 2)
        surt = parts[0]
        json_content = orjson.loads(parts[2])
        url = json_content.get("url")
        if start_idx == -1 and ('.pt/' in url or '/pt/' in url or '/pt-' in url or '/pt_' in url or url.endswith('/pt') or surt.startswith("pt,")):
            start_idx = idx
        elif start_idx != -1 and end_idx == -1 and not ('.pt/' in url or '/pt/' in url or '/pt-' in url or '/pt_' in url or url.endswith('/pt') or surt.startswith("pt,")):
            end_idx = idx
        elif start_idx != -1 and end_idx != -1 and ('.pt/' in url or '/pt/' in url or '/pt-' in url or '/pt_' in url or url.endswith('/pt') or surt.startswith("pt,")):
            end_idx = -1

    if start_idx != -1 and end_idx == -1:
        end_idx = len(filtered_lines) - 1

    return start_idx, end_idx


def process_lines(filtered_lines, filtered_cdxj_path, shared_data, thread_id):
    """
    Process the filtered lines and write them to the filtered CDXJ file and update the db.
    This function is called by each thread.
    """
    filtered_batch = []
    check_batch = set()

    for line in tqdm(filtered_lines, total=len(filtered_lines), unit="lines",
                  smoothing=0, desc=f"Thread #{thread_id}"):

        # Line is {surt} {timestamp} {json_content}
        parts = line.split(" ", 2)
        surt = parts[0]
        timestamp = parts[1]
        json_content = orjson.loads(parts[2])
        url = json_content.get("url")

        # Write line if the URL contains 'pt'
        if '.pt/' in url or '/pt/' in url or '/pt-' in url or '/pt_' in url or url.endswith('/pt') or surt.startswith(
                "pt,"):
            surt_hash = hash_value(surt)

            if (surt_hash, timestamp) in check_batch:
                continue

            filtered_batch.append(line)
            check_batch.add((surt_hash, timestamp))

            if len(filtered_batch) >= BATCH_SIZE:
                with shared_data['write_lock']:
                    with open(filtered_cdxj_path, "a") as writer:
                        for filtered_line in filtered_batch:
                            writer.write(filtered_line + "\n")
                    filtered_batch.clear()
                    check_batch.clear()

    if len(filtered_batch) > 0:
        with shared_data['write_lock']:
            with open(filtered_cdxj_path, "a") as writer:
                for filtered_line in filtered_batch:
                    writer.write(filtered_line + "\n")


def process_cdxj(cdxj_path, filtered_path, num_workers=1):
    """
    Reads all CDXJ files in the path that haven't been filtered.
    Filters the lines with status 200, mime text/html and a pt domain.
    Saves the filtered CDXJ file.
    :param cdxj_path: Path to the CDXJ files directory
    :param filtered_path: Path to save the filtered CDXJ files
    :param num_workers: Number of threads to split the lines among
    """

    # Iterate over the CDXJ files
    for cdxj_file in os.listdir(cdxj_path):
        if not cdxj_file.endswith(".cdxj") or not cdxj_file.startswith(COLLECTION):
            print(f"Skipped file {cdxj_file}.", flush=True)
            continue

        cdxj_file_path = os.path.join(cdxj_path, cdxj_file)
        filtered_cdxj_path = os.path.join(filtered_path, "filtered_" + cdxj_file)

        # Check if the file has already been processed
        if os.path.exists(filtered_cdxj_path):
            print(f"Filtered file {cdxj_file} already exists. Skipping...", flush=True)
            continue

        start_filtering_time = time.time()

        # Use grep to filter the CDXJ file for lines with status 200 and mime text/html
        grep_command = f"cat {cdxj_file_path} | grep '\"status\": \"200\"' | grep '\"mime\": \"text/html\"'"
        filtered_lines = subprocess.check_output(grep_command, shell=True).decode('utf-8').splitlines()

        # Get the initial and end indexes of the lines with pt domains
        start_idx, end_idx = get_pt_indexes(filtered_lines)
        if start_idx == -1:
            print(f"No pt domains found in {cdxj_file}. Skipping...", flush=True)
            continue

        print(f"Processing {cdxj_file} with {len(filtered_lines)} lines. Start {start_idx} ; End {end_idx}", flush=True)
        filtered_lines = filtered_lines[start_idx:end_idx + 1]

        with Manager() as manager:
            # Initialize shared data structure
            shared_data = {
                'write_lock': manager.Lock()
            }

            # Split lines among threads
            total_lines = len(filtered_lines)
            lines_per_thread = total_lines // num_workers
            threads = []

            # Create and start threads
            for i in range(num_workers):
                start_idx = i * lines_per_thread
                end_idx = start_idx + lines_per_thread if i < num_workers - 1 else total_lines
                thread_lines = filtered_lines[start_idx:end_idx]

                thread = threading.Thread(
                    target=process_lines,
                    args=(thread_lines, filtered_cdxj_path, shared_data, i)
                )
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

        print(f"File {cdxj_file} processed successfully. Created filtered_{cdxj_file}.", flush=True)
        print(f"Filtering time: {time.time() - start_filtering_time} seconds.", flush=True)
        print("", flush=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parallel process cdxj files')
    parser.add_argument('files_path', metavar='filesPath', help='path to cdxj files directory')
    parser.add_argument('-c', '--collection', dest='collection', required=True,
                        help='collection name prefix to filter input files')
    parser.add_argument('-o', dest='outFolderPath', help='the path where to write the filtered cdxj files to',
                        required=True)
    parser.add_argument('-w', dest='numWorkers', help='number of workers to split each file amongst', default=1,
                        type=int)

    args = parser.parse_args()

    if not os.path.isdir(args.files_path):
        print(f"The given cdxj folder doesn't exist, create it and rerun. - {args.files_path}")
        exit()

    if not os.path.isdir(args.outFolderPath):
        print(f"The folder given to -o doesn't exist, create it and rerun. - {args.outFolderPath}")
        exit()

    print("Starting to process CDXJ files...", flush=True)
    process_cdxj(args.files_path, args.outFolderPath, args.collection, args.numWorkers)
    print("Finished processing CDXJ files.", flush=True)
