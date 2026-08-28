from trafilatura import extract
import mmap
import argparse
import orjson
import gzip
import os
import time
from tqdm import tqdm
from multiprocessing import Process, Manager

SEP = bytes("\n", 'utf-8')



def scrape_html(html_content):
    """
    Scrape html.
    :param html_content: html content in string format.
    :return: a dict containing the scraped information.
    """
    try:
        scraped_content = extract(html_content, output_format="json", with_metadata=True)
        json_content = orjson.loads(scraped_content)
        title = json_content.get('title')
        text = json_content.get('text')

        return {
            'text': text,
            'title': title,
            'metadata': {
                'url': None,
                'date': None,
                'file_path': None
            }
        }
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return {'text': None, 'title': None, 'metadata': {}}


def process_batch(html_paths, scraped_dir_path, batch_id):
    """
    Process the jsonl html files and write them to the respective scraped file.
    This function is called by each thread.
    """
    for html_file_path in tqdm(html_paths, total=len(html_paths), unit="files", desc=f"Batch {batch_id}",
                               position=batch_id):

        scraped_file_path = os.path.join(scraped_dir_path, os.path.basename(html_file_path)) + ".gz"

        if os.path.exists(scraped_file_path) or not html_file_path.endswith(".jsonl"):
            continue

        scraped_file = []

        with open(html_file_path, "r") as hf:
            with mmap.mmap(hf.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                for line in iter(mm.readline, b""):
                    json_content = orjson.loads(line)
                    url = json_content.get("url")
                    timestamp = json_content.get("timestamp")
                    html_content = json_content.get("html")

                    # Parse the html content
                    scraped_content = scrape_html(html_content)
                    if scraped_content.get('text'):
                        scraped_content['metadata'] = {
                            'url': url,
                            'date': timestamp,
                            'file_path': html_file_path
                        }
                        scraped_file.append(scraped_content)

        if scraped_file:
            with gzip.open(scraped_file_path, "wb") as jlf:
                for scraped in scraped_file:
                    jlf.write(orjson.dumps(scraped, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY))
                    jlf.write(SEP)


def process_html(html_path, scrape_path, num_workers=1):
    """
    Process all html files and generate json files with the scraped content.
    :param html_path: Path to the html json files directory.
    :param scrape_path: Path to save the scraped json files.
    :param num_workers: Number of threads to split the lines among.
    """

    # Iterate over the directories with jsonl
    for html_dir in os.listdir(html_path):
        html_dir_path = os.path.join(html_path, html_dir)

        if not os.path.isdir(html_dir_path) or not html_dir.startswith(COLLECTION):
            print(f"Skipped file {html_dir}.", flush=True)
            continue

        scraped_dir_path = os.path.join(scrape_path, html_dir)

        start_scraping_time = time.time()
        os.makedirs(scraped_dir_path, exist_ok=True)
        print(f"Scraping dir {html_dir}...", flush=True)

        # Get list of html files to process
        html_files = [os.path.join(html_dir_path, f) for f in os.listdir(html_dir_path)
                      if os.path.isfile(os.path.join(html_dir_path, f)) and f.endswith(".jsonl")]

        if not html_files:
            print(f"No files to process in {html_dir}.", flush=True)
            continue

        with Manager():
            total_files = len(html_files)
            files_per_worker = len(html_files) // num_workers
            threads = []

            for i in range(num_workers):
                start_idx = i * files_per_worker
                end_idx = start_idx + files_per_worker if i < num_workers - 1 else total_files
                batch = html_files[start_idx:end_idx]

                thread = Process(
                    target=process_batch,
                    args=(batch, scraped_dir_path, i)
                )
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

        print(f"Dir {html_dir} scraped successfully.", flush=True)
        print(f"Filtering time: {time.time() - start_scraping_time} seconds.\n", flush=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parallel scrape html files')
    parser.add_argument('files_path', metavar='filesPath', help='path to html files directory')
    parser.add_argument('-c', '--collection', dest='collection', required=True,
                        help='collection name prefix to filter input directories')
    parser.add_argument('-o', dest='outFolderPath', help='the path where to write the scraped html files to',
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

    print("\nScrapping HTML files...", flush=True)
    process_html(args.files_path, args.outFolderPath, args.collection, args.numWorkers)
    print("HTML files scrapped successfully.", flush=True)
