import argparse
import random
import time
import os
import orjson
from tqdm import tqdm
import requests
import threading
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import concurrent.futures


# The ArquivoPT API has a limit of 4000 requests/minute
MAX_REQUESTS_PER_MINUTE = int(4000 * 0.99)
BATCH_SIZE = min(1000, MAX_REQUESTS_PER_MINUTE // 4)
DELAY_BETWEEN_REQUESTS = 60 / MAX_REQUESTS_PER_MINUTE
DELAY_BETWEEN_BATCHES = (BATCH_SIZE / MAX_REQUESTS_PER_MINUTE) * 60

class CustomRetry(Retry):
    def get_backoff_time(self):
        backoff_time = super().get_backoff_time()
        if hasattr(self, 'status') and self.status:
            if self.status == 429:
                backoff_time = 60  # Wait 1 minute
                print(f"Rate limit exceeded. Waiting {backoff_time} seconds...", flush=True)
        elif backoff_time:
            backoff_time += random.uniform(0, 1) # Small jitter for other retries
        return backoff_time

    def increment(self, *args, **kwargs):
        try:
            response = kwargs.get('response')
            if response is not None and response.status == 429:
                self.status = response.status
                print(f"Rate limit exceeded (429). Setting flag for custom backoff.", flush=True)
            else:
                self.status = None  # Reset for other retries

            # Now call the original increment, which will use get_backoff_time()
            return super().increment(*args, **kwargs)

        except Exception as e:
            # Handle cases where increment might fail or for logging
            print(f"Error in CustomRetry.increment: {e}", flush=True)
            raise  # Re-raise the exception


def setup_session(num_workers):
    session = requests.Session()
    retry = CustomRetry(
        total=5,
        backoff_factor=DELAY_BETWEEN_REQUESTS,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    pool_size = max(min(num_workers * 8, 150), 100)
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=pool_size,
        pool_maxsize=pool_size
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


def process_url(url, timestamp, session):
    """Process a single URL and return the results"""
    html_response = request_html(url, timestamp, session)

    if not html_response or not html_response.text:
        return {"status": "empty"}

    status_code = html_response.status_code
    size = len(html_response.content) * 8  # Convert size to bits

    if status_code != 200:
        return {
            "status": html_response.status_code,
            "size": size
        }

    return {
        "status": html_response.status_code,
        "html": html_response.text,
        "size": size
    }


def process_batch(urls, html_json_file, num_workers):
    """Process a batch of URLs concurrently"""
    response_counts = {}
    results_buffer = []
    total_size = 0

    start_time = time.time()

    # Set up the requests session
    session = setup_session(num_workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_url = {
            executor.submit(
                process_url, url, timestamp, session
            ): (url, timestamp)
            for (surt, timestamp, url) in urls
        }

        for future in concurrent.futures.as_completed(future_to_url):
            try:
                result = future.result(timeout=120)
                status_code = result["status"]
                response_counts[status_code] = response_counts.get(status_code, 0) + 1
                url, timestamp = future_to_url[future]
                if status_code == 200:
                    results_buffer.append(orjson.dumps({
                        "url": url,
                        "timestamp": timestamp,
                        "html": result["html"]
                    }).decode("utf-8"))
                total_size += result["size"]
            except concurrent.futures.TimeoutError:
                response_counts["timeout"] = response_counts.get("timeout", 0) + 1
            except Exception as e:
                response_counts["error"] = response_counts.get("error", 0) + 1

    end_time = time.time()
    total_access_time = end_time - start_time
    download_rate = total_size / total_access_time if total_access_time > 0 else 0

    print(f"Batch processed in {total_access_time:.2f} seconds with {len(results_buffer)} lines.",
          flush=True)
    print(f"Total download size: {total_size:.2f} bits", flush=True)
    print(f"Download rate: {download_rate:.2f} bps", flush=True)

    write = time.time()

    # Write all results at once
    if len(results_buffer) > 0:
        with open(html_json_file, "a") as writer:
            writer.write("\n".join(results_buffer) + "\n")

    print(f"Batch results written in {time.time() - write}", flush=True)

    session.close()

    return {
        "response_counts": response_counts,
        "total_access_time": time.time() - start_time
    }


def get_html(filtered_path, html_path, num_workers=1):
    """
    Reads the filtered CDXJ file and extracts the HTML from each URL present, requesting the ArquivoPT API.
    Saves the HTML files in the "html" directory.
    :param filtered_path: Path to the ArquivoPT filtered CDXJ directory
    :param html_path: Path to save the extracted HTML files
    :param num_workers: Number of threads to split the URLs among
    """

    # Iterate over the filtered CDXJ files
    for cdxj_file in os.listdir(filtered_path):
        if not cdxj_file.endswith(".cdxj") or not cdxj_file.startswith(f"filtered_{COLLECTION}"):
            print(f"Skipped file {cdxj_file}.", flush=True)
            continue

        # Check if the file has already been processed
        cdxj_name = cdxj_file.split(".")[0].split("_")[1]
        html_json_dir = os.path.join(html_path, cdxj_name)

        # if os.path.exists(html_json_dir):
        #     print(f"HTML from file {cdxj_name} already extracted. Skipping...", flush=True)
        #     continue

        os.makedirs(html_json_dir, exist_ok=True)

        # Iterate over the lines in the filtered CDXJ file
        with open(os.path.join(filtered_path, cdxj_file), "r") as f:

            access_time = 0
            response_counts = {}
            total_lines = f.readlines()

            first_unprocessed_batch_idx = next(
                (batch_idx for batch_idx, i in enumerate(range(0, len(total_lines), BATCH_SIZE))
                 if not os.path.exists(os.path.join(html_json_dir, f"{batch_idx:05d}.jsonl"))),
                None
            )

            if first_unprocessed_batch_idx is None:
                print(f"All batches for {cdxj_file} are already processed. Skipping...", flush=True)
                continue

            print(
                f"Processing {cdxj_file} with {len(total_lines)} lines starting in batch {first_unprocessed_batch_idx}",
                flush=True)

            # Process URLs in batches
            for batch_idx, i in enumerate(
                    tqdm(range(first_unprocessed_batch_idx * BATCH_SIZE, len(total_lines), BATCH_SIZE),
                         total=(len(total_lines) - first_unprocessed_batch_idx * BATCH_SIZE) // BATCH_SIZE,
                         unit="batch", smoothing=0), start=first_unprocessed_batch_idx):
                html_json_file = os.path.join(html_json_dir, f"{batch_idx:05d}.jsonl")
                if os.path.exists(html_json_file):
                    print(f"Batch {batch_idx} already processed. Skipping...\n", flush=True)
                    continue

                batch = total_lines[i:i + BATCH_SIZE]
                urls_batch = []

                for line in batch:
                    # Line -> {surt} {timestamp} {json}
                    parts = line.split(" ", 2)
                    json_content = orjson.loads(parts[2])
                    urls_batch.append((parts[0], parts[1], json_content.get("url")))

                results = process_batch(urls_batch, html_json_file, num_workers)

                access_time += results["total_access_time"]
                for status_code, count in results["response_counts"].items():
                    response_counts[status_code] = response_counts.get(status_code, 0) + count

                # Wait remaining time to respect rate limit
                global retry_count
                sleep_time = max(0, DELAY_BETWEEN_BATCHES - results["total_access_time"] +
                                 retry_count * DELAY_BETWEEN_REQUESTS)
                print(f"Retry count: {retry_count}", flush=True)
                print(f"Sleeping for {sleep_time} seconds...\n", flush=True)
                time.sleep(sleep_time)
                retry_count = 0

            print(f"File {cdxj_file} processed successfully.", flush=True)
            print(f"Total API access time: {access_time}", flush=True)
            print(f"Response status codes: {response_counts}", flush=True)
            print("", flush=True)


def request_html(url, timestamp, session):
    """
    Request the HTML content from the ArquivoPT API.
    :param url: URL of the webpage
    :param timestamp: timestamp of the webpage
    :param session: requests session
    :return: HTML content of the webpage
    """
    try:
        html_response = session.get(f"https://arquivo.pt/noFrame/replay/{timestamp}id_/{url}",
                                    timeout=(5, 15))
        return html_response
    except requests.exceptions.RequestException as error:
        print(f"\tError: {error} - URL {url} - timestamp {timestamp}", flush=True)
        return None


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parallel process cdxj files')
    parser.add_argument('files_path', metavar='filesPath', help='path to filtered cdxj files directory')
    parser.add_argument('-o', dest='outFolderPath', help='the path where to write the html files to', required=True)
    parser.add_argument('-w', dest='numWorkers', help='number of workers to split each file amongst', default=1,
                        type=int)

    args = parser.parse_args()

    if not os.path.isdir(args.files_path):
        print(f"The given filtered cdxj folder doesn't exist, create it and rerun. - {args.files_path}")
        exit()

    if not os.path.isdir(args.outFolderPath):
        print(f"The folder given to -o doesn't exist, create it and rerun. - {args.outFolderPath}")
        exit()

    try:
        response = requests.get("https://arquivo.pt")
        time.sleep(DELAY_BETWEEN_REQUESTS)
        if response.status_code == 200:
            print("\nConnection to arquivo successful", flush=True)
            print("Starting to get HTML from filtered CDXJ files...", flush=True)

            start_process_time = time.time()
            get_html(args.files_path, args.outFolderPath, args.collection, args.numWorkers)
            end_process_time = time.time()
            print("Finished getting HTML from filtered CDXJ files.", flush=True)
            print(f"Total processing time: {end_process_time - start_process_time}", flush=True)
        else:
            print(f"Received status code: {response.status_code}")
    except requests.exceptions.RequestException as connect_error:
        print(f"Connection Error: {connect_error}")
