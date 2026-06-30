<div align="center">

<img src=src="https://github.com/AMALIA-LLM/amalia-llm.github.io/blob/main/source/_static/logo/logo-color-black.png?raw=true" alt="AMALIA" width="300"/>

<br/>

**A Fully Open Large Language Model for European Portuguese**

[![Website](https://img.shields.io/badge/Website-amaliallm.pt-blue)](https://amaliallm.pt/)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-amalia--llm-yellow)](https://huggingface.co/amalia-llm)
[![Paper](https://img.shields.io/badge/Paper-Technical%20Report-red?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.26511)
 
</div>

---

# ArquivoPT Data Extraction

Code with the pipeline to extract, process, filter and deduplicate data from ArquivoPT.

This data is divided into several types of collections, each with its own crawling methods, timestamps and seeds.
The collections that have been/are being processed, divided by processing technique, the collections that have been/are
being filtered and deduplicated, and a list with collections to be processed can be
found [here](https://docs.google.com/spreadsheets/d/1rflsqslYSvnnrsNFhGBpF6A3JdcP3DRhow5X5bINa7s/edit?usp=sharing).

## Data Extaction Methods

The data extraction methods are divided into two main categories: API Extraction and WARC Processing.

In all the methods, you need to change which collection are you processing directly in the .py files. I recommend also
to change the job names in the .sh files to avoid confusion when running multiple jobs at the same time and ease the
logging.

### API Extraction

Download the CDXJ files from the [CDXJ List](https://arquivo.pt/datasets/cdxj/), filter them to get only the ones that
have pt domains, and then get the html of the pages using the API. Finally, scrape the html to extract the data. This
method is slow, taking 3-4 days to process a collection with 20 million entries after filtering.

- `filter_cdxj.py`: Filter the CDXJ files' entries to get only the ones with urls with the desired domains (pt domains,
  mainly);
- `cdxj_get_html.py`: Request the ArquivoPT API for the html of the pages using the filtered CDXJ files;
- `scraper.py`: Scrape the html to extract the text data;

### WARC Processing

Text data is extracted from the WARC files. This method is a lot faster than the API extraction, taking around 10 hours
to process a collection with 20 million entries.

For this approach, we use our version of the [Datatrove library](https://github.com/AMALIA-NOVA/datatrove-amalia) from
HuggingFace, which is optimized for our needs. Be careful that depending on the collection you may have to change the
`WarcReader` parameters, as the collections may have different file structures.

**Internet Archive**

Download the WARC files from
the [Internet Archive](https://archive.org/details/@daniel_gomes?and%5B%5D=collection%3A%22portuguese-web-archive%22)
and process them.

- `ia_warc_process`: Extract the text from the WARC files downloaded from the Internet Archive;

## Filtering and Deduplication

After the data extraction, the data is filtered and deduplicated. The filtering is done to remove unwanted data,
following heuristic and model-based filters, and the deduplication is done to remove duplicate entries.

In all the methods, you need to change which collection are you processing directly in the .py files. I recommend also
to change the job names to avoid confusion when running multiple jobs at the same time and ease the logging.

1. `filters/data_filtering.py`: Applies post-scraping steps to the extracted text data, and use language and heuristic
   filters
   to remove unwanted data;
2. `filters/minhash_dedup.py`: Applies MinHash deduplication to the filtered data to remove duplicate entries;
3. `filters/model_classifier.py`: Applies a model-based classifier to the filtered data, dividing it into high, medium
   and low;
   quality data;

After collecting a significant amount of data and tokens, perform a deduplication across all collections.

- `filters/minhash_dedup_quality.py`: Applies MinHash deduplication to the high, medium and low quality data to remove
   duplicate
   entries across all collections;

Alternatively, the scripts `filter_dedup_all.py` and `classify_all.py` can be used to batch filter, 
deduplicate and classify a directory of collections. Pay attention to not 
duplicate the post-scraping and language filtering steps in case they were performed during the extraction step.

## Stats

Get statistics and plots from the data, such as the number of entries, the number of tokens, token length distribution,
domain distribution, filtering reasons, etc.

## Citation

```bibtex
@misc{simplício2026amaliatechnicalreportfully,
      title={AMALIA Technical Report: A Fully Open Source Large Language Model for European Portuguese}, 
      author={Afonso Simplício and Gonçalo Vinagre and Miguel Moura Ramos and Diogo Tavares and Rafael Ferreira and Giuseppe Attanasio and Duarte M. Alves and Inês Calvo and Inês Vieira and Rui Guerra and James Furtado and Beatriz Canaverde and Iago Paulo and Vasco Ramos and Diogo Glória-Silva and Miguel Faria and Marcos Treviso and Daniel Gomes and Pedro Gomes and David Semedo and André Martins and João Magalhães},
      year={2026},
      eprint={2603.26511},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.26511}, 
}
