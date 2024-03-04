#!/usr/bin/env python3
# Load a patched ir_datasets that loads the injected data inside the TIRA sandbox
from tira.third_party_integrations import load_rerank_data, persist_and_normalize_run
from pathlib import Path
import pandas as pd

def find_top_snippets(query, document_text):
    # Our baseline re-ranker just emits 1 + the score of the previous ranker.
    return [{'snippet_score':1, 'snippet_text': 'ddsfds'}, {'snippet_score':10.9, 'snippet_text': 'a'}, ]

if __name__ == '__main__':
    # In the TIRA sandbox, this is the injected re-ranking dataset, injected via the environment variable TIRA_INPUT_DIRECTORY
    re_rank_dataset = load_rerank_data(default='workshop-on-open-web-search/re-ranking-20231027-training')

    # Alternatively, you could use the scored docs of ir_datasets, e.g.:
    # from tira.third_party_integrations import ir_dataset
    # dataset = ir_datasets.load(default='workshop-on-open-web-search/document-processing-20231027-training')

    document_snippets = []

    for _, i in re_rank_dataset.iterrows():
        document_snippets += [{'qid': i['qid'], 'docno': i['docno'], 'snippets': find_top_snippets(i['query'], i['text'])}]
    
    document_snippets = pd.DataFrame(document_snippets)
    document_snippets.to_json('./re-rank.jsonl.gz', lines=True, orient='records')