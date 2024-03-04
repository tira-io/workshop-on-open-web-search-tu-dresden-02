#!/usr/bin/env python3
# Load a patched ir_datasets that loads the injected data inside the TIRA sandbox
from tira.third_party_integrations import load_rerank_data, persist_and_normalize_run
from pathlib import Path
import os
import shutil
import pandas as pd
import pyterrier as pt
if not pt.started():
  pt.init()

import sys
sys.path.insert(0, '/home/trec-cast-tools/corpus_processing/')
from passage_chunkers import spacy_passage_chunker

def split_into_snippets(document_text):
    chunker = spacy_passage_chunker.SpacyPassageChunker()
    return chunker.process_batch([{
            "id" : 0,
            "url": '',
            "title" : '',
            "contents" : document_text
        }])[0]['contents']

def transform_snippet_format(snippets):
    df = pd.DataFrame({
                    'docno': [str(snippet['id']) for snippet in snippets],
                    'text': [snippet['body'] for snippet in snippets]
                    })
    return df

def rank_snippets(query, snippets_df):
    if os.path.exists('pd_index'):
    # Remove the directory and all its contents
        shutil.rmtree('pd_index')
    pd_indexer = pt.DFIndexer("./pd_index")
    indexref3 = pd_indexer.index(snippets_df["text"], snippets_df["docno"])
    index = pt.IndexFactory.of(indexref3)
    bm25 = pt.BatchRetrieve(index, controls = {"wmodel": "BM25"})
    #print(query)

    #remove ? due to error in terrier query parser
    query = query.replace('?', '')
    if os.path.exists('pd_index'):
    # Remove the directory and all its contents
        shutil.rmtree('pd_index')
    return bm25.search(query)

def find_top_snippets(query, document_text):
    # First: split document_text into snippets
    # https://github.com/grill-lab/trec-cast-tools/tree/master/corpus_processing/passage_chunkers
    
    snippets = split_into_snippets(document_text)

    # Second: transform snippet format from output of split_into_snippets to input of rank_snippets

    snippets_df = transform_snippet_format(snippets)

    # Third: rank snippets

    ranking = rank_snippets(query, snippets_df)
    
    # Return values
    #return ranking
    return [{'snippet_score':1, 'snippet_text': 'ddsfds'}, {'snippet_score':10.9, 'snippet_text': 'a'}, ]

if __name__ == '__main__':
    # In the TIRA sandbox, this is the injected re-ranking dataset, injected via the environment variable TIRA_INPUT_DIRECTORY
    re_rank_dataset = load_rerank_data(default='workshop-on-open-web-search/re-ranking-20231027-training')

    # Alternatively, you could use the scored docs of ir_datasets, e.g.:
    # from tira.third_party_integrations import ir_dataset
    # re_rank_dataset = ir_datasets.load(default='workshop-on-open-web-search/document-processing-20231027-training')

    document_snippets = []

    for _, i in re_rank_dataset.iterrows():
        document_snippets += [{'qid': i['qid'], 'docno': i['docno'], 'snippets': find_top_snippets(i['query'], i['text'])}]
    
    document_snippets = pd.DataFrame(document_snippets)
    document_snippets.to_json('./re-rank.jsonl.gz', lines=True, orient='records')