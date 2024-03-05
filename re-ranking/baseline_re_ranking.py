#!/usr/bin/env python3
import os
import shutil
import sys

#!git -C ColBERT/ pull || git clone https://github.com/stanford-futuredata/ColBERT.git
sys.path.insert(0, 'ColBERT/')
import colbert
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection

import pandas as pd
import pyterrier as pt
from passage_chunkers import spacy_passage_chunker
# Load a patched ir_datasets that loads the injected data inside the TIRA sandbox
from tira.third_party_integrations import load_rerank_data, ensure_pyterrier_is_loaded

ensure_pyterrier_is_loaded()

import pyterrier_colbert.ranking

def split_into_snippets(document_text):
    chunker = spacy_passage_chunker.SpacyPassageChunker()
    return chunker.process_batch([{
        "id": 0,
        "url": '',
        "title": '',
        "contents": document_text
    }])[0]['contents']


def transform_snippet_format(snippets):
    df = pd.DataFrame({
        'docno': [str(snippet['id']) for snippet in snippets],
        'text': [snippet['body'] for snippet in snippets]
    })
    return df


def rank_snippets_BM25(query, snippets_df):
    if os.path.exists('pd_index'):
        # Remove the directory and all its contents
        shutil.rmtree('pd_index')
    pd_indexer = pt.DFIndexer("./pd_index")
    indexref3 = pd_indexer.index(snippets_df["text"], snippets_df["docno"])
    index = pt.IndexFactory.of(indexref3)
    bm25 = pt.BatchRetrieve(index, controls={"wmodel": "BM25"})

    #remove ? due to error in terrier query parser
    query = query.replace('?', '')
    if os.path.exists('pd_index'):
        # Remove the directory and all its contents
        shutil.rmtree('pd_index')

    result = bm25.search(query)

    merged_df = pd.merge(snippets_df, result, on='docno')

    # Convert to list of dictionaries
    result_list = merged_df.apply(lambda row: {'score': row['score'], 'text': row['text']}, axis=1).tolist()

    return result


def rank_snippets_trColBERT(query, snippets_df):
    checkpoint="http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip"
    factory = pyterrier_colbert.ranking.ColBERTFactory(checkpoint, None, None, gpu = False)
    result = factory.explain_text("why did the us voluntarily enter ww1", "the USA entered ww2 because of pearl harbor")
    print(result)
    return result

def rank_snippets_ColBERT(query, snippets_df):
    #indexer
    n_gpu: int = 1  # Set your number of available GPUs
    experiment: str = "newex"  # Name of the folder where the logs and created indices will be stored
    index_name: str = "new"
    with Run().context(RunConfig(nranks=n_gpu, experiment=experiment)):
        config = ColBERTConfig(
          doc_maxlen=300  # Our model supports 8k context length for indexing long documents
        )
        indexer = Indexer(
          checkpoint="colbert-ir/colbertv2.0",
          config=config,
        )
        documents = []
        indexer.index(name=index_name, collection=documents)
    
    #searcher 
    n_gpu: int = 0
    k: int = 10  # how many results you want to retrieve

    with Run().context(RunConfig(nranks=n_gpu, experiment=experiment)):
        config = ColBERTConfig(
          query_maxlen=128  # Although the model supports 8k context length, we suggest not to use a very long query, as it may cause significant computational complexity and CUDA memory usage.
        )
        searcher = Searcher(
          index=index_name, 
          config=config
        )  # You don't need to specify the checkpoint again, the model name is stored in the index.
        query = "How to use ColBERT for indexing long documents?"
        result = searcher.search(query, k=k)
    print(result)
    return result

def find_top_snippets(query, document_text, ranker = 'BM25'):
    # First: split document_text into snippets
    # https://github.com/grill-lab/trec-cast-tools/tree/master/corpus_processing/passage_chunkers

    snippets = split_into_snippets(document_text)

    # Second: transform snippet format from output of split_into_snippets to input of rank_snippets

    snippets_df = transform_snippet_format(snippets)

    # Third: rank snippets

    if ranker == 'BM25':
        ranking = rank_snippets_BM25(query, snippets_df)
    elif ranker == 'ColBERT':
        pass
        #non functional
        #ranking = rank_snippets_ColBERT(query, snippets_df)



    # Return values
    return ranking


if __name__ == '__main__':
    # In the TIRA sandbox, this is the injected re-ranking dataset, injected via the environment variable TIRA_INPUT_DIRECTORY
    re_rank_dataset = load_rerank_data(default='workshop-on-open-web-search/re-ranking-20231027-training')

    # Alternatively, you could use the scored docs of ir_datasets, e.g.:
    # from tira.third_party_integrations import ir_dataset
    # re_rank_dataset = ir_datasets.load(default='workshop-on-open-web-search/document-processing-20231027-training')

    document_snippets = []

    for _, i in re_rank_dataset.iterrows():
        document_snippets += [
            {'qid': i['qid'], 'docno': i['docno'], 'snippets': find_top_snippets(i['query'], i['text'],'BM25')}]

    document_snippets = pd.DataFrame(document_snippets)
    document_snippets.to_json('./re-rank.jsonl.gz', lines=True, orient='records')
