#!/usr/bin/env python3
import os
import argparse
import shutil

import pandas as pd
import pyterrier as pt
import pyterrier_dr

from sentence_transformers import CrossEncoder
from parameterized_spacy_passage_chunker import ParameterizedSpacyPassageChunker
# Load a patched ir_datasets that loads the injected data inside the TIRA sandbox
from tira.third_party_integrations import load_rerank_data, ensure_pyterrier_is_loaded

ensure_pyterrier_is_loaded()


def split_into_snippets(document_text, snippet_size=250):
    chunker = ParameterizedSpacyPassageChunker(snippet_size)
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


def rank_snippets_lexical(query, snippets_df, ranker):
    if os.path.exists('pd_index'):
        # Remove the directory and all its contents
        shutil.rmtree('pd_index')
    pd_indexer = pt.DFIndexer("./pd_index")
    indexref3 = pd_indexer.index(snippets_df["text"], snippets_df["docno"])
    index = pt.IndexFactory.of(indexref3)
    retrieved = pt.BatchRetrieve(index, controls={"wmodel": ranker})

    #remove ? due to error in terrier query parser
    query = query.replace('?', '')
    if os.path.exists('pd_index'):
        # Remove the directory and all its contents
        shutil.rmtree('pd_index')

    result = retrieved.search(query)

    merged_df = pd.merge(snippets_df, result, on='docno')
    merged_df = merged_df.sort_values('score', ascending=False)

    # Convert to list of dictionaries
    result_list = merged_df.apply(lambda row: {'score': row['score'], 'text': row['text']}, axis=1)

    return result_list.tolist()


def crossencode(query, top_k_snippets):
    top_k_texts = [d['text'] for d in top_k_snippets]
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
    pairs = [(query, doc) for doc in top_k_texts]
    scores = model.predict(pairs)
    reranked_top_k = [{'score': scores[i], 'text': top_k_texts[i]} for i in range(len(top_k_texts))]
    return reranked_top_k


def colbert_pipeline(docs_df: pd.DataFrame, query):
    colbert_model = pyterrier_dr.TctColBert('sentence-transformers/all-MiniLM-L12-v2')
    docs_df['qid'] = '0'
    docs_df['query'] = query
    #print(docs_df)
    result_df = colbert_model(docs_df)
    #print(result_df)
    merged_df = pd.merge(docs_df, result_df, on='docno')
    merged_df = merged_df.sort_values('score', ascending=False)
    #print(merged_df)

    # Convert to list of dictionaries
    result_list = merged_df.apply(lambda row: {'score': row['score'], 'text': row['text_x']}, axis=1)

    return result_list.tolist()


def find_top_snippets(query, document_text, ranker='Tf', max_snippets=3, snippet_size=250, use_crossencoder=True):
    # First: split document_text into snippets
    # https://github.com/grill-lab/trec-cast-tools/tree/master/corpus_processing/passage_chunkers

    snippets = split_into_snippets(document_text, snippet_size)

    # Second: transform snippet format from output of split_into_snippets to input of rank_snippets

    snippets_df = transform_snippet_format(snippets)

    # Third: rank snippets

    if ranker in ('BM25', 'PL2', 'Tf'):
        ranking = rank_snippets_lexical(query, snippets_df, ranker)
        if use_crossencoder:
            ranking = crossencode(query, ranking[0:max_snippets])
    elif ranker == 'ColBERT':
        #non functional
        ranking = colbert_pipeline(snippets_df, [query])
        if use_crossencoder:
            ranking = crossencode(query, ranking[0:max_snippets])

    # Return values
    return ranking[0:max_snippets]


def parse_arguments():
    parser = argparse.ArgumentParser(description="Argument Parser")

    parser.add_argument("--retrieval", choices=["BM25", "PL2", "Tf", "ColBERT"], default="Tf", help="The retrieval "
                                                                                                    "model")
    parser.add_argument("--cross-encode", action="store_true", default=False, help="Use a cross-encoder to re-rank "
                                                                                   "the top-k passages of the "
                                                                                   "retrieval model")
    parser.add_argument("--snippet-size", default=250, help="The approximate size of created snippets")
    parser.add_argument("--top-snippets", default=3, help="Number k for top k snippets that are retrieved.")

    return parser.parse_args()


if __name__ == '__main__':
    # In the TIRA sandbox, this is the injected re-ranking dataset, injected via the environment variable TIRA_INPUT_DIRECTORY
    re_rank_dataset = load_rerank_data(default='workshop-on-open-web-search/re-ranking-20231027-training')

    # Alternatively, you could use the scored docs of ir_datasets, e.g.:
    # from tira.third_party_integrations import ir_dataset
    # re_rank_dataset = ir_datasets.load(default='workshop-on-open-web-search/document-processing-20231027-training')

    args = parse_arguments()
    document_snippets = []
    for _, i in re_rank_dataset.iterrows():
        document_snippets += [
            {'qid': i['qid'], 'docno': i['docno'], 'snippets': find_top_snippets(i['query'], i['text'], args.retrieval,
                                                                                 args.top_snippets, args.snippet_size,
                                                                                 args.cross_encode)}]

    document_snippets = pd.DataFrame(document_snippets)
    document_snippets.to_json(f'./snippets_{args.retrieval}_crossencoder{args.cross_encode}_'
                              f'snippets-size{args.snippet_size}_top-snippets{args.top_snippets}.jsonl.gz',
                              lines=True, orient='records')
