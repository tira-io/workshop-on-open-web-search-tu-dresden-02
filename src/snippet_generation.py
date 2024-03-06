#!/usr/bin/env python3
import os
import argparse
import re
import shutil
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import pyterrier as pt
import pyterrier_dr

from sentence_transformers import CrossEncoder
from src.parameterized_spacy_passage_chunker import ParameterizedSpacyPassageChunker
# Load a patched ir_datasets that loads the injected data inside the TIRA sandbox
from tira.third_party_integrations import load_rerank_data, ensure_pyterrier_is_loaded, get_output_directory

ensure_pyterrier_is_loaded()

tokeniser = pt.autoclass("org.terrier.indexing.tokenisation.Tokeniser").getTokeniser()
def pt_tokenise(text):
    return ' '.join(tokeniser.getTokens(text))

def split_dataframe_into_snippets(documents: pd.DataFrame, snippet_size=250) -> pd.DataFrame:
    document_list = documents.rename(columns={'text': 'contents'}).to_dict('records')

    chunker = ParameterizedSpacyPassageChunker(snippet_size)
    document_list = chunker.process_batch(document_list)

    return pd.DataFrame(document_list).rename(columns={'contents': 'text'})


def split_into_snippets(document_text: str, snippet_size=250) -> list[dict]:
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
    pd_indexer = pt.DFIndexer(index_path="memory_index",type=pt.index.IndexingType.MEMORY)
    indexref3 = pd_indexer.index(snippets_df["text"], snippets_df["docno"])
    index = pt.IndexFactory.of(indexref3)
    retrieved = pt.BatchRetrieve(index, controls={"wmodel": ranker})

    #remove ? due to error in terrier query parser
    query = pt_tokenise(query)
    result = retrieved.search(query)

    merged_df = pd.merge(snippets_df, result, on='docno')
    merged_df = merged_df.sort_values('score', ascending=False)

    # Convert to list of dictionaries
    result_list = merged_df.apply(lambda row: {'score': row['score'], 'text': row['text']}, axis=1)

    return result_list.tolist()

def crossencode(query, top_k_snippets):
    top_k_texts = [d['text'] for d in top_k_snippets]
    #model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
    pairs = [(query, doc) for doc in top_k_texts]

    model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
    tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')

    features = tokenizer(pairs,  padding=True, truncation=True, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        scores = model(**features).logits
        scores = scores.flatten().tolist()

    #scores = model.predict(pairs)
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


def find_top_snippets(query, snippets, ranker='Tf', max_snippets=3, snippet_size=250, use_crossencoder=True):
    # Check if document or query is empty
    regexp = re.compile(r'[a-zA-Z0-9]') 
    if not regexp.search(snippets) or not regexp.search(query):
        return []

    # First: split document_text into snippets
    # https://github.com/grill-lab/trec-cast-tools/tree/master/corpus_processing/passage_chunkers

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
    print(ranking)
    return ranking[0:max_snippets]


def parse_arguments():
    parser = argparse.ArgumentParser(description="Argument Parser")

    parser.add_argument("--retrieval", choices=["BM25", "PL2", "Tf", "ColBERT"], default="Tf", help="The retrieval "
                                                                                                    "model")
    parser.add_argument("--cross-encode", action="store_true", default=False, help="Use a cross-encoder to re-rank "
                                                                                   "the top-k passages of the "
                                                                                   "retrieval model")
    parser.add_argument("--snippet-size", default=250, type=int, help="The approximate size of created snippets")
    parser.add_argument("--top-snippets", default=3, type=int, help="Number k for top k snippets that are retrieved.")

    return parser.parse_args()


if __name__ == '__main__':
    # In the TIRA sandbox, this is the injected re-ranking dataset, injected via the environment variable TIRA_INPUT_DIRECTORY
    re_rank_dataset = load_rerank_data(default='workshop-on-open-web-search/re-ranking-20231027-training')
    print(re_rank_dataset)
    # Alternatively, you could use the scored docs of ir_datasets, e.g.:
    # from tira.third_party_integrations import ir_dataset
    # re_rank_dataset = ir_datasets.load(default='workshop-on-open-web-search/document-processing-20231027-training')

    args = parse_arguments()
    preprocessed_docs = split_into_snippets(re_rank_dataset)
    document_snippets = []
    for _, i in preprocessed_docs:
        document_snippets += [
            {'qid': i['qid'], 'docno': i['docno'], 'snippets': find_top_snippets(i['query'], i['text'], args.retrieval,
                                                                                 args.top_snippets, args.snippet_size,
                                                                                 args.cross_encode)}]

    document_snippets = pd.DataFrame(document_snippets)

    # The expected output directory, injected via the environment variable TIRA_OUTPUT_DIRECTORY
    output_dir = get_output_directory('.')

    output_file = Path(output_dir) / 'documents.jsonl.gz'
    document_snippets.to_json(output_file, lines=True, orient='records')