#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import torch
import pandas as pd
import pyterrier as pt
import pyterrier_dr
import spacy

from passage_chunkers.abstract_passage_chunker import AbstractPassageChunker


# Load a patched ir_datasets that loads the injected data inside the TIRA sandbox
from tira.third_party_integrations import load_rerank_data, ensure_pyterrier_is_loaded, get_output_directory

ensure_pyterrier_is_loaded()

tokeniser = pt.autoclass("org.terrier.indexing.tokenisation.Tokeniser").getTokeniser()
def pt_tokenise(text):
    return ' '.join(tokeniser.getTokens(text))


nlp = spacy.load("en_core_web_sm", exclude=[
    "parser", "tagger", "ner", "attribute_ruler", "lemmatizer", "tok2vec"])
nlp.enable_pipe("senter")
nlp.max_length = 2000000  # for documents that are longer than the spacy character limit


class ParameterizedSpacyPassageChunker(AbstractPassageChunker):
    """
    Adapted from https://github.com/grill-lab/trec-cast-tools/blob/master/corpus_processing/passage_chunkers/spacy_passage_chunker.py
    Basically the same as #SpacyPassageChunker. Only difference is that the snippet size can be set in #__init__
    """
    def __init__(self, snippet_size=250):
        self.snippet_size = snippet_size

    def process_batch(self, document_batch: list[dict]) -> list[dict]:
        """
        Divides the documents of the given document list into snippets.
        :param document_batch: List of documents. Documents are dicts - content must be at 'contents' key.
        :return: List of documents. Documents are dicts - snippets can be found at 'contents' key.
        """
        regexp = re.compile(r'[a-zA-Z0-9]')
        document_batch = list(filter(lambda doc: regexp.search(doc['contents']), document_batch))
        batch_document_texts = [document['contents'] for document in document_batch]
        processed_document_texts = nlp.pipe(batch_document_texts, n_process=1)

        for index, document in tqdm(enumerate(processed_document_texts), total=len(document_batch)):
            document_sentences = list(document.sents)
            sentences_word_count = [
                len([token for token in sentence])
                for sentence in document_sentences
            ]

            generated_passages = self.chunk_document(document_sentences, sentences_word_count, self.snippet_size)
            document_batch[index]['contents'] = generated_passages

        return document_batch


def split_dataframe_into_snippets(documents: pd.DataFrame, snippet_size=250) -> pd.DataFrame:
    document_list = documents.rename(columns={'text': 'contents'}).to_dict('records')

    chunker = ParameterizedSpacyPassageChunker(snippet_size)
    document_list = chunker.process_batch(document_list)
    ret = {}
    for i in document_list:
        if i['qid'] not in ret:
            ret[i['qid']] = {'query': i['query'], 'documents': {}}

        ret[i['qid']]['documents'][i['docno']] = i['contents']

    return ret


def split_into_snippets(document_text: str, snippet_size=250) -> list[dict]:
    chunker = ParameterizedSpacyPassageChunker(snippet_size)
    return chunker.process_batch([{
        "id": 0,
        "url": '',
        "title": '',
        "contents": document_text
    }])[0]['contents']


def crossencode(ret, model, tokenizer):
    results = []
    for obj in ret:
        pairs = [(obj['query'],s['text'])for s in obj['snippets']]

        features = tokenizer(pairs,  padding=True, truncation=True, return_tensors="pt")

        model.eval()
        with torch.no_grad(): 
            scores = model(**features).logits
            scores = scores.flatten().tolist()

        newsnippets = []
        for score, snippet in zip(scores,obj['snippets']):
            snippet['score'] = score
            snippet['wmodel'] = 'cross_encode'
            newsnippets.append(snippet)
        obj['snippets'] = newsnippets
        results.append(obj)
    return results    

def colbert_pipeline(docs_df: pd.DataFrame, query):
    colbert_model = pyterrier_dr.TctColBert('sentence-transformers/all-MiniLM-L12-v2')
    docs_df['qid'] = '0'
    docs_df['query'] = query
    
    result_df = colbert_model(docs_df)

    merged_df = pd.merge(docs_df, result_df, on='docno')
    merged_df = merged_df.sort_values('score', ascending=False)

    # Convert to list of dictionaries
    result_list = merged_df.apply(lambda row: {'score': row['score'], 'text': row['text_x']}, axis=1)

    return result_list.tolist()

def find_top_snippets_for_all_documents(qid, query, documents, wmodel, cross_encode, model=None, tokenizer=None):
    query = pt_tokenise(query)
    df = []
    covered_docnos = set()
    for docno, passages in documents.items():
        for passage in passages:
            covered_docnos.add(docno + '_' + str(passage['id']))
            df += [{
                'qid': str(qid),
                'query': query,
                'docno': docno + '_' + str(passage['id']),
                'original_docno': docno,
                'text': passage['body']
                }]
    df = pd.DataFrame(df)
    textscorer = pt.batchretrieve.TextScorer(takes="docs", body_attr="text", wmodel=wmodel)
    rtr = textscorer.transform(df)
    ret_docs = {}

    for _, i in rtr.iterrows():
        if i['original_docno'] not in ret_docs:
            ret_docs[i['original_docno']] = []
        ret_docs[i['original_docno']] += [{'wmodel': wmodel, 'score': i['score'], 'text': i['text']}]
    
    ret = []
    for docno, snippets in ret_docs.items():
        ret += [{'qid': qid, 
                 'query': query, 
                 'docno': docno,
                 'snippets': sorted(snippets, key=lambda j: j['score'], reverse=True)[:3]
                 }]
    if cross_encode and model and tokenizer:
        crossencode(ret, model, tokenizer)
    return ret


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

    # Alternatively, you could use the scored docs of ir_datasets, e.g.:
    # from tira.third_party_integrations import ir_dataset
    # re_rank_dataset = ir_datasets.load(default='workshop-on-open-web-search/document-processing-20231027-training')

    #re_rank_dataset = pd.read_json('rerank-01.json.gz', lines=True, chunksize=1000).read()

    args = parse_arguments()
    preprocessed_docs = split_dataframe_into_snippets(re_rank_dataset, args.snippet_size)
    
    document_snippets = []

    model, tokenizer = None, None
    if args.cross_encode:
        print('Loading cross-encoder model')
        model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
        tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print('Done cross-encoder model is loaded.')

    for qid, i in tqdm(preprocessed_docs.items()):
        document_snippets += find_top_snippets_for_all_documents(qid, i['query'], i['documents'], args.retrieval, args.cross_encode, model=model, tokenizer=tokenizer)
        
    document_snippets = pd.DataFrame(document_snippets)

    # The expected output directory, injected via the environment variable TIRA_OUTPUT_DIRECTORY
    output_dir = get_output_directory('.')

    output_file = Path(output_dir) / 'documents.jsonl.gz'
    document_snippets.to_json(output_file, lines=True, orient='records')
