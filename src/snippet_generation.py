#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import pandas as pd
import pyterrier as pt
import spacy
import torch
from passage_chunkers.abstract_passage_chunker import AbstractPassageChunker
# Load a patched ir_datasets that loads the injected data inside the TIRA sandbox
from tira.third_party_integrations import load_rerank_data, ensure_pyterrier_is_loaded, get_output_directory
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ensure_pyterrier_is_loaded()

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

tokeniser = pt.autoclass("org.terrier.indexing.tokenisation.Tokeniser").getTokeniser()

nlp = spacy.load("en_core_web_sm", exclude=[
    "parser", "tagger", "ner", "attribute_ruler", "lemmatizer", "tok2vec"])
nlp.enable_pipe("senter")
nlp.max_length = 2000000  # for documents that are longer than the spacy character limit


class ParameterizedSpacyPassageChunker(AbstractPassageChunker):
    """
    Adapted from
    https://github.com/grill-lab/trec-cast-tools/blob/master/corpus_processing/passage_chunkers/spacy_passage_chunker.py
    Basically the same as #SpacyPassageChunker. Only difference is that the snippet size can be set in #__init__
    """

    def __init__(self, snippet_size=250):
        self.snippet_size = snippet_size

    def process_batch(self, document_batch: list[dict[str, str]]) -> list[dict[str, str | list[dict[str, int | str]]]]:
        """
        Divides the documents of the given document list into snippets.
        :param document_batch: List of documents. Documents are dicts - content must be at 'contents' key.
        :return: List of documents. Documents are dicts - snippets can be found at 'contents' key.
        """
        regexp = re.compile(r'[a-zA-Z0-9]')
        document_batch = list(filter(lambda document: regexp.search(document['contents']), document_batch))
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


def split_dataframe_into_snippets(documents: pd.DataFrame, snippet_size: int = 250) -> (
        dict)[str, dict[str, str | dict[str, list[dict[str, int | str]]]]]:
    """
    Splits the documents in the given DataFrame into snippets of the given token length.
    @param documents: DataFrame of query-document pairs with rows 'qid' (i.e., the query id), 'query'
    (i.e., the query string), 'docno' (i.e., the document id) and 'text' (i.e., the document text).
    @param snippet_size: Approximate size of the generated snippets. Actual size may be lower.
    @return: A dict mapping query ids to query dicts.
    Query dicts contain the keys 'query' (i.e., the query string) and a document dict.
    A document dict maps document numbers (docno) to the respective list of snippets.
    Snippets are also dicts and contain the keys 'id' and 'body'.
    """
    document_list = documents.rename(columns={'text': 'contents'}).to_dict('records')

    chunker = ParameterizedSpacyPassageChunker(snippet_size)
    document_list = chunker.process_batch(document_list)
    ret = {}
    for document in document_list:
        if document['qid'] not in ret:
            ret[document['qid']] = {'query': document['query'], 'documents': {}}

        ret[document['qid']]['documents'][document['docno']] = document['contents']

    return ret


def cross_encode(document_snippet_list: list[dict[str, str | dict[str, float | str]]], model, tokenizer) -> (
        list)[dict[str, str | dict[str, float | str]]]:
    """
    Re-ranks the given snippets using the given cross-encoder model and tokenizer.
    @param document_snippet_list: List of dicts, each representing a document-query pair.
    A document-query dict contains keys 'qid' which is the id of the query, 'query' which is the actual query string
    query, 'docno' which is the id of the respective document and 'snippets' which is a list of dicts that each
    represent a snippet.
    A snippet is a dict containing keys 'wmodel' which is a str defining the weighting model that calculated  the score,
    'score' which is a floating point scoring value and 'text' which is the actual text content of the snippet.
    @param model: Cross-encoder model (should not be None when do_cross_encode is true).
    @param tokenizer: Cross-encoder tokenizer (should not be None when do_cross_encode is true).
    @return: A list of dicts, each representing a document-query pair.
    A document-query dict contains keys 'qid' which is the id of the query, 'query' which is the actual query string
    query, 'docno' which is the id of the respective document and 'snippets' which is a list of dicts that each
    represent a snippet.
    A snippet is a dict containing keys 'wmodel' which is a str defining the weighting model that calculated the score
    before the cross-encoder, 'score' which is a floating point scoring value and 'text' which is
    the actual text content of the snippet.
    """
    pairs = []
    for document in document_snippet_list:
        for snippet in document['snippets']:
            pairs += (document['query'], snippet['text'])

    features = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt").to(device)

    model.eval()
    with torch.no_grad():
        scores = model(**features).logits
        scores = scores.flatten().tolist()

    score_index = 0

    for document in document_snippet_list:
        snippet: dict[str, float | str]
        for snippet in document["snippets"]:
            snippet["score"] = scores[score_index]
            score_index += 1
    return document_snippet_list


def find_top_snippets_for_all_documents(query_id: str, query: str, documents: dict[str, list[dict[str, str]]],
                                        wmodel: str, top_snippets: int, do_cross_encode: bool, model=None,
                                        tokenizer=None) -> list[dict[str, str | list[dict[str, float | str]]]]:
    """
    Ranks the given snippets as described in README
    @param query_id: Id of the respective query.
    @param query: String that has been searched.
    @param documents: Dict, that maps document numbers (docno) to the respective list of snippets.
    Snippets are also dicts and must contain the keys 'id' and 'body'.
    @param wmodel: Str defining the weighting model used for pre-ranking (either 'BM25', 'PL2' or 'Tf').
    @param top_snippets: Int defining the number k of top snippets (see README).
    @param do_cross_encode: Bool defining whether step 4 (see README) should be included.
    @param model: Cross-encoder model (should not be None when do_cross_encode is true).
    @param tokenizer: Cross-encoder tokenizer (should not be None when do_cross_encode is true).
    @return: A list of dicts, each representing a document-query pair.
    A document-query dict contains keys 'qid' which is equal to parameter query_id, 'query' which is equal to parameter
    query, 'docno' which is the id of the respective document and 'snippets' which is a list of dicts that each
    represent a snippet.
    A snippet is a dict containing keys 'wmodel' which is a str equal to parameter wmodel, 'score' which is a floating
    point scoring value and 'text' which is the actual text content of the snippet.
    """
    # tokenise query
    query = ' '.join(tokeniser.getTokens(query))
    df = []
    covered_docnos = set()
    for docno, passages in documents.items():
        for passage in passages:
            covered_docnos.add(docno + '_' + str(passage['id']))
            df += [{
                'qid': str(query_id),
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
        ret += [{'qid': query_id,
                 'query': query,
                 'docno': docno,
                 'snippets': sorted(snippets, key=lambda j: j['score'], reverse=True)[:top_snippets]
                 }]
    if do_cross_encode and model and tokenizer:
        cross_encode(ret, model, tokenizer)
    return ret


def parse_arguments() -> argparse.Namespace:
    """
    Parses the given command line argument.
    @return: An argparse.Namespace containing the respective values.
    """
    parser = argparse.ArgumentParser(description="Argument Parser")

    # select retrieval model - BM25, PL2 or Tf (default)
    parser.add_argument("--retrieval", choices=["BM25", "PL2", "Tf"], default="Tf", help="The retrieval model")

    # use cross-encoder (default false)
    parser.add_argument("--cross-encode", action="store_true", default=False, help="Use a cross-encoder to re-rank "
                                                                                   "the top-k passages of the "
                                                                                   "retrieval model")

    # select snippet size in tokens (default 250)
    parser.add_argument("--snippet-size", default=250, type=int, help="The approximate size of created snippets")

    # select number of top snippets to
    parser.add_argument("--top-snippets", default=3, type=int, help="Number k for top k snippets that are retrieved.")

    return parser.parse_args()


if __name__ == '__main__':
    # In the TIRA sandbox, this is the injected re-ranking dataset, injected via the environment variable
    # TIRA_INPUT_DIRECTORY
    re_rank_dataset = load_rerank_data(default='workshop-on-open-web-search/re-ranking-20231027-training')

    # Alternatively, you could use the scored docs of ir_datasets, e.g.:
    # from tira.third_party_integrations import ir_dataset
    # re_rank_dataset = ir_datasets.load(default='workshop-on-open-web-search/document-processing-20231027-training')

    # get command line arguments
    args = parse_arguments()

    # split docs into snippets (see step 1 in README)
    preprocessed_docs = split_dataframe_into_snippets(re_rank_dataset, args.snippet_size)

    document_snippets = []

    # initialize cross-encoder if requested
    ce_model, ce_tokenizer = None, None
    if args.cross_encode:
        print('Loading cross-encoder model')
        ce_model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2').to(device)
        ce_tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print('Done cross-encoder model is loaded.')

    # perform re-ranking for all queries
    for qid, doc in tqdm(preprocessed_docs.items()):
        document_snippets += find_top_snippets_for_all_documents(qid, doc['query'], doc['documents'], args.retrieval,
                                                                 args.top_snippets, args.cross_encode, model=ce_model,
                                                                 tokenizer=ce_tokenizer)

    document_snippets = pd.DataFrame(document_snippets)

    # The expected output directory, injected via the environment variable TIRA_OUTPUT_DIRECTORY
    output_dir = get_output_directory('.')

    output_file = Path(output_dir) / 'documents.jsonl.gz'
    document_snippets.to_json(output_file, lines=True, orient='records')
