import unittest
import pandas as pd
from approvaltests import verify_as_json
import json
from src.snippet_generation import find_top_snippets_for_all_documents, split_dataframe_into_snippets


class EndToEndSnippetGeneratorTest(unittest.TestCase):
    def test_top_snippet_asserts_non_empty_query(self):
        query = " \n  \n "
        qid = '1'
        documents  = pd.DataFrame([{'docno': 'docno', 'contents': 'fghjklljhjgh' , 'qid': qid, 'query': query}])
        documents = split_dataframe_into_snippets(documents)
        actual = find_top_snippets_for_all_documents(qid, query, documents['1']['documents'], wmodel='PL2', cross_encode=False)
        verify_as_json(actual)

    def test_top_snippet_asserts_non_empty_document(self):
        query = "sajfasd"
        qid = '1'
        documents  = pd.DataFrame([{'docno': 'docno', 'contents': ' \n \n ' , 'qid': qid, 'query': query}])
        actual = split_dataframe_into_snippets(documents)
        verify_as_json(actual)

    def test_top_snippets_example_02_pl2(self):
        data = pd.read_json('test/test-rerank-example-02.json', lines=True)
        qid = data[0]['qid']
        query = data[0]['query']
        documents = data.drop(columns=['qid', 'query', 'original_query', 'original_document']).to_dict()
        actual = find_top_snippets_for_all_documents(qid=qid, query=query, documents=documents, wmodel='PL2', cross_encode=False)
        verify_as_json(actual)


    def test_top_snippets_example_02_tf(self):
        query_doc_pair = json.load(open('test/test-rerank-example-02.json'))
        document = query_doc_pair['text']
        query = query_doc_pair['query']
        qid = query_doc_pair['qid']
        actual = find_top_snippets_for_all_documents(qid, query, document, ranker='Tf', use_crossencoder=False)
        verify_as_json(actual)



    def test_top_snippets_example_03_pl2(self):
        query_doc_pair = json.load(open('test/test-rerank-example-03.json'))
        document = query_doc_pair['text']
        query = query_doc_pair['query']
        qid = query_doc_pair['qid']
        actual = find_top_snippets_for_all_documents(qid, query, document, ranker='PL2', max_snippets=30, use_crossencoder=False)
        verify_as_json(actual)


    def test_top_snippets_example_03_tf(self):
        query_doc_pair = json.load(open('test/test-rerank-example-03.json'))
        document = query_doc_pair['text']
        query = query_doc_pair['query']
        qid = query_doc_pair['qid']
        actual = find_top_snippets_for_all_documents(qid, query, document, ranker='Tf', max_snippets=30, use_crossencoder=False)
        verify_as_json(actual)


    def test_top_snippets_example_04_pl2(self):
        query_doc_pair = json.load(open('test/test-rerank-example-04.json'))
        document = query_doc_pair['text']
        query = query_doc_pair['query']
        qid = query_doc_pair['qid']
        actual = find_top_snippets_for_all_documents(qid, query, document, ranker='PL2', max_snippets=30, use_crossencoder=False)
        verify_as_json(actual)


    def test_top_snippets_example_04_tf(self):
        query_doc_pair = json.load(open('test/test-rerank-example-04.json'))
        document = query_doc_pair['text']
        query = query_doc_pair['query']
        qid = query_doc_pair['qid']
        actual = find_top_snippets_for_all_documents(qid, query, document, ranker='Tf', max_snippets=30, use_crossencoder=False)
        verify_as_json(actual)


if __name__ == '__main__':
    unittest.main()
