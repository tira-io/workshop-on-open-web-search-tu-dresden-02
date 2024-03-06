import unittest
from approvaltests import verify_as_json
import json
from src.snippet_generation import find_top_snippets_for_all_documents


class EndToEndSnippetGeneratorTest(unittest.TestCase):
    def test_top_snippet_asserts_non_empty_query(self):
        query_doc_pair = json.load(open('test/test-rerank-example-01.json'))
        document = query_doc_pair['text']
        query = " \n  \n "
        qid = '1'
        documents  = [{'body': 'fghjklljhjgh' , 'id':1}]
        actual = find_top_snippets_for_all_documents(qid, query, documents, ranker='PL2', use_crossencoder=False)
        verify_as_json(actual)

    def test_top_snippet_asserts_non_empty_document(self):
        query_doc_pair = json.load(open('test/test-rerank-example-01.json'))
        document = " \n  \n "
        query = query_doc_pair['query']
        qid = query_doc_pair['qid']
        documents  = [{'docno': '1' , 'contents': [{'contents':'fghj'}]}]
        actual = find_top_snippets_for_all_documents(qid, query, documents, ranker='PL2', use_crossencoder=False)
        verify_as_json(actual)

    def test_top_snippets_example_01_pl2(self):
        query_doc_pair = json.load(open('test/test-rerank-example-01.json'))
        document = {'docno': '1', 'text': query_doc_pair['text']}
        query = query_doc_pair['query']
        qid = query_doc_pair['qid']
        actual = find_top_snippets_for_all_documents(qid, query, documents, ranker='PL2', use_crossencoder=False)
        verify_as_json(actual)

    def test_top_snippets_example_01_tf(self):
        query_doc_pair = json.load(open('test/test-rerank-example-01.json'))
        document = query_doc_pair['text']
        query = query_doc_pair['query']
        qid = query_doc_pair['qid']
        actual = find_top_snippets_for_all_documents(qid, query, documents, ranker='Tf', use_crossencoder=False)
        verify_as_json(actual) 


    def test_top_snippets_example_02_pl2(self):
        query_doc_pair = json.load(open('test/test-rerank-example-02.json'))
        document = query_doc_pair['text']
        query = query_doc_pair['query']
        qid = query_doc_pair['qid']
        actual = find_top_snippets_for_all_documents(qid, query, document, ranker='PL2', use_crossencoder=False)
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
