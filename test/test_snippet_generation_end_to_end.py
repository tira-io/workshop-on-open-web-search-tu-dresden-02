import unittest
from approvaltests import verify_as_json
import json
from src.snippet_generation import find_top_snippets


class EndToEndSnippetGeneratorTest(unittest.TestCase):
    def test_top_snippet_asserts_non_empty_query(self):
        query_doc_pair = json.load(open('test/test-rerank-example-01.json'))
        document = query_doc_pair['text']
        query = " \n  \n "
        actual = find_top_snippets(query, document, ranker='PL2', max_snippets=3, use_crossencoder=False)
        verify_as_json(actual)

    def test_top_snippets_example_01_pl2(self):
        query_doc_pair = json.load(open('test/test-rerank-example-01.json'))
        document = query_doc_pair['text']
        query = query_doc_pair['query']
        actual = find_top_snippets(query, document, ranker='PL2', max_snippets=3, use_crossencoder=False)
        verify_as_json(actual)

    def test_top_snippets_example_01_tf(self):
        query_doc_pair = json.load(open('test/test-rerank-example-01.json'))
        document = query_doc_pair['text']
        query = query_doc_pair['query']
        actual = find_top_snippets(query, document, ranker='Tf', max_snippets=3, use_crossencoder=False)
        verify_as_json(actual) 


    def test_top_snippets_example_02_pl2(self):
        query_doc_pair = json.load(open('test/test-rerank-example-02.json'))
        document = query_doc_pair['text']
        query = query_doc_pair['query']
        actual = find_top_snippets(query, document, ranker='PL2', max_snippets=30, use_crossencoder=False)
        verify_as_json(actual)


    def test_top_snippets_example_02_tf(self):
        query_doc_pair = json.load(open('test/test-rerank-example-02.json'))
        document = query_doc_pair['text']
        query = query_doc_pair['query']
        actual = find_top_snippets(query, document, ranker='Tf', max_snippets=30, use_crossencoder=False)
        verify_as_json(actual)



    def test_top_snippets_example_03_pl2(self):
        query_doc_pair = json.load(open('test/test-rerank-example-03.json'))
        document = query_doc_pair['text']
        query = query_doc_pair['query']
        actual = find_top_snippets(query, document, ranker='PL2', max_snippets=30, use_crossencoder=False)
        verify_as_json(actual)


    def test_top_snippets_example_03_tf(self):
        query_doc_pair = json.load(open('test/test-rerank-example-03.json'))
        document = query_doc_pair['text']
        query = query_doc_pair['query']
        actual = find_top_snippets(query, document, ranker='Tf', max_snippets=30, use_crossencoder=False)
        verify_as_json(actual)


    def test_top_snippets_example_04_pl2(self):
        query_doc_pair = json.load(open('test/test-rerank-example-04.json'))
        document = query_doc_pair['text']
        query = query_doc_pair['query']
        actual = find_top_snippets(query, document, ranker='PL2', max_snippets=30, use_crossencoder=False)
        verify_as_json(actual)


    def test_top_snippets_example_04_tf(self):
        query_doc_pair = json.load(open('test/test-rerank-example-04.json'))
        document = query_doc_pair['text']
        query = query_doc_pair['query']
        actual = find_top_snippets(query, document, ranker='Tf', max_snippets=30, use_crossencoder=False)
        verify_as_json(actual)


if __name__ == '__main__':
    unittest.main()
