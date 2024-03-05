import unittest
from approvaltests import verify_as_json
import json
from src.snippet_generation import find_top_snippets


class EndToEndSnippetGeneratorTest(unittest.TestCase):
    def test_top_snippets_example_01_pl2(self):
        query_doc_pair = json.load(open('test/test-rerank-example-01.json'))
        document = query_doc_pair['text']
        query = query_doc_pair['query']
        actual = find_top_snippets(query, document, 'PL2', 3)
        verify_as_json(actual)

    def test_top_snippets_example_01_tf(self):
        query_doc_pair = json.load(open('test/test-rerank-example-01.json'))
        document = query_doc_pair['text']
        query = query_doc_pair['query']
        actual = find_top_snippets(query, document, 'Tf', 3)
        verify_as_json(actual) 


    def test_top_snippets_example_02_pl2(self):
        query_doc_pair = json.load(open('test/test-rerank-example-02.json'))
        document = query_doc_pair['text']
        query = query_doc_pair['query']
        actual = find_top_snippets(query, document, 'PL2', 30)
        verify_as_json(actual)


    def test_top_snippets_example_02_tf(self):
        query_doc_pair = json.load(open('test/test-rerank-example-02.json'))
        document = query_doc_pair['text']
        query = query_doc_pair['query']
        actual = find_top_snippets(query, document, 'Tf', 30)
        verify_as_json(actual)



    def test_top_snippets_example_03_pl2(self):
        query_doc_pair = json.load(open('test/test-rerank-example-03.json'))
        document = query_doc_pair['text']
        query = query_doc_pair['query']
        actual = find_top_snippets(query, document, 'PL2', 30)
        verify_as_json(actual)


    def test_top_snippets_example_03_tf(self):
        query_doc_pair = json.load(open('test/test-rerank-example-03.json'))
        document = query_doc_pair['text']
        query = query_doc_pair['query']
        actual = find_top_snippets(query, document, 'Tf', 30)
        verify_as_json(actual)


    def test_top_snippets_example_04_pl2(self):
        query_doc_pair = json.load(open('test/test-rerank-example-04.json'))
        document = query_doc_pair['text']
        query = query_doc_pair['query']
        actual = find_top_snippets(query, document, 'PL2', 30)
        verify_as_json(actual)


    def test_top_snippets_example_04_tf(self):
        query_doc_pair = json.load(open('test/test-rerank-example-04.json'))
        document = query_doc_pair['text']
        query = query_doc_pair['query']
        actual = find_top_snippets(query, document, 'Tf', 30)
        verify_as_json(actual)


if __name__ == '__main__':
    unittest.main()
