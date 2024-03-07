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

    def run_sample(self, path_to_data: str, wmodel: str, cross_encode: bool) -> list[dict]:
        data = pd.read_json(path_to_data, lines=True)
        preprocessed_docs = split_dataframe_into_snippets(data)
        snippets = []
        for qid, i in preprocessed_docs.items():
            snippets += find_top_snippets_for_all_documents(qid, i['query'], i['documents'], wmodel=wmodel, cross_encode=cross_encode)
        return snippets


    def test_top_snippets_example_02_pl2(self):
        actual = self.run_sample('test/test-rerank-example-02.json', wmodel='PL2', cross_encode=False)
        verify_as_json(actual)


    def test_top_snippets_example_02_tf(self):
        actual = self.run_sample('test/test-rerank-example-02.json', wmodel='Tf', cross_encode=False)
        verify_as_json(actual)


    def test_top_snippets_example_02_bm25(self):
        actual = self.run_sample('test/test-rerank-example-02.json', wmodel='BM25', cross_encode=False)
        verify_as_json(actual)


    def test_top_snippets_example_02_pl2_cross_encoder(self):
        actual = self.run_sample('test/test-rerank-example-02.json', wmodel='PL2', cross_encode=True)
        verify_as_json(actual)


    def test_top_snippets_example_02_tf_cross_encoder(self):
        actual = self.run_sample('test/test-rerank-example-02.json', wmodel='Tf', cross_encode=True)
        verify_as_json(actual)


    def test_top_snippets_example_02_bm25_cross_encoder(self):
        actual = self.run_sample('test/test-rerank-example-02.json', wmodel='BM25', cross_encode=True)
        verify_as_json(actual)


if __name__ == '__main__':
    unittest.main()
