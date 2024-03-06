import unittest
import json
import gzip
from src.snippet_generation import find_top_snippets


class BigDataSnippetGeneratorTest(unittest.TestCase):
    def test_top_snippets_example_05_pl2(self):
        example_file = open('test/test-rerank-example-05.json', 'r')
        print('read complete')
        
        for line in example_file.readlines():
            data = json.loads(line)
            find_top_snippets(data['query'], data['text'], ranker='PL2', max_snippets=3, use_crossencoder=False)


if __name__ == '__main__':
    unittest.main()
