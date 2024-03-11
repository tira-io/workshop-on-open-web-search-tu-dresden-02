# Re-Ranking using Cross-Encoders and Smart Snippets

---

When documents exceed the length cross-encoders can process, typically only the first _m_ tokens are considered, where _m_ represents the processing limit of the cross-encoder along with the query.
This naive method may lead to an unfair ranking, i.e., documents are treated less favorably when their most important section (in relation to the query) starts after the first _m_ tokens.
Therefore, we propose the extraction of _smart snippets_, i.e., document passages that represent the content of a document in relation to a query.
Smart snippets can be used instead of the first _m_ tokens of a document for fairer ranking using cross-encoders.

---

## Method

The re-ranking process with smart snippets consists of five steps.

First, we subdivide all documents into snippets.
The maximum length of those snippets may be chosen arbitrarily--we defaulted to 250 tokens which is the passage size used in [TREC CAsT](http://dx.doi.org/10.48550/arXiv.2003.13624).
The actual length of the snippets may vary since the division process aims to retain context by not separating sentences.

In step 2, we need to pre-rank all extracted snippets in relation to the query.
To accomplish this, we view the set of all snippets as a corpus.
From this corpus, we can create a ranking for the query using one of the following weighting models:
* term frequency (Tf),
* BM25, or
* PL2.

We do not use cross-encoders for the pre-ranking of documents, because there may be a multitude of snippets per document depending on document length and therefore ranking all snippets using a cross-encoder can drastically slow down the re-ranking process.

In step 3, we can obtain the top _k_ relevant snippets of each document from the pre-ranking, which are later ranked using a cross-encoder.
This step ensures that the cross-encoder only needs to rank _n_ times _k_ snippets for _n_ documents instead of all snippets.
In order to reduce computational cost, we defaulted to _k = 3_.

In step 4, the top _k_ snippets of all documents are ranked using a cross-encoder.
That way, similar to step 2, we can more accurately deduce which snippets best match the query--but now the ranking is more precise since we used a cross-encoder instead of the simple weighting models used in step 2.
This final document ranking ensues from this snippet ranking in step 5, i.e., the document that provided the best snippet is ranked first.

---

## Usage

You can run it directly via (please install `tira` via `pip3 install tira`, Docker, and Python >= 3.7 on your machine): 

```shell
tira-run \
	--input-dataset workshop-on-open-web-search/re-ranking-20231027-training \
	--image mam10eks/wows-baselines:re-ranker-0.0.1
```

## Development

You can build the Docker image via:

```
docker build -t mam10eks/wows-baselines:re-ranker-0.0.1 .
```

To publish the image to dockerhub, run:

```
docker push mam10eks/wows-baselines:re-ranker-0.0.1
```
