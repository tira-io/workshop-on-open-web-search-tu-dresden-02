#!/usr/bin/bash

cd /code
git clone https://github.com/grill-lab/trec-cast-tools.git
pip install -r trec-cast-tools/corpus_processing/requirements.txt


# ENV PYTHONPATH "${PYTHONPATH}:/trec-cast/trec-cast-tools/corpus_processing"