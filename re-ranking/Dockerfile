#see https://github.com/OpenWebSearch/wows-code/tree/main/ecir24/baselines/dev-container
FROM webis/wows-2024-baseline:0.0.5

ADD baseline_re_ranking.py /code/baseline_re_ranking.py

# add trec-cast chunker
# https://github.com/grill-lab/trec-cast-tools
RUN mkdir -p /trec-cast \
    && cd /trec-cast \
    && git clone https://github.com/grill-lab/trec-cast-tools.git \
    && pip install -r trec-cast-tools/corpus_processing/requirements.txt
ENV PYTHONPATH "${PYTHONPATH}:/trec-cast/trec-cast-tools/corpus_processing"

ENTRYPOINT [ "python3", "/code/baseline_re_ranking.py" ]
