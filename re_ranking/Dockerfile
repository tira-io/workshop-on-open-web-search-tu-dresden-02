# docker build -t bla re_ranking
#see https://github.com/OpenWebSearch/wows-code/tree/main/ecir24/baselines/dev-container
FROM webis/wows-2024-baseline:0.0.5

ENV PYTERRIER_VERSION='5.8'
ENV PYTERRIER_HELPER_VERSION='0.0.8'

RUN apt-get update \
	&& DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y git openjdk-11-jdk curl wget gcc \
	&& pip3 install python-terrier pandas jupyterlab runnb tira \
	&& python3 -c "import pyterrier as pt; pt.init(version='${PYTERRIER_VERSION}', helper_version='${PYTERRIER_HELPER_VERSION}');" \
	&& python3 -c "import pyterrier as pt; pt.init(version='${PYTERRIER_VERSION}', helper_version='${PYTERRIER_HELPER_VERSION}', boot_packages=['com.github.terrierteam:terrier-prf:-SNAPSHOT']);"

# install trec-cast-tools for chunking of documents
RUN cd /home && git clone https://github.com/grill-lab/trec-cast-tools.git
RUN pip install -r /home/trec-cast-tools/corpus_processing/requirements.txt
ENV PYTHONPATH "${PYTHONPATH}:/home/trec-cast-tools/corpus_processing/"

# install colbert
RUN pip install --upgrade git+https://github.com/terrierteam/pyterrier_colbert.git


ADD baseline_re_ranking.py /code/baseline_re_ranking.py

ENTRYPOINT [ "python3", "/code/baseline_re_ranking.py" ]
