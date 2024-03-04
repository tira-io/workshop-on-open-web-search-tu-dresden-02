#docker build -t webis/wows-2024-baseline:0.0.1 .
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

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


ADD baseline_re_ranking.py /code/baseline_re_ranking.py

ENTRYPOINT [ "python3", "/code/baseline_re_ranking.py" ]
