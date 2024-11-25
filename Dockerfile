ARG PYTHON_VERSION=3.10

FROM docker.io/python:${PYTHON_VERSION}-slim-bookworm

RUN apt-get update \
    && apt-get -y upgrade \
    && apt-get --no-install-recommends -y install \
    build-essential libgdal-dev libboost-numpy-dev

COPY API/requirements.txt api-requirements.txt

RUN \
    python3 -m pip install --upgrade pip \
    && python3 -m pip install -r api-requirements.txt


COPY predictor /app/predictor
COPY setup.py /app/setup.py
COPY README.md /app/README.md


WORKDIR /app

RUN python3 setup.py install 

COPY API/main.py /app/main.py

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]