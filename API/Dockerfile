ARG PYTHON_VERSION=3.10

FROM docker.io/python:${PYTHON_VERSION}-slim-bookworm

RUN apt-get update \
    && apt-get -y upgrade \
    && apt-get --no-install-recommends -y install \
       build-essential libgdal-dev libboost-numpy-dev

COPY requirements.txt requirements.txt

RUN \
   python3 -m pip install --upgrade pip \
   && python3 -m pip install -r requirements.txt

WORKDIR /app

COPY main.py /app/main.py

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
