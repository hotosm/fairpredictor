ARG PYTHON_VERSION=3.10

FROM docker.io/python:${PYTHON_VERSION}-slim-bookworm

RUN apt-get update \
    && apt-get -y upgrade \
    && apt-get --no-install-recommends -y install \
       build-essential libgdal-dev libboost-numpy-dev

RUN \
   python3 -m pip install --upgrade pip \
   && python3 -m pip install \
        fastapi==0.103.2 \
        uvicorn==0.22.0 \
        fairpredictor \
        tflite-runtime==2.14.0

RUN pip install fastapi==0.103.2 fairpredictor tflite-runtime==2.14.0

WORKDIR /app

COPY main.py /app/main.py

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
