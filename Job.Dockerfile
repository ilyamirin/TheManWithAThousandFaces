FROM ubuntu:18.04

RUN apt-get update

# Download pretrained fastText
RUN apt-get install -y wget

RUN wget "http://files.deeppavlov.ai/embeddings/ft_native_300_ru_wiki_lenta_lower_case/ft_native_300_ru_wiki_lenta_lower_case.bin"

WORKDIR /home/app

# Resolve app dependencies
RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install -U virtualenv

COPY requirements.txt ./
RUN virtualenv --system-site-packages -p python3 ./.venv
RUN ./.venv/bin/pip install --upgrade pip
RUN ./.venv/bin/pip install --upgrade -r requirements.txt

# Copy app sources
RUN mkdir -p src/resources/pretrained && mv /ft_native_300_ru_wiki_lenta_lower_case.bin src/resources/pretrained/dp-fasttext.bin
COPY src/app src/app

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENTRYPOINT [".venv/bin/python3.6"]
