FROM ubuntu:18.04

RUN apt-get update
RUN apt-get install -y python3-pip

WORKDIR /home/app

COPY requirements.txt ./
COPY src/app/web.py src/app/
COPY src/app/core src/app/core
COPY src/resources/production src/resources/production
COPY src/resources/pretrained/dp-fasttext.bin src/resources/pretrained/

RUN pip3 install -U virtualenv
RUN virtualenv --system-site-packages -p python3 ./.venv
RUN ./.venv/bin/pip install --upgrade pip
RUN ./.venv/bin/pip install --upgrade -r requirements.txt

EXPOSE 5000

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENTRYPOINT ./.venv/bin/python3.6 src/app/web.py
