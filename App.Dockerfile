FROM ubuntu:18.04

RUN apt-get update

# Download pretrained fastText
RUN apt-get install -y wget

RUN wget "http://files.deeppavlov.ai/embeddings/ft_native_300_ru_wiki_lenta_lower_case/ft_native_300_ru_wiki_lenta_lower_case.bin"

# Init google drive
RUN apt-get install -y curl
RUN apt-get install -y unzip

RUN curl https://rclone.org/install.sh | bash
COPY .rclone.conf /root/.config/rclone/rclone.conf

WORKDIR /home/app

# Resolve app dependencies
RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install -U virtualenv

COPY requirements.txt ./
RUN virtualenv --system-site-packages -p python3 ./.venv
RUN ./.venv/bin/pip install --upgrade pip
RUN ./.venv/bin/pip install --upgrade -r requirements.txt

# Copy app sources
COPY src/app/web.py src/app/
COPY src/app/core src/app/core
RUN mkdir -p src/resources/pretrained && mv /ft_native_300_ru_wiki_lenta_lower_case.bin src/resources/pretrained/dp-fasttext.bin

EXPOSE 5000

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Disable docker build cache for next commands
ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache

# Download trained model from Google Drive
RUN rclone sync KeterideDrive:Financial-Analytics-Classifier/resources src/resources/production

ENTRYPOINT ./.venv/bin/python3.6 src/app/web.py
