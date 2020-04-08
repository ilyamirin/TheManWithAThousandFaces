cd $(dirname $0)/../..

if ! [ -x "$(command -v virtualenv)" ]; then
    echo "Installing python virtual environment..."
    
    sudo apt install -y python3-pip
    sudo pip3 install -U virtualenv

    echo "├── Complete"
fi

if ! [ -x "$(command -v npm)" ]; then
    echo "Installing NPM..."

    sudo apt install -y npm
    sudo npm install -g npm
    hash -d npm

    echo "├── Complete"
fi

if [ ! -d src/resources/pretrained/rubert ]; then
    echo "Downloading BERT embedding model..."

    wget "http://files.deeppavlov.ai/deeppavlov_data/bert/rubert_cased_L-12_H-768_A-12_pt.tar.gz"
    mkdir src/resources/pretrained
    tar -xvzf rubert_cased_L-12_H-768_A-12_pt.tar.gz -C src/resources/pretrained/
    mv src/resources/pretrained/rubert_cased_L-12_H-768_A-12_pt src/resources/pretrained/rubert
    mv src/resources/pretrained/rubert/bert_config.json src/resources/pretrained/rubert/config.json
    rm rubert_cased_L-12_H-768_A-12_pt.tar.gz

    echo "├── Complete"
fi

if [ ! -f src/resources/pretrained/dp-fasttext.bin ]; then
    echo "Downloading fastText embedding model..."

    wget "http://files.deeppavlov.ai/embeddings/ft_native_300_ru_wiki_lenta_lower_case/ft_native_300_ru_wiki_lenta_lower_case.bin"
    mkdir src/resources/pretrained
    mv ft_native_300_ru_wiki_lenta_lower_case.bin src/resources/pretrained/dp-fasttext.bin

    echo "├── Complete"
fi

if [ ! -d .venv ]; then
    echo "Resolve model (python) dependencies..."
    
    virtualenv --system-site-packages -p python3 ./.venv
    ./.venv/bin/pip install --upgrade pip
    ./.venv/bin/pip install --upgrade -r requirements.txt

    echo "├── Complete"
fi

if [ ! -d src/ui/node_modules ]; then
    echo "Resolve UI (js) dependencies..."

    cd src/ui
    npm install

    echo "├── Complete"
fi
