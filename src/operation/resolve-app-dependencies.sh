cd $(dirname $0)/../..

if [ ! -d src/resources/embedding/rubert ]; then
    echo "
    Downloading BERT embedding model...
    "

    wget "http://files.deeppavlov.ai/deeppavlov_data/bert/rubert_cased_L-12_H-768_A-12_pt.tar.gz"
    mkdir src/resources/embedding
    tar -xvzf rubert_cased_L-12_H-768_A-12_pt.tar.gz -C src/resources/embedding/
    mv src/resources/embedding/rubert_cased_L-12_H-768_A-12_pt src/resources/embedding/rubert
    mv src/resources/embedding/rubert/bert_config.json src/resources/embedding/rubert/config.json
    rm rubert_cased_L-12_H-768_A-12_pt.tar.gz

    echo "
    ├── Complete
    "
fi

if [ ! -f src/resources/embedding/dp-fasttext.bin ]; then
    echo "
    Downloading fastText embedding model...
    "

    wget "http://files.deeppavlov.ai/embeddings/ft_native_300_ru_wiki_lenta_lower_case/ft_native_300_ru_wiki_lenta_lower_case.bin"
    mkdir src/resources/embedding
    mv ft_native_300_ru_wiki_lenta_lower_case.bin src/resources/embedding/dp-fasttext.bin

    echo "
    ├── Complete
    "
fi

if [ ! -d .venv ]; then
    echo "
    Resolve model (python) dependencies...
    "
    
    virtualenv --system-site-packages -p python3 ./.venv
    ./.venv/bin/pip install --upgrade pip
    ./.venv/bin/pip install --upgrade -r requirements.txt

    echo "
    ├── Complete
    "
fi

# if [ ! -d ui/node_modules ]; then
#     echo "
#     Initializing UI...
#     "

#     cd ui
#     npm install
# fi
