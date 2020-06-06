#!/bin/bash

set -e

realpath $1

echo "Running Supplement Dataset Job..."
docker run -v $PWD/src/resources/production:/home/app/src/resources/production -v `realpath $1`:/tmp/patch.csv --rm fac-job src/app/supplement_dataset.py /tmp/patch.csv
echo "├── Complete"
