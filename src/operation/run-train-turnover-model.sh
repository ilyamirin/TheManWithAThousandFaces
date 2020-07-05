#!/bin/bash

set -e

echo "Running Train Turnover Model Job..."
docker run -v $PWD/src/resources/production:/home/app/src/resources/production --rm fac-job src/app/train_turnover_model.py
echo "├── Complete"
