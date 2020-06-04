#!/bin/bash

echo "Running Train Turnover Model Job..."
    docker run -v $PWD/src/resources/production:/home/app/src/resources/production --rm fac-train-turnover-model-job
    echo "├── Complete"
