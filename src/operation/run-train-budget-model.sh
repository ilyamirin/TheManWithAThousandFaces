#!/bin/bash

echo "Running Train Budget Model Job..."
    docker run -v src/resources/production:/home/app/src/resources/production --rm fac-train-budget-model-job
    echo "├── Complete"
