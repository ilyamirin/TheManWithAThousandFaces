#!/bin/bash

echo "Running Train Budget Model Job..."
docker run -v $PWD/src/resources/production:/home/app/src/resources/production --rm fac-job src/app/train_budget_model.py
echo "├── Complete"
