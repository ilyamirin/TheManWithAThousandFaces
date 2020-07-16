#!/bin/bash

set -e

echo "Running Train Budget Model Job..."
docker run -v $PWD/src/resources/production:/home/app/src/resources/production -v $PWD/src/ui/src/categorical/:/home/app/src/ui/src/categorical/ --rm fac-job src/app/train_budget_model.py
echo "├── Complete"
