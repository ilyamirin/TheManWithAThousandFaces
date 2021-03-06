#!/bin/bash

set -e

cd $(dirname $0)/../..

echo "Building App..."
docker build -t fac-app -f App.Dockerfile .
echo "├── Complete"

echo "Building UI..."
docker build --build-arg app_api_url=${REACT_APP_API_URL} -t fac-ui -f UI.Dockerfile .
echo "├── Complete"

echo "Building job image..."
docker build -t fac-job -f Job.Dockerfile .
echo "├── Complete"
