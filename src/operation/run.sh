#!/bin/bash

set -e

if [ ! "$(docker ps -f "name=fac-app" --format '{{.Names}}')" ]; then
    echo "Running App on port ${APP_PORT}..."
    docker run -p ${APP_PORT}:5000 -d --name fac-app fac-app
    echo "├── Complete"
else
    echo "App is already running"
fi

if [ ! "$(docker ps -f "name=fac-ui" --format '{{.Names}}')" ]; then
    echo "Running UI on port ${UI_PORT}..."
    docker run -p ${UI_PORT}:80 -d --name fac-ui fac-ui
    echo "├── Complete"
else
    echo "UI is already running"
fi
