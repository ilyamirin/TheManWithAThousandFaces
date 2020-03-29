#!/bin/bash

if [ ! "$(docker ps -f "name=fac-app" --format '{{.Names}}')" ]; then
    echo "Running App on port ${APP_PORT}..."
    docker run -p ${APP_PORT}:5000 -d --rm --name fac-app fac-app
    echo "├── Complete"
else
    echo "App is already running"
fi
