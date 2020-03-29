#!/bin/bash

if [ "$(docker ps -f "name=fac-app" --format '{{.Names}}')" == "fac-app" ]; then
    echo "Stopping App..."
    docker rm -f fac-app
else
    echo "App already stopped"
fi

if [ "$(docker ps -f "name=fac-ui" --format '{{.Names}}')" == "fac-ui" ]; then
    echo "Stopping UI..."
    docker rm -f fac-ui
else
    echo "UI already stopped"
fi
