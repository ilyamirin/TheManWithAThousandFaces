#!/bin/bash

if [ "$(docker ps -f "name=fac-app" --format '{{.Names}}')" == "fac-app" ]; then
    echo "Stopping App..."
    docker rm -f fac-app
else
    echo "App already stopped"
fi
