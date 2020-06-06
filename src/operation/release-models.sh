#!/bin/bash

set -e

echo "Releasing models..."

git add src/resources/production/*
git commit -m "Release models ($(date +'%d-%m-%Y %H:%M'))"
git push

echo "├── Complete"
