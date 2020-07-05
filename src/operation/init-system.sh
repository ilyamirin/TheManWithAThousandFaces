#!/bin/bash

set -e

if ! [ -x "$(command -v npm)" ] || ! [ -x "$(command -v docker)" ]; then
    echo "Updating system..."
    
    sudo apt update

    echo "├── Complete"
fi

 if ! [ -x "$(command -v make)" ]; then
    echo "Installing make..."
    
    sudo apt install make

    echo "├── Complete"
fi

if ! [ -x "$(command -v docker)" ]; then
    echo "Installing docker..."

    sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu bionic stable"
    sudo apt update
    sudo apt install -y docker-ce
    sudo usermod -aG docker ${USER}

    echo "├── Complete"
fi
