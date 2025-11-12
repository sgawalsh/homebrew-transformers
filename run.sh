#!/bin/bash
# Build the image
docker build -t transformers .

# Run the container with mounted directories
docker run \
    -v "${PWD}/checkpoints:/checkpoints" \
    -v "${PWD}/data:/data" \
    transformers