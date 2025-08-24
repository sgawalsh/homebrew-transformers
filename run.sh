#!/bin/bash
docker build -t homebrew-transformers .
docker run --rm homebrew-transformers