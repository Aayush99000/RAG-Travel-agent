#!/bin/bash

echo "Building Docker image..."
docker build -t travel-app .

echo "Running container on port 8080..."
docker run -p 8080:8080 travel-app

echo "Access at http://localhost:8080"