#!/bin/bash

# Default mode is CPU
MODE="cpu"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            MODE="gpu"
            shift
            ;;
        --help)
            echo "Usage: ./run.sh [--gpu]"
            echo ""
            echo "Options:"
            echo "  --gpu    Run with GPU support (requires NVIDIA Container Toolkit)"
            echo "  --help   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Creating .env file from example..."
    cp .env.example .env
    echo "Please edit .env file with your API keys"
    exit 1
fi

# Check for Docker and Docker Compose
if ! command -v docker &> /dev/null || ! command -v docker-compose &> /dev/null; then
    echo "Docker and/or Docker Compose not found! Please install them first."
    exit 1
fi

# Create necessary directories
mkdir -p models
mkdir -p temp

# Build and start the container
if [ "$MODE" == "gpu" ]; then
    echo "Building and starting Satoshi AI container with GPU support..."
    
    # Check for nvidia-smi to ensure GPU is available
    if ! command -v nvidia-smi &> /dev/null; then
        echo "Warning: nvidia-smi not found. Make sure your NVIDIA drivers are installed properly."
        echo "Continuing anyway, but the container might not work correctly."
    fi
    
    # Run with GPU profile
    docker-compose --profile gpu up -d --build
else
    echo "Building and starting Satoshi AI container (CPU mode)..."
    docker-compose up -d --build
fi

# Check if container is running
if [ $? -eq 0 ]; then
    echo "Container started successfully!"
    echo "API is available at http://localhost:8080"
    echo "Run 'docker-compose logs -f' to view logs"
else
    echo "Failed to start container. Please check logs with 'docker-compose logs'"
    exit 1
fi 