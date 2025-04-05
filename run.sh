#!/bin/bash

# Default mode is local (CPU) and HTTP
MODE="cpu"
DEPLOY_MODE="local"  # "cloud" for deployment on GCE

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            MODE="gpu"
            shift
            ;;
        --deploy)
            DEPLOY_MODE="cloud"
            shift
            ;;
        --help)
            echo "Usage: ./run.sh [--gpu] [--deploy]"
            echo ""
            echo "Options:"
            echo "  --gpu     Run with GPU support (requires NVIDIA Container Toolkit)"
            echo "  --deploy  Build, push, and deploy container to Google Cloud (requires gcloud)"
            echo "  --help    Show this help message"
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
    echo "Please edit .env file with your API keys and cloud configuration variables."
    exit 1
fi

# Load environment variables from .env
set -o allexport
source .env
set +o allexport

# For cloud deployment, ensure required variables are set
if [ "$DEPLOY_MODE" == "cloud" ]; then
    for var in PROJECT_ID REGION REPO IMAGE_NAME INSTANCE_NAME; do
        if [ -z "${!var}" ]; then
            echo "Error: Environment variable $var must be set in .env for cloud deployment."
            exit 1
        fi
    done
fi

# Check for Docker and Docker Compose
if ! command -v docker &> /dev/null || ! command -v docker-compose &> /dev/null; then
    echo "Docker and/or Docker Compose not found! Please install them first."
    exit 1
fi

# For cloud deployment, also require gcloud
if [ "$DEPLOY_MODE" == "cloud" ] && ! command -v gcloud &> /dev/null; then
    echo "gcloud CLI not found! Please install the Google Cloud SDK."
    exit 1
fi

# Create necessary directories
mkdir -p models
mkdir -p temp

if [ "$DEPLOY_MODE" == "local" ]; then
    # Local run using docker-compose
    if [ "$MODE" == "gpu" ]; then
        echo "Building and starting Satoshi AI container with GPU support locally..."
        if ! command -v nvidia-smi &> /dev/null; then
            echo "Warning: nvidia-smi not found. Ensure your NVIDIA drivers are installed."
            echo "Continuing anyway..."
        fi
        sudo docker-compose --profile gpu up -d --build
    else
        echo "Building and starting Satoshi AI container (CPU mode) locally..."
        sudo docker-compose up -d --build
    fi

    if [ $? -eq 0 ]; then
        echo "Container started successfully!"
        echo "API is available at http://localhost:8080"
        echo "Run 'sudo docker-compose logs -f' to view logs"
    else
        echo "Failed to start container. Please check logs with 'sudo docker-compose logs'"
        exit 1
    fi

else
    # Cloud deployment: build image, push to Artifact Registry, and create a GCE instance

    echo "Building Docker image for cloud deployment..."
    sudo docker build -t "$IMAGE_NAME:latest" .

    # Tag the image with Artifact Registry URL.
    # Expected format: REGION-docker.pkg.dev/PROJECT_ID/REPO/IMAGE_NAME:latest
    CLOUD_IMAGE_TAG="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:latest"
    sudo docker tag "$IMAGE_NAME:latest" "$CLOUD_IMAGE_TAG"

    echo "Authenticating Docker with Artifact Registry..."
    gcloud auth configure-docker "${REGION}-docker.pkg.dev" || {
        echo "Failed to configure Docker authentication with gcloud."
        exit 1
    }

    echo "Pushing image to Artifact Registry..."
    sudo docker push "$CLOUD_IMAGE_TAG" || {
        echo "Image push failed."
        exit 1
    }

    echo "Ensuring firewall rule for HTTPS (TCP:443) exists..."
    # Create a firewall rule to allow HTTPS traffic on port 443 if it doesn't exist
    gcloud compute firewall-rules create allow-https \
      --allow tcp:443 \
      --target-tags=http-server \
      --quiet || echo "Firewall rule 'allow-https' already exists or was updated."

    echo "Deploying container to Google Compute Engine..."
    # Create a GCE instance that runs the container.
    # We use preemptible instance with an NVIDIA T4 GPU.
    # Note: The container must listen on port 443 to serve HTTPS.
    gcloud compute instances create-with-container "${INSTANCE_NAME}" \
      --zone="${REGION}-a" \
      --accelerator=type=nvidia-tesla-t4,count=1 \
      --metadata=install-nvidia-driver=True \
      --preemptible \
      --container-image="$CLOUD_IMAGE_TAG" \
      --container-ports=443 \
      --tags=http-server || {
          echo "Failed to create GCE instance."
          exit 1
      }

    echo "Deployment successful!"
    # Retrieve the external IP of the instance
    EXTERNAL_IP=$(gcloud compute instances describe "${INSTANCE_NAME}" --zone="${REGION}-a" --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
    echo "Your application is available via HTTPS at: https://${EXTERNAL_IP}"
    echo "Use 'gcloud compute ssh ${INSTANCE_NAME} --zone=${REGION}-a' to access the instance if needed."
fi
