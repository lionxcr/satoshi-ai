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

    # Optional: Domain name for SSL certificates
    if [ ! -z "$DOMAIN_NAME" ]; then
        echo "Domain name detected. Will set up SSL certificates."
    fi

    # Check that an active account is selected
    ACTIVE_ACCOUNT=$(gcloud config get-value account)
    if [ -z "$ACTIVE_ACCOUNT" ]; then
        echo "No active gcloud account set. Please run: gcloud auth login"
        exit 1
    fi
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

    # Set up SSL certificates if domain is provided
    if [ ! -z "$DOMAIN_NAME" ]; then
        echo "Setting up Google Certificate Manager..."
        # Enable required APIs if not already enabled
        gcloud services enable certificatemanager.googleapis.com || {
            echo "Failed to enable Certificate Manager API."
            exit 1
        }

        # DNS authorization name based on domain
        DNS_AUTH_NAME="satoshi-ai-dns-auth"
        
        # Check if DNS authorization exists, create if it doesn't
        if ! gcloud certificate-manager dns-authorizations describe "$DNS_AUTH_NAME" --project="$PROJECT_ID" &>/dev/null; then
            echo "Creating DNS authorization for domain verification..."
            gcloud certificate-manager dns-authorizations create "$DNS_AUTH_NAME" \
                --domain="$DOMAIN_NAME" \
                --project="$PROJECT_ID" || {
                echo "Failed to create DNS authorization."
                exit 1
            }
            
            # Get the DNS record details that need to be created
            DNS_RECORD=$(gcloud certificate-manager dns-authorizations describe "$DNS_AUTH_NAME" \
                --project="$PROJECT_ID" \
                --format="value(dnsResourceRecord)")
            
            # Extract the data from DNS record (format: "{name} {type} {data}")
            DNS_RECORD_DATA=(${DNS_RECORD})
            RECORD_NAME=${DNS_RECORD_DATA[0]}
            RECORD_TYPE=${DNS_RECORD_DATA[1]}
            RECORD_VALUE=${DNS_RECORD_DATA[2]}
            
            echo "====================================================================="
            echo "IMPORTANT: Create the following DNS record in your domain registrar:"
            echo "Record name: $RECORD_NAME"
            echo "Record type: $RECORD_TYPE"
            echo "Record value: $RECORD_VALUE"
            echo "====================================================================="
            echo "After adding this DNS record, wait a few minutes for DNS propagation."
            
            # Ask user to confirm DNS record has been added
            read -p "Have you added the DNS record? (yes/no): " DNS_CONFIRMED
            if [[ "$DNS_CONFIRMED" != "yes" ]]; then
                echo "Please add the DNS record and run the script again."
                exit 1
            fi
        else
            echo "Using existing DNS authorization: $DNS_AUTH_NAME"
            
            # Show the DNS record again in case user needs it
            DNS_RECORD=$(gcloud certificate-manager dns-authorizations describe "$DNS_AUTH_NAME" \
                --project="$PROJECT_ID" \
                --format="value(dnsResourceRecord)")
            
            DNS_RECORD_DATA=(${DNS_RECORD})
            RECORD_NAME=${DNS_RECORD_DATA[0]}
            RECORD_TYPE=${DNS_RECORD_DATA[1]}
            RECORD_VALUE=${DNS_RECORD_DATA[2]}
            
            echo "DNS record for verification:"
            echo "Record name: $RECORD_NAME"
            echo "Record type: $RECORD_TYPE"
            echo "Record value: $RECORD_VALUE"
        fi

        # Check if certificate exists
        CERT_NAME="satoshi-ai-cert"
        if ! gcloud certificate-manager certificates describe "$CERT_NAME" --project="$PROJECT_ID" &>/dev/null; then
            echo "Creating Google-managed SSL certificate with DNS verification..."
            gcloud certificate-manager certificates create "$CERT_NAME" \
                --domains="$DOMAIN_NAME" \
                --dns-authorizations="$DNS_AUTH_NAME" \
                --project="$PROJECT_ID" || {
                echo "Failed to create certificate. Make sure DNS authorization is valid."
                exit 1
            }
            
            echo "Certificate creation initiated. This may take some time..."
            # Wait for certificate provisioning to complete (it's an async operation)
            for i in {1..30}; do
                CERT_STATE=$(gcloud certificate-manager certificates describe "$CERT_NAME" --project="$PROJECT_ID" --format="value(state)")
                if [ "$CERT_STATE" == "ACTIVE" ]; then
                    echo "Certificate is now active and ready to use."
                    break
                fi
                echo "Certificate provisioning in progress... (status: $CERT_STATE)"
                sleep 10
            done
            
            if [ "$CERT_STATE" != "ACTIVE" ]; then
                echo "Certificate did not become active in the allotted time."
                echo "You can check its status later with:"
                echo "gcloud certificate-manager certificates describe $CERT_NAME --project=$PROJECT_ID"
            fi
        else
            echo "Using existing certificate: $CERT_NAME"
        fi
        
        # Create secrets for certificate if they don't exist
        if ! gcloud secrets describe "ssl-cert" --project="$PROJECT_ID" &>/dev/null; then
            echo "Creating secret for SSL certificate..."
            gcloud secrets create "ssl-cert" --replication-policy="automatic" --project="$PROJECT_ID"
        fi
        
        if ! gcloud secrets describe "ssl-key" --project="$PROJECT_ID" &>/dev/null; then
            echo "Creating secret for SSL key..."
            gcloud secrets create "ssl-key" --replication-policy="automatic" --project="$PROJECT_ID"
        fi
        
        # Note: For managed certificates, the VM should have permission to access Google Certificate Manager
        echo "Note: For managed certificates, ensure the VM has permissions to access Google Certificate Manager."
    fi

    echo "Ensuring firewall rule for HTTPS (TCP:443) exists..."
    gcloud compute firewall-rules create allow-https \
      --allow tcp:443 \
      --target-tags=http-server \
      --quiet || echo "Firewall rule 'allow-https' already exists or was updated."

    echo "Checking if instance ${INSTANCE_NAME} already exists..."
    if gcloud compute instances describe "${INSTANCE_NAME}" --zone="${REGION}-a" &>/dev/null; then
        echo "Instance ${INSTANCE_NAME} already exists. Updating it with the new container image..."
        gcloud compute instances update-container "${INSTANCE_NAME}" \
          --zone="${REGION}-a" \
          --container-image="$CLOUD_IMAGE_TAG" || {
              echo "Failed to update GCE instance container."
              exit 1
          }
          
        # Update metadata if needed
        if [ ! -z "$DOMAIN_NAME" ]; then
            echo "Updating instance metadata..."
            METADATA="domain-name=$DOMAIN_NAME,cert-name=$CERT_NAME"
            gcloud compute instances add-metadata "${INSTANCE_NAME}" \
              --zone="${REGION}-a" \
              --metadata="$METADATA" || {
                  echo "Failed to update instance metadata."
                  # Not exiting here as this is not critical
              }
        fi
        
        echo "Instance updated successfully!"
    else
        echo "Creating new instance ${INSTANCE_NAME}..."
        # Create a GCE instance running the container
        if [ "$MODE" == "gpu" ]; then
            # GPU mode: include accelerator flag
            ACCEL_FLAGS="--accelerator=type=nvidia-tesla-t4,count=1 --metadata=install-nvidia-driver=True"
        else
            # CPU mode: no GPU accelerator flags
            ACCEL_FLAGS=""
        fi

        # Create metadata for SSL configuration
        METADATA="startup-script=echo 'Container is starting with SSL configuration'"
        if [ ! -z "$DOMAIN_NAME" ]; then
            METADATA="$METADATA,domain-name=$DOMAIN_NAME,cert-name=$CERT_NAME"
        fi
        
        # Increase boot disk size to address performance warning
        echo "Creating a GCE instance with a 200GB boot disk to ensure good I/O performance..."
        gcloud compute instances create-with-container "${INSTANCE_NAME}" \
          --zone="${REGION}-a" \
          $ACCEL_FLAGS \
          --boot-disk-size=200GB \
          --preemptible \
          --container-image="$CLOUD_IMAGE_TAG" \
          --metadata="$METADATA" \
          --scopes=cloud-platform \
          --tags=http-server || {
              echo "Failed to create GCE instance."
              exit 1
          }
    fi

    echo "Deployment successful!"
    EXTERNAL_IP=$(gcloud compute instances describe "${INSTANCE_NAME}" --zone="${REGION}-a" --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
    
    if [ ! -z "$DOMAIN_NAME" ]; then
        echo "Your application will be available via HTTPS at: https://${DOMAIN_NAME}"
        echo "Please ensure your DNS is configured to point to: $EXTERNAL_IP"
    else
        echo "Your application is available at: https://${EXTERNAL_IP}"
        echo "Note: Using self-signed certificates - you will need to accept browser warnings."
    fi
    
    echo "Use 'gcloud compute ssh ${INSTANCE_NAME} --zone=${REGION}-a' to access the instance if needed."
fi
