FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    apt-transport-https \
    ca-certificates \
    gnupg \
    openssl \
    gcc \
    g++ \
    libcudart11.0 \
    libblas-dev \
    liblapack-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Google Cloud SDK
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update && apt-get install -y google-cloud-cli && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python dependencies
COPY requirements.txt .

# Copy .env file first (this allows caching of layers below if .env doesn't change)
COPY .env /app/.env

# Fix for bitsandbytes library
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.10/site-packages/bitsandbytes/

# Install Python dependencies with specific pip options for bitsandbytes
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --force-reinstall bitsandbytes

# Copy application code (excluding .env which was already copied)
COPY . .

# Create a directory for temporary files and set proper permissions
RUN mkdir -p temp && chmod 777 temp

# Create a directory for certificates
RUN mkdir -p /certs

# Copy the startup script
COPY startup.sh /app/
RUN chmod +x /app/startup.sh

# Expose port for HTTPS traffic
EXPOSE 443

# Use the startup script to fetch certificates and start the application
CMD ["/app/startup.sh"]
