FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create a directory for temporary files and set proper permissions
RUN mkdir -p temp && chmod 777 temp

# Expose port 443 for HTTPS traffic
EXPOSE 443

# Start the application using Uvicorn with HTTPS support.
# Ensure your SSL key and certificate are available at /certs/key.pem and /certs/cert.pem.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "443", "--ssl-keyfile", "/certs/key.pem", "--ssl-certfile", "/certs/cert.pem"]
