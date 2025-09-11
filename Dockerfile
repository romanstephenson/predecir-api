# Use Python 3.11 slim base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libffi-dev \
    libssl-dev \
    libjpeg-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    wget \
    curl \
    dnsutils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Download model file from Azure Blob Storage
RUN mkdir -p model/v1 && \
    wget "<yoururltoyourblobstoragehere>" \
    -O model/v1/BreastCancerRecurrence_Predecir_2.pkl

# Copy all application files
COPY . .

# Expose FastAPI port
EXPOSE 8001

# Run FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
