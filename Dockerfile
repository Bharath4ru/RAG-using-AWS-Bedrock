# Base image with Python 3.9
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the Python script and other necessary files to the container
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir \
    boto3 \
    streamlit \
    langchain-community \
    langchain \
    faiss-cpu \
    pypdf \
    awscli

# Expose the Streamlit default port
EXPOSE 8501

# Set the entrypoint with explicit server settings
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
