# Use an official NVIDIA CUDA base image
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
# We install torch with CUDA support specifically
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN pip3 install fastapi uvicorn

# Copy your project files
COPY . .

# Create the offload folder as required by your Windows troubleshooting
RUN mkdir -p /app/offload_folder

# Expose the FastAPI port
EXPOSE 8000

# Run the backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]