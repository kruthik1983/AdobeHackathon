# Dockerfile
# Use a lightweight Python base image for AMD64 architecture
FROM --platform=linux/amd64 python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the working directory
COPY requirements.txt .

# Install Python dependencies.
# --no-cache-dir: Reduces image size by not storing build cache.
# -r requirements.txt: Installs packages listed in requirements.txt.
# Ensure all dependencies are installed *during the build*.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire source code directory into the container's /app directory
COPY src/ ./src

# Set the command to run the main script when the container starts.
# This makes the container automatically execute the outline extraction logic.
CMD ["python", "-m", "src.main"]