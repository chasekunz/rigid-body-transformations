# Use the official Python image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a workspace directory
WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

# Set the default user
ARG USERNAME=vscode
RUN useradd -ms /bin/bash $USERNAME
USER $USERNAME

# Set the default working directory
WORKDIR /workspace
