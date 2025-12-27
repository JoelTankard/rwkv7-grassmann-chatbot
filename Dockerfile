# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Install system dependencies
USER root
RUN apt-get update && apt-get install -y \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*
USER user

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY --chown=user . .

# Download the model weights and tokenizer if they don't exist
# Note: In a real HF Space, you might want to use the huggingface_hub library to download these
# but for this specific implementation, we'll ensure they are present or downloaded.
RUN wget -O model.safetensors https://huggingface.co/RWKV/RWKV7-Goose-World2.8-0.1B-HF/resolve/main/model.safetensors && \
    wget -O rwkv_vocab_v20230424.txt https://raw.githubusercontent.com/BlinkDL/RWKV-LM/main/RWKV-v7/rwkv_vocab_v20230424.txt

# Expose the port the app runs on
EXPOSE 7860

# Command to run the application
CMD ["python", "app.py"]
