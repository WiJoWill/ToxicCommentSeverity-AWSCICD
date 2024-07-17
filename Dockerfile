FROM python:3.8-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install dependencies including git and git-lfs
RUN apt-get update -y && \
    apt-get install -y git git-lfs awscli ffmpeg libsm6 libxext6 unzip

# Initialize Git LFS
RUN git lfs install

# Copy the content of the local directory to the container
COPY . /app

# Pull the LFS files
RUN git lfs pull

# Install Python dependencies
RUN pip install -r requirements.txt

# Command to run the application
CMD ["python3", "app.py"]
