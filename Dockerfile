# Use an official Python runtime as a parent image
FROM --platform=linux/amd64 python:3.11

# Set the working directory to /app
WORKDIR /root

COPY .dockerignore /root/.dockerignore

# Copy the current directory contents into the container at /app
COPY . /root
# Install base utilities
RUN apt-get update && apt-get -y upgrade \
    && apt-get install -y --no-install-recommends \
    git \
    wget \
    g++ \
    gcc \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && echo "Running $(conda --version)" && \
    conda init bash && \
    . /root/.bashrc && \
    conda update conda

# Install any needed packages specified in environment.yml
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "diffusion-venv", "/bin/bash", "-c"]

# Set up data
CMD ["gdown", "13sjrCYbshJEOzLAWex7fnzLa7zcQMWDv"]
CMD ["unzip", "-q", "data.zip", "&&", "rm", "data.zip"]

# Run segmentation_diffuser_two.py with the specified arguments when the container launches
ENTRYPOINT ["python", "segmentation_diffuser_two.py"]
