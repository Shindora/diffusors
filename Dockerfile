# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory to /app
WORKDIR /root

COPY .dockerignore /root/.dockerignore

# Copy the current directory contents into the container at /app
COPY . /root

RUN apt-get update && apt-get install -y apt-utils && apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    ./Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    /opt/conda/bin/conda init bash


# Install any needed packages specified in environment.yml
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "diffusion-venv", "/bin/bash", "-c"]

# Set up data
CMD ["gdown", "13sjrCYbshJEOzLAWex7fnzLa7zcQMWDv"]
CMD ["unzip", "-q", "data.zip", "&&", "rm", "data.zip"]

# Run segmentation_diffuser_two.py with the specified arguments when the container launches
ENTRYPOINT ["python", "segmentation_diffuser_two.py"]
