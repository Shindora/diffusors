# Use an official Pytorch runtime as a parent image
FROM mambaorg/micromamba:jammy-cuda-12.1.0

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV ENV_NAME=diffusors
EXPOSE 8888

WORKDIR /workspace

USER root
RUN apt update && apt install -y git tmux nvtop \
    && apt clean autoremove -y

USER $MAMBA_USER
RUN micromamba install -y -n base -c conda-forge python=3.11 \
    && micromamba clean --all --yes

# Install any needed packages specified in environment.yml
COPY environment.yml environment.yml
RUN micromamba create -n $ENV_NAME -f environment.yml -y \
    && micromamba env export --name $ENV_NAME --explicit > env.lock \
    && micromamba clean --all --yes

# Set up data
CMD ["gdown", "13sjrCYbshJEOzLAWex7fnzLa7zcQMWDv"]
CMD ["unzip", "-q", "data.zip", "&&", "rm", "data.zip"]

CMD ["bash", "-c",  "jupyter lab --ip=0.0.0.0 --no-browser"]