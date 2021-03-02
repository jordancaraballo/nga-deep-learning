# Arguments to pass to the image
ARG CUDA_VER=11.0
ARG LINUX_VER=centos7
ARG FROM_IMAGE=nvcr.io/nvidia/rapidsai/rapidsai

# Import RAPIDS container as the BASE Image (cuda base image)
# CentOS 7 image with CUDA 11.0 support. runtime container
# includes examples notebooks and jupyterhub server.
FROM ${FROM_IMAGE}:cuda${CUDA_VER}-runtime-${LINUX_VER}

LABEL NGA Deep Learning
LABEL OS CentOS
LABEL OS_MAJOR_VERS 7
MAINTAINER NASA GSFC Code 587 - Jordan A. Caraballo-Vega

# Adding ENV variables
ENV RAPIDS_DIR=/rapids

COPY . /opt/apt
WORKDIR /opt/apt
RUN source activate rapids && \
    pip install -r requirements.txt

HEALTHCHECK NONE

CMD ["/bin/bash"]
