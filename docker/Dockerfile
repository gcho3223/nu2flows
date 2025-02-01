FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# Local and environment variables
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PIP_ROOT_USER_ACTION=ignore
ENV PIP_NO_CACHE_DIR=false
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /nu2flows

RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get -y install \
        build-essential \
        wget \
        curl \
        git \
        make \
        hdf5-tools \
        gcc \
        graphviz \
        #sudo \
        texlive \
        texlive-latex-extra \
        texlive-fonts-recommended \
        dvipng


# Update python pip
RUN python -m pip install --upgrade pip
RUN python --version
RUN python -m pip --version

# Intall python dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade -r requirements.txt

# 프로젝트 파일 복사 (nu2flows/ 전체를 이미지 내 /nu2flows로 복사)
COPY . .

# 서브모듈 초기화 (mltools 포함)
RUN git submodule update --init --recursive
