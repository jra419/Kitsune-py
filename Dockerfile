FROM --platform=linux/amd64 ubuntu:focal

# https://stackoverflow.com/questions/51023312/docker-having-issues-installing-apt-utils
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /opt

RUN apt update
RUN apt -y install \
	build-essential \
	python3 \
	python3-pip \
	wireshark-common \
	tshark \
	cython3

RUN pip3 install \
	numpy \
	pandas \
	jsonpickle \
	scipy \
	scikit-learn

COPY . kitsune
WORKDIR /opt/kitsune
