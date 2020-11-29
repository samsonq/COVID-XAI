FROM continuumio/miniconda:4.5.4

USER root

RUN apt-get update && \
    apt-get upgrade -y && \
    apt install -y python3

COPY requirements.txt ./COVID-LRP/requirements.txt
RUN pip install -r ./COVID-LRP/requirements.txt
