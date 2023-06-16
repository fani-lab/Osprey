FROM nvidia/cuda:11.1.1-devel-centos7

SHELL ["/bin/bash", "-c"]

RUN yum install -y curl bzip2
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba && micromamba shell init --shell=bash --prefix=~/micromamba

COPY ./environment.yml /opt/environment.yml

RUN micromamba create -n osprey --file /opt/environment.yml && eval "$(micromamba shell hook --shell=bash)"
RUN micromamba shell init --shell=bash --prefix=~/micromamba

ENV PATH=/root/micromamba/envs/osprey/bin:$PATH
RUN python -m nltk.downloader stopwords punkt

CMD ["bash"]
