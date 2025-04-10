FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
ARG APT_COMMAND="apt-get -o Acquire::Retries=3 -y"

RUN ${APT_COMMAND} update && ${APT_COMMAND} upgrade
RUN ${APT_COMMAND} install clang build-essential libssl-dev wget
RUN ${APT_COMMAND} install verilator

# cmake #
RUN cd /tmp && wget https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0.tar.gz && \
    tar -zxvf cmake-3.20.0.tar.gz && cd cmake-3.20.0 && ./bootstrap && make && make install

WORKDIR /home/ 
COPY run_docker.sh /
CMD ["/bin/bash"]