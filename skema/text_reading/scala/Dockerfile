FROM  ubuntu:22.04
CMD   bash

# ==============================================================================
# INSTALL SOFTWARE VIA THE UBUNTU PACKAGE MANAGER
# =============================================================================
ARG DEBIAN_FRONTEND=noninteractive
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
RUN apt-get update && \
    apt-get -y --no-install-recommends install apt-utils

# Use individual commands to prevent excess time usage when re-building
RUN apt-get -y --no-install-recommends install curl wget gnupg2 git 
# TODO: switch to Java 11
# RUN apt-get -y --no-install-recommends install openjdk-11-jdk antlr4 doxygen
# RUN update-alternatives --set java $(update-alternatives --list java | grep java-11)
RUN apt-get -y --no-install-recommends install openjdk-8-jdk antlr4 doxygen
RUN update-alternatives --set java $(update-alternatives --list java | grep java-8)
RUN apt-get -y --no-install-recommends install gcc build-essential pkg-config
RUN apt-get -y --no-install-recommends install python3-dev python3-pip python3-venv

# Add Scala and SBT
RUN echo "deb https://repo.scala-sbt.org/scalasbt/debian all main" | tee /etc/apt/sources.list.d/sbt.list
RUN echo "deb https://repo.scala-sbt.org/scalasbt/debian /" | tee /etc/apt/sources.list.d/sbt_old.list
RUN curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823" | apt-key add
RUN apt-get update && apt-get -y --no-install-recommends install sbt

RUN apt-get clean && rm -rf /var/lib/apt/lists/*
# =============================================================================

# =============================================================================
# Add PACKAGES FOR TR PIPELINE
# =============================================================================


RUN mkdir -p /TR_utils
WORKDIR /TR_utils
RUN git clone https://github.com/lum-ai/regextools.git
WORKDIR /TR_utils/regextools
RUN sbt publishLocal
# =============================================================================

# =============================================================================
# CREATE A PYTHON VENV AND UPGRADE PYTHON TOOLS
# =============================================================================
#ENV VIRTUAL_ENV=/opt/automates_venv
#RUN python3 -m venv $VIRTUAL_ENV
#ENV PATH="$VIRTUAL_ENV/bin:$PATH"
#
#RUN pip install --upgrade setuptools
#RUN pip install wheel
# =============================================================================

# =============================================================================
# SETUP THE SKEMA TR ENVIRONMENT
# =============================================================================
RUN mkdir -p /skema/text_reading
COPY . /skema/text_reading
WORKDIR /skema/text_reading
# Compile the fat jar
# NOTE: we don't need a fat JAR
#RUN sbt -J-Xmx6g assembly

# =============================================================================

# =============================================================================
# SETUP THE PYTHON ENVIRONMENT TO PROCESS THE PARQUET FILES
# =============================================================================
RUN pip3 install pandas fastparquet tqdm

# =============================================================================
# PREPARE THE WORKING ENVIRONMENT
# =============================================================================
RUN mkdir -p /data/inputs
RUN mkdir -p /data/json_inputs
RUN mkdir -p /data/outputs
RUN chmod 777 /data/json_inputs

RUN chmod +x annotate_cosmos_json



# =============================================================================
# ENTRY POINT IS THE  WEB API
# =============================================================================
CMD ["sbt", "webapp/run"]
# =============================================================================