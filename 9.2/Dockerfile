FROM nvidia/cuda:10.0-cudnn7-runtime-centos7

MAINTAINER astars <>

RUN yum install wget -y
RUN yum -y install zip unzip
RUN wget --no-check-certificate --no-cookies --header "Cookie: oraclelicense=accept-securebackup-cookie" http://download.oracle.com/otn-pub/java/jdk/8u131-b11/d54c1d3a095b4ff2b6607d096fa80163/jdk-8u131-linux-x64.tar.gz
RUN mkdir /var/tmp/jdk
RUN cp jdk-8u131-linux-x64.tar.gz /var/tmp/jdk
RUN tar xzf /var/tmp/jdk/jdk-8u131-linux-x64.tar.gz -C /var/tmp/jdk
RUN rm -rf /var/tmp/jdk/jdk-8u131-linux-x64.tar.gz
RUN wget https://downloads.cs.stanford.edu/nlp/software/stanford-corenlp-4.2.2.zip
RUN unzip stanford-corenlp-4.2.2.zip
RUN rm stanford-corenlp-4.2.2.zip

ENV JAVA_HOME=/var/tmp/jdk/jdk1.8.0_131
ENV PATH=$JAVA_HOME/bin:$PATH
ENV CLASSPATH=.:$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tool.jar

RUN export CLASSPATH="`find . -name '*.jar'`"

RUN yum install -y gcc gcc-c++ && yum install -y wget && wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh -O ~/miniconda.sh && \
/bin/bash ~/miniconda.sh -b -p /opt/conda && \
rm ~/miniconda.sh && \
/opt/conda/bin/conda clean -tipsy && \
ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
echo "conda activate base" >> ~/.bashrc

ENV PATH /opt/conda/bin:$PATH

COPY env.yml /env.yml
RUN conda env update --file /env.yml

ENV LD_LIBRARY_PATH /usr/local/cuda-10.0/targets/x86_64-linux/lib:/usr/lib64:$LD_LIBRARY_PATH
RUN cd /usr/local/cuda-10.0/targets/x86_64-linux/lib && ln -s libcudart.so.10.0.130 libcudart.so.10.1 && ln -s libcublas.so.10.0.130 libcublas.so.10 && ln -s libcufft.so.10.0.145 libcufft.so.10 && ln -s libcurand.so.10.0.130 libcurand.so.10 && ln -s libcusolver.so.10.0.130 libcusolver.so.10 && ln -s libcusparse.so.10.0.130 libcusparse.so.10 && ln -s libcublas.so.10.0.130 libcublas.so && ln -s libcudnn.so.7.6.5 libcudnn.so


RUN mkdir -p /app
RUN mkdir -p /app/log
COPY ../8.27 /app
WORKDIR /app

ENTRYPOINT ["bash","run.sh"]
