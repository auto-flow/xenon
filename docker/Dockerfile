FROM continuumio/miniconda3
MAINTAINER qichun.tang qichun.tang@xtalpi.com
WORKDIR /root
ADD requirements.txt /root
ADD Makefile /root
RUN apt install make -y
RUN make change_apt_source
RUN make change_pip_source
RUN make install_apt_deps
RUN make install_pip_deps
# torch is installed for Xenon 4.0 prototype
# if you are normal user, ignore it
#RUN pip install torch==1.5.0

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8