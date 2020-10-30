FROM python:3.7
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
RUN pip install torch==1.5.0

# install xenon in image
#ADD setup.py setup.py
#ADD dsmac dsmac
#ADD generic_fs generic_fs
#ADD xenon_client xenon_client
#ADD xenon xenon
#ADD README.rst README.rst
#RUN python setup.py install

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8