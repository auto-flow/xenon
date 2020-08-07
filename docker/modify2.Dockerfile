FROM xenon:v3.0
MAINTAINER qichun.tang qichun.tang@xtalpi.com

ADD setup.py setup.py
ADD dsmac dsmac
ADD generic_fs generic_fs
ADD xenon_client xenon_client
ADD xenon xenon
ADD README.rst README.rst
RUN python setup.py install  &&  pip install liquidpy==0.5.0
