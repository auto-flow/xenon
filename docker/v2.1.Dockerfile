FROM harbor.atompai.com/nitrogen/xenon:v1.2
MAINTAINER qichun.tang qichun.tang@xtalpi.com
RUN pip install ultraopt
RUN pip install xlearn