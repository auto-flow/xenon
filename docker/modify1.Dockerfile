FROM xenon:v3.0
MAINTAINER qichun.tang qichun.tang@xtalpi.com

RUN pip install tabulate
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8