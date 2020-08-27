FROM harbor.atompai.com/nitrogen/xenon:v3.0
MAINTAINER qichun.tang qichun.tang@xtalpi.com

RUN  pip install Pyro4 sympy hint
