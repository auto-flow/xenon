FROM harbor.atompai.com/nitrogen/xenon:v1.2
MAINTAINER qichun.tang qichun.tang@xtalpi.com
RUN apt install make cmake -y
RUN pip install -U pip &&  pip install tabular_nn>=0.1.1 category_encoders==2.0.0 \
    scikit-optimize==0.7.4 Pyro4==4.80 terminaltables tqdm seaborn==0.10.0 \
    hiplot