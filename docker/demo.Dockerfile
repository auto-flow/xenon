# NOTE: notebook版本的镜像是手动commit打出来的
# + 镜像构建步骤：
# 1. clone xenon代码：git clone git@bitbucket.org:xtalpi/xenon.git  # 下载后确认当前目录有xenon文件夹
# 2. 根据Dockerfile构建镜像：docker build -t 你的镜像:你的tag -f xenon/docker/demo.Dockerfile  .
# 3. 将镜像推送到远程： docker push 你的镜像:你的tag 
# + 进行镜像简单测试一下：
# 1. mkdir -p savedpath && docker run --rm -v $PWD/savedpath:/root/savedpath 你的镜像:你的tag  /bin/bash /root/python_packages/xenon/simple_test.sh
# 2. 确认当前目录有savedpath且运行成功

# 写成你的基镜像(以ADMET为例)
FROM harbor.atompai.com/nitrogen/admet:v2.0.2
# 改一下维护人
MAINTAINER qichun.tang qichun.tang@xtalpi.com
# 装xenon依赖环境
WORKDIR /root
ADD xenon/requirements.txt /root
ADD xenon/Makefile /root
RUN apt install make -y
# RUN make change_apt_source
RUN make change_pip_source
RUN make install_apt_deps
RUN make install_pip_deps
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
# 把xenon代码装到PYTHONPATH中
ADD xenon /root/xenon
WORKDIR /root/xenon
RUN bash -e setup.sh
WORKDIR /root
ENV PYTHONPATH=/root/python_packages/xenon
