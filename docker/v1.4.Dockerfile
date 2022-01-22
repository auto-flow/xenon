# NOTE: notebook版本的镜像是手动commit打出来的
# + 镜像构建步骤：
# 1. clone xenon代码：git clone git@bitbucket.org:xtalpi/xenon.git  # 下载后确认当前目录有xenon文件夹
# 2. 根据Dockerfile构建镜像：docker build -t harbor.atompai.com/nitrogen/xenon:v1.4 -f xenon/docker/v1.4.Dockerfile  .
# 3. 将镜像推送到远程 docker push harbor.atompai.com/nitrogen/xenon:v1.4
# + 进行镜像简单测试一下：
# 1. mkdir -p savedpath && docker run --rm -v $PWD/savedpath:/root/savedpath harbor.atompai.com/nitrogen/xenon:v1.4 /bin/bash /root/python_packages/xenon/simple_test.sh
# 2. 确认当前目录有savedpath且运行成功

FROM harbor.atompai.com/nitrogen/xenon:v1.2_notebook
MAINTAINER qichun.tang qichun.tang@xtalpi.com

ADD xenon /root/xenon
WORKDIR /root/xenon
RUN bash -e setup.sh
WORKDIR /root
ENV PYTHONPATH=/root/python_packages/xenon
