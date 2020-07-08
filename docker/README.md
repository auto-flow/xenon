```bash
$ cd Xenon
$ ls
data  docs  docker dsmac  examples ...
$ docker build . -f docker/Dockerfile -t xenon:v3.0
$ docker tag xenon:v3.0 harbor.atompai.com/nitrogen/xenon:v3.0
$ docker push harbor.atompai.com/nitrogen/xenon:v3.0
```