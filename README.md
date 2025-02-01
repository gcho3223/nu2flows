# KNU, LXPLUS-GPU 서버에서 nu2flows 설치

## 그전에..
- nu2flows는 설치방법을 README.md로 제공하나 부실함
- docker로 설치가 가능하지만 knu에서는 apptainer를 사용해야함
- lxplus는 docker는 root 권한이 필요하여 rootless 환경을 제공하는 podman으로 해야함

## Idea
### 1) windows에서 wsl을 통해 docker file을 build해서 image를 만든다
- mac에서 docker로 build를 해봤으나 architecture가 달라서 build가 되지 않음
    - platform을 리눅스로 지정해도 애초에 mac의 실리콘 칩과 호환이 안되는 것 같음
    - 따라서 windows에서 wsl에 Ubuntu를 설치하고 여기서 dockerfile을 build 했음
    - **build시 nu2flows/ 디렉토리 내에서 한다.**
```
wsl -d Ubuntu #wsl 실행
cd /path/to/docker/file #dockerfile 경로로 이동
docker build -f docker/Dockerfile -t nu2flows_v2:v2.0 . #docker image 빌드
```
- 그리고 이미지 파일을 서버에 업로드할 수 있게 tar 파일로 저장한다
```
docker save -o nu2flows_v2.tar nu2flows_v2:v2.0
```
- tar파일은 현재 있는 디렉토리에 생성됨
### 2) 이 image를 knu서버에 업로드한다.
- lxplus-gpu에서 이미지 변환하려고 하면 용량때문에 안되고 eos에서 하려고 해도 경로를 못 찾아서 안됐음음
- scp 이용해서 업로드함.
- docker image 파일의 용량은 10GB 이상임...
```
scp nu2flows_v2.tar gcho@cms02.knu.ac.kr:/u/user/gcho/TopPhysics/CPV/MachineLearning
```
### 3) knu에서 apptainer를 이용해 docker image를 변환한다
- docker image를 singularity image로 변환한다
```
singularity build nu2flows_v2.sif docker-archive:///u/user/gcho/TopPhysics/CPV/MachineLearning/nu2flows_v2.tar
```
- lxplus에서 쓰려면 sif 파일을 scp로 업로드해야한다
### 4) 사용해본다...
```
singularity run nu2flows_v2.sif #image 실행->실행시 컨테이너 내부로 들어와짐
singularity shell nu2flows_v2.sif #image 내 shell 접근->위와 동일한 효과
singularity run --nv nu2flows_v2.sif #gpu 사용 가능
singularity exec --nv nu2flows_v2.sif python scripts/train.py #training code 실행
```

## mltools
- nu2flows를 git clone하면 mltools는 submodule로 가져와짐
- 이를 가져오려면 nu2flows 디렉토리에서
```
git submodule update --init --recursive
```
을 하면 mltools도 cloning 해서 다운받아진다!!
