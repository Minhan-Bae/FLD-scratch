# Komedi Landmark Detection Deployments

피측정자의 안면이 포함된 이미지로 부터, 안면 영역을 크롭하고, 27종의 랜드마크를 탐지합니다.

## 구동 방법

**Step 1: 도커 이미지 생성**
- `cd ./APP` 명령어를 입력해 `APP` 디렉토리로 이동합니다.

```console
$ cd ./APP
$ docker build -t komedi-api .
```

**Step 2: 서버 생성**
- API 서버에서 8000 포트를 사용하고, 데모 페이지에서 8501 포트를 사용합니다.
- 추후 도커 종료 및 삭제를 위해 도커명은 komedi-api로 고정합니다.
- 사용 디바이스는 cpu로 고정합니다.

```console
$ docker run --name komedi-api -p 8000:8000 -p 80:8501 komedi-api
```

**Step 4: 데모 페이지 실행**

다음 명령어 실행 이후, [데모페이지](http:133.186.250.203)에서 확인할 수 있습니다.

```console
$ sh ./run/demo_run.sh
```

**Step 5: 데모 페이지 종료**

`ctrl+c` 를 입력 후 터미널에 아래 쉘을 실행합니다.

```console
$ docker exec komedi-api sh ./run/demo_end.sh
```

**Step 6: 이미지 및 서버 삭제**

```console
$ sh ./run/docker_end.sh
$ docker container prune
$ docker image prune
```