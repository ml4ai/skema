# Building `skema` images

We publish all project images publicly to Docker Hub.  If you'd like to build images locally to test features, see the instructions below.

## `lumai/askem-skema-py`

```bash
TAG=local
BUILDER_KIT=1 docker build --no-cache -f "Dockerfile.skema-py" -t "lumai/askem-skema-py:$TAG" .
```

Published images: [`lumai/askem-skema-py`](https://hub.docker.com/r/lumai/askem-skema-py)

## `lumai/askem-skema-rs`

```bash
TAG=local
BUILDER_KIT=1 docker build --no-cache -f "Dockerfile.skema-rs" -t "lumai/askem-skema-rs:$TAG" .
```

Published images: [`lumai/askem-skema-rs`](https://hub.docker.com/r/lumai/askem-skema-rs)

## `lumai/askem-skema-text-reading`

???+ note "Dependencies for Dockerfile generation"

    The Dockerfile file for our text reading image is generated using [`sbt`](https://www.scala-sbt.org/download.html)

```bash
TAG=local
cd skema/text_reading/scala
# generate dockerfile
sbt "webapp/docker:stage"
# build image
# NOTE: the current image is only compatible with amd64
cd webapp/target/docker/stage
BUILDER_KIT=1 docker build --no-cache --platform "linux/amd64" -t "lumai/askem-skema-text-reading:$TAG" .
```

Published images: [`lumai/askem-skema-text-reading`](https://hub.docker.com/r/lumai/askem-skema-text-reading)