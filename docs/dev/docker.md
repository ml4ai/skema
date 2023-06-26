# Building `skema` images

We publish all project images publicly to Docker Hub.  If you'd like to build images locally to test features, see the instructions below.

## `lumai/askem-skema-py`

```bash
BUILDER_KIT=1 docker build --no-cache -f "Dockerfile.skema-py" -t "lumai/askem-skema-py:local" .
```

Published images: [`lumai/askem-skema-py`](https://hub.docker.com/r/lumai/askem-skema-py)

## `lumai/askem-skema-rs`

```bash
BUILDER_KIT=1 docker build --no-cache -f "Dockerfile.skema-rs" -t "lumai/askem-skema-rs:local" .
```

Published images: [`lumai/askem-skema-rs`](https://hub.docker.com/r/lumai/askem-skema-rs)

## `lumai/askem-skema-text-reading`

```bash
cd skema/text_reading/scala
docker build -f "Dockerfile" -t "lumai/askem-skema-text-reading:local" .
```

Published images: [`lumai/askem-skema-text-reading`](https://hub.docker.com/r/lumai/askem-skema-text-reading)