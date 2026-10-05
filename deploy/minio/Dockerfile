FROM golang:1.24.3-bookworm AS build
ENV GOMAXPROCS=2
WORKDIR /source
RUN git init && git remote add origin https://github.com/minio/minio.git && git fetch --depth 1 origin f19c534b9f457773dcd043d977433e1a71525c3b && git checkout FETCH_HEAD
RUN --mount=type=cache,target=/go/pkg/mod --mount=type=cache,target=/root/.cache/go-build CGO_ENABLED=0 go build -p 2 -trimpath -o /minio .
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*
COPY --from=build /minio /usr/local/bin/minio
ENTRYPOINT ["minio"]
