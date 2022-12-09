# skema-rs

This workspace contains Rust packages developed for the SKEMA program.

Currently, it has the following members:

- `comment_extraction`: Library and binary for
  extracting comments from code.
- `mathml`: Library and binary for parsing MathML.
- `skema`:
    - Library for working the GroMEt interchange format.
    - Binary for exposing SKEMA functionality via a REST API (`skema_service`).
      See below for instructions on running the service.

## skema_service

This is a web service to expose the functionality of the SKEMA Rust components
via REST APIs.

### Running the service

#### End-user

To run the service via Docker, invoke the following command:

```
docker-compose up --build
```

#### Developer

If you are a developer working on this project, you will likely want to launch
the `skema_service` outside of Docker:

```
cargo run --bin skema_service
```

And if you want to test API endpoints that rely on the Memgraph database, you
will want to launch Memgraph separately. You can do so using the following
command:

```
docker-compose -f docker-compose.memgraph.yml up -d
```

You can skip the `-d` flag in the above invocation if you want to leave the
Dockerized Memgraph service in the foreground. test

### Documentation

OpenAPI documentation for the service can be found at
`http://localhost:8080/api-docs/`.
