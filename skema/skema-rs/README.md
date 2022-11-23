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

#### Running without Docker

To run the service natively (without Docker), invoke the following command:

```
cargo run --bin skema_service
```

#### Running with Docker

To run the service via Docker, invoke the following command:

```
docker-compose up --build
```

### Documentation

OpenAPI documentation for the service can be found at
`http://localhost:8080/api-docs/`.
