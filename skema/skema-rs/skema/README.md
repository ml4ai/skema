# SKEMA

This package contains code that is very specific to the SKEMA project.

## skema_service

This is a web service to expose the functionality of the SKEMA Rust components
via REST APIs. To run the program, do:

```
cargo run --bin skema_service
```

Currently, there is one API endpoint implemented: comment extraction.

OpenAPI documentation for the services can be found at
`http://localhost:8080/api-docs/`.
