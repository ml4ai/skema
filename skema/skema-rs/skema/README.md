# SKEMA

This package contains code that is very specific to the SKEMA project.

## skema_service

This is a web service to expose the functionality of the SKEMA Rust components
via REST APIs. To run the program, do:

```
cargo run --bin skema_service
```

Currently, there is one API endpoint implemented: comment extraction. To get
comments for a piece of source code (currently, only a limited subset of Python
is supported), send an HTTP GET request to the service (by default, it will run
at `localhost:8080`), with a JSON payload that looks like the following:

```json
{
    "language": "python",
    "code" "...the source code you wish to process"
}
```
