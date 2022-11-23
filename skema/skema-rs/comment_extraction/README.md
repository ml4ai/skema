Comment extraction
==================

This crate provides a library (`comment_extraction`) as well as a program
(`extract_comments`) for extracting comments from source code.

## `extract_comments`

The `extract_comments` program produces a JSON representation of the comments
in a source file.

The quickest way to get started and running the program is using Cargo. The
Cargo invocation (assuming your current working directory is the
`comment_extraction` directory) is:

```console
cargo run -- <INPUT> [OUTPUT]
```

Where `INPUT` is the name of the input source file, and `OUTPUT` is the name of
the file to write the JSON output to. If the `OUTPUT` argument is not provided,
the program prints the JSON to standard output instead.
