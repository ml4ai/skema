mathml
======

This crate provides a library (`mathml`) for parsing MathML,
as well as a program (`parse_mathml`) that parses an input MathML document and
outputs a graph representation of the document in the Graphviz DOT format.

## `parse_mathml`

The `parse_mathml` program produces a DOT representation of a MathML
document.

Invocation:

```console
cargo run -- <INPUT>
```

Where `INPUT` is the name of the input file. The output DOT representation is
printed to the terminal.

## API

The `parse` function (`mathml::parsing::parse`) takes a string and outputs a
MathML document AST representation.

## Limitations

The parser currently handles only a limited subset of the full MathML
specification - whatever is necessary to meet the goals of the SKEMA program.
More specifically, we focus on handling element types that we expect to see
from the SKEMA equation reading (ER) pipeline. If we encounter MathML
constructs from the ER pipeline that the parser cannot handle, we will extend
the parser to handle them.
