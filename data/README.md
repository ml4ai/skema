## GroMEt Schema and Examples

The contents of this directory include documents summarizing the GroMEt schema along with a collection of examples based on very simple programming idioms (currently all in Python) to demonstrate how they are represented in GroMEt.

- `epidemiology`: Contains subdirectories, one for each epidemiology model that we are currently providing data products for.
- `gromet`: contains the following...
	- `gromet_FN-schema-v0.1.2.pdf` : A pdf with several styles of diagrams that describe the GroMEt schema from different perspectives. The diagram on the bottom-left can be interpreted as a kind of ER-diagram (database entity-relation diagram), where each box represents a table for a type of object, and arrows represent foreign key references to elements of other tables. This is the basis for how we serialize gromet as JSON.
	- `gromet_FN-examples.v0.1.2.pdf` : This draws GroMEt _wiring diagrams_ for each of the example program idioms, all summarized in one pdf. On the left is a legend describing the wiring diagram building-blocks and on the right are Python examples with corresponding wiring diagrams.
	- `gromet_data_types_v0.1.0.pdf` : A set of notes about abstracted primitive data types that GroMEt Function Netowrks natively support. This is presented as a "forest" where parents are super classes. Each box represents a type and some include "interface" functions. For example, the Iterable primitive type provides the function `_iterator()` that will return the Iterator corresponding to an Iterable provided as an argument. Children types inherit the parent interface functions. This is a work in progress. Within GroMEt wiring diagrams, we are not yet displaying "values", so you don't see instances of these types, although you can see instances of interface function calls (which are treated as primitive operator functions (i.e., functions with no FN contents)).
	- Finally, the `examples/` directory contains various representations of simple Python code examples represented as GroMEt. The `examples/README.md` provides more explanation of different file types. The files `*--Gromet-FN-auto.json` are the current JSON serializations of the GroMEt FN representation extracted from the code.

### Swagger object model

Swagger object models are available for several of our internal representations, here: [https://ml4ai.github.io/automates-v2/](https://ml4ai.github.io/automates-v2/)

- `CAST` : Common Abstract Syntax Tree -- this is the internal AST representation that is the target of the Code2FN front-ends (currently supporting front-ends for Python and GCC (Fortran, C, C++)).
- `GroMEt Function Network` : This is the object model for GroMEt Function Networks -- this is what gets serialized to JSON.
- `GroMEt Metadata` : A separate metadata specification for different types of metadata. Anything that starts with `<...>` is a metadatum type, where the `<...>` specifies the type of GroMEt object that the metadata may be associated with.

We use swagger-codegen to auto-generate Python class definitions for these object models and use these internally. This has the advantage it is easier to synchronize the published model definition (on this web page) with how it shows up in the JSON serialization.