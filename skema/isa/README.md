# Incremental Structural Alignment (ISA)

This folder contains code for Incremental Structural Alignment (ISA) server, the functions for structural alignment of mathematical expressions. Additionally, it involves integration with `skema_service` to support specific tasks related to MathML processing. Follow the instructions below to set up and run both services.

## ISA Server

To launch the ISA server, execute the following command:

```bash
uvicorn isa_service:app --reload --port <port number>
```

Replace `<port number>` with the desired port for the ISA server.

## skema_service Integration

In addition to starting the ISA server, you need to run the `skema_service` to enable MathML processing tasks. Refer to the following link for specific instructions on launching `skema_service`: [skema_service Setup](https://github.com/ml4ai/skema/tree/main/skema/skema-rs).

### ISA service Input Parameters

The `ISA service` takes the following input parameters:

- `file1: str`: Path to the first MathML file.
- `file2: str`: Path to the second MathML file.
- `mention_json1: str = ""`: Path to the mentions file for the first MathML article (optional).
- `mention_json2: str = ""`: Path to the mentions file for the second MathML article (optional).
- `mode: int = 1`: Alignment mode (default is 1).

### Alignment Modes

- Mode 0: No prior considerations for seed selection.
- Mode 1: Seed selection based on the similarity of variable names.
- Mode 2: Seed selection based on the similarity of variable definitions in the text.

If paths for both mentions files are provided, the mode is automatically set to 2. If mention file access fails, the mode reverts to 1.

###  Output

The output of `ISA service` is a tuple with the following components:

- `matching_ratio`: Matching ratio between equations 1 and 2.
- `num_diff_edges`: Number of different edges between equations 1 and 2.
- `node_labels1`: List of variable and term names in equation 1.
- `node_labels2`: List of variable and term names in equation 2.
- `aligned_indices1`: Aligned indices in the name list of equation 1.
- `aligned_indices2`: Aligned indices in the name list of equation 2.
- `union_graph`: Visualization of the alignment result.
- `perfectly_matched_indices1`: Strictly matched node indices in Graph 1.

Feel free to adjust the paths and parameters based on your specific use case.