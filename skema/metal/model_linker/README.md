# SKEMA Model Linking Utility

Link ASKEM models, such as `AMR` and `GrometFN` with text extractions in their [canonical form](https://github.com/ml4ai/ASKEM-TA1-DataModel).
Currently, we support linking only to `AMR` representations, such as _PetriNets_ and _RegNets_.
Soon, we will support linking to Gromet function networks.

```

## Usage

To link an `AMR` json file to [canonical text extractions](https://github.com/ml4ai/ASKEM-TA1-DataModel), use the `link_amr` utility.

```
SYNOPSIS
    link_amr AMR_PATH ATTRIBUTE_COLLECTION AMR_TYPE <flags>

DESCRIPTION
    Links and AMR model to an attribute collections from ASKEM text reading pipelines

POSITIONAL ARGUMENTS
    AMR_PATH
        Type: str
    ATTRIBUTE_COLLECTION
        Type: str
    AMR_TYPE
        Type: str

FLAGS
    -o, --output_path=OUTPUT_PATH
        Type: Optional[typing.Optional[str]]
        Default: None
    --similarity_model=SIMILARITY_MODEL
        Type: str
        Default: 'sentence-t...
    --similarity_threshold=SIMILARITY_THRESHOLD
        Type: float
        Default: 0.7
    -d, --device=DEVICE
        Type: Optional[typing.Optional[str]]
        Default: None

NOTES
    You can also use flags syntax for POSITIONAL ARGUMENTS
```