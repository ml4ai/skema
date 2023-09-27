<!-- ## [v1.9.?] (pending)
## [v1.9.?](https://github.com/ml4ai/skema/releases/tag/v1.9.?)

### Code2FN

### TextReading

### Eqn Reading

### ISA

### MORAE -->

# Docker images

We publish tagged images to dockerhub for each commit made to our primary branch (`main`), as well as each semver (described below).

[![Docker lumai/askem-skema-py Image Version (latest by date)](https://img.shields.io/docker/v/lumai/askem-skema-py?sort=date&logo=docker&label=lumai%2Faskem-skema-py)](https://hub.docker.com/r/lumai/askem-skema-py)  
[![Docker lumai/askem-skema-img2mml Image Version (latest by date)](https://img.shields.io/docker/v/lumai/askem-skema-img2mml?sort=date&logo=docker&label=lumai%2Faskem-skema-img2mml)](https://hub.docker.com/r/lumai/askem-skema-img2mml)  
[![Docker lumai/askem-skema-rs Image Version (latest by date)](https://img.shields.io/docker/v/lumai/askem-skema-rs?sort=date&logo=docker&label=lumai%2Faskem-skema-rs)](https://hub.docker.com/r/lumai/askem-skema-rs)  
[![Docker lumai/askem-skema-text-reading Image Version (latest by date)](https://img.shields.io/docker/v/lumai/askem-skema-text-reading?sort=date&logo=docker&label=lumai%2Faskem-skema-text-reading)](https://hub.docker.com/r/lumai/askem-skema-text-reading)

# Changes

!!! note "Semantic versioning"

    The minor version component of our tags corresponds to a program milestone.  Each increment to the patch version corresponds to changes introduced in a two-week sprint (v1.9.1 -> Changes introduced during the first sprint following the completion of Milestone 9).

<!-- ## v1.9.2 (pending)

- [PRs](https://github.com/ml4ai/skema/pulls?q=is%3Apr+is%3Amerged+merged%3A2023-09-04..2023-09-17)
- [resolved issues](https://github.com/ml4ai/skema/issues?q=is%3Aissue+is%3Aclosed+closed%3A2023-09-04..2023-09-17) -->

## [v1.9.2](https://github.com/ml4ai/skema/releases/tag/v1.9.2)

- [PRs](https://github.com/ml4ai/skema/pulls?q=is%3Apr+is%3Amerged+merged%3A2023-09-04..2023-09-18)
- [resolved issues](https://github.com/ml4ai/skema/issues?q=is%3Aissue+is%3Aclosed+closed%3A2023-09-04..2023-09-18)

## [v1.9.1](https://github.com/ml4ai/skema/releases/tag/v1.9.1)

- [PRs](https://github.com/ml4ai/skema/pulls?q=is%3Apr+is%3Amerged+merged%3A2023-08-21..2023-09-03)
- [resolved issues](https://github.com/ml4ai/skema/issues?q=is%3Aissue+is%3Aclosed+closed%3A2023-08-21..2023-09-03)
<!-- - [bug fixes](https://github.com/ml4ai/skema/issues?q=is%3Aissue+is%3Aclosed+closed%3A2023-08-21..2023-09-03+label%3A%22bug%22)
- [resolved issues](https://github.com/ml4ai/skema/issues?q=is%3Aissue+is%3Aclosed+closed%3A2023-08-21..2023-09-03+-label%3A%22bug%22+) -->


<!-- is:pr is:merged merged:2023-08-21..2023-09-03 label:"Code2FN"  -->
<!-- - [PRs](https://github.com/ml4ai/skema/pulls?q=is%3Apr+is%3Amerged+merged%3A2023-08-21..2023-09-03+label%3A%22Code2FN%22) -->


## [v1.9.0](https://github.com/ml4ai/skema/releases/tag/v1.9.0)
  
Corresponds to ASKEM SKEMA Milestone 9 release.

- [PRs](https://github.com/ml4ai/skema/pulls?q=is%3Apr+is%3Amerged+merged%3A2023-05-01..2023-07-31)
- [bug fixes](https://github.com/ml4ai/skema/issues?q=is%3Aissue+is%3Aclosed+closed%3A2023-05-01..2023-07-31+label%3A%22bug%22)
- [resolved issues](https://github.com/ml4ai/skema/issues?q=is%3Aissue+is%3Aclosed+closed%3A2023-05-01..2023-07-31+-label%3A%22bug%22+)

#### Code2FN
- Python idiom support
  - nested functions (function closures)
  - recursively called functions
- TS2CAST Fortran front-end developments
  - preprocessor (id unsupported idioms, identify missing include files, fixing unsupported `&` line continuation character)
  - compiler directives using GCC pre-processor
  - derived types (classes/structs) as FN Records
  - representing program, module and "outside" code in FN module namespaces
  - handling Fortran `contains`
- Initial support for tree-sitter-based MATLAB front-end
- Generalized JSON2GroMEt
- Additional GroMEt ingestion front-end
- source code comment to FN alignment
- bug fixes

#### TextReading
- unified TA-1 metadata extractions library
- unified TA-1 text reading REST API
- updates to TR and Scenario Context extraction with initial support for climate and earth science domain
- added AMR linking utility to text extractions with scenario contexts; includes support for AMR Petri Net and RegNet
- bug fixes

#### Eqn Reading
- new conversion service and REST API
- improvements to pipeline for generating data for training equation extraction model
- evaluation dataset cleanup
- service structure reorganization
- image2MathML model improvements
- service response time improvements
- MathML inspection and annotation GUIs
- new support for interpretation of presentation MathML to generate content MathML
- improvements to DECAPODES interpretation of dynamics equations

### ISA
- improved seed selection for seeded graph matching (SGM) algorithm
- variable name similarity measures
- expanded SGM method in graph matching

### MORAE
- improved support for model identification and extraction out of FN
- Eqn2PetriNet produces AMR PetriNet
- Eqn2RegNet produces AMR RegNet
- work on ABM representation


<!-- ## [v1.9.?] (pending)
## [v1.9.?](https://github.com/ml4ai/skema/releases/tag/v1.9.?)

### Code2FN

### TextReading

### Eqn Reading

### ISA

### MORAE -->