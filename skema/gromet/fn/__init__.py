# coding: utf-8

# flake8: noqa
"""
    Grounded Model Exchange (GroMEt) schema for Function Networks

    This document defines the GroMEt Function Network data model. Note that Metadata is defined in separate spec.  __Using Swagger to Generate Class Structure__  To automatically generate Python or Java models corresponding to this document, you can use [swagger-codegen](https://swagger.io/tools/swagger-codegen/). This can be used to generate the client code based off of this spec, and in the process this will generate the data model class structure.  1. Install via the method described for your operating system    [here](https://github.com/swagger-api/swagger-codegen#Prerequisites).    Make sure to install a version after 3.0 that will support openapi 3. 2. Run swagger-codegen with the options in the example below.    The URL references where the yaml for this documentation is stored on    github. Make sure to replace CURRENT_VERSION with the correct version.    (The current version is `0.1.7`.)    To generate Java classes rather, change the `-l python` to `-l java`.    Change the value to the `-o` option to the desired output location.    ```    swagger-codegen generate -l python -o ./client -i https://raw.githubusercontent.com/ml4ai/automates-v2/master/docs/source/gromet_FN_v{CURRENT_VERSION}.yaml    ``` 3. Once it executes, the client code will be generated at your specified    location.    For python, the classes will be located in    `$OUTPUT_PATH/skema.gromet.fn.`.    For java, they will be located in    `$OUTPUT_PATH/src/main/java/io/swagger/client/model/`  If generating GroMEt schema data model classes in SKEMA (AutoMATES), then after generating the above, follow the instructions here: ``` <automates>/automates/model_assembly/gromet/model/README.md ```   # noqa: E501

    OpenAPI spec version: 0.1.10
    Contact: claytonm@arizona.edu
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""

from __future__ import absolute_import

# import models into model package
from skema.gromet.fn.function_type import FunctionType
from skema.gromet.fn.gromet_box import GrometBox
from skema.gromet.fn.gromet_box_conditional import GrometBoxConditional
from skema.gromet.fn.gromet_box_function import GrometBoxFunction
from skema.gromet.fn.gromet_box_loop import GrometBoxLoop
from skema.gromet.fn.gromet_fn import GrometFN
from skema.gromet.fn.gromet_fn_module import GrometFNModule
from skema.gromet.fn.gromet_fn_module_collection import GrometFNModuleCollection
from skema.gromet.fn.gromet_fn_module_dependency_reference import GrometFNModuleDependencyReference
from skema.gromet.fn.gromet_object import GrometObject
from skema.gromet.fn.gromet_port import GrometPort
from skema.gromet.fn.gromet_wire import GrometWire
from skema.gromet.fn.import_source_type import ImportSourceType
from skema.gromet.fn.import_type import ImportType
from skema.gromet.fn.literal_value import LiteralValue
from skema.gromet.fn.metadata import Metadata
from skema.gromet.fn.typed_value import TypedValue
