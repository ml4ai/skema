# coding: utf-8

"""
    Grounded Model Exchange (GroMEt) schema for Function Networks

    This document defines the GroMEt Function Network data model. Note that Metadata is defined in separate spec.  __Using Swagger to Generate Class Structure__  To automatically generate Python or Java models corresponding to this document, you can use [swagger-codegen](https://swagger.io/tools/swagger-codegen/). This can be used to generate the client code based off of this spec, and in the process this will generate the data model class structure.  1. Install via the method described for your operating system    [here](https://github.com/swagger-api/swagger-codegen#Prerequisites).    Make sure to install a version after 3.0 that will support openapi 3. 2. Run swagger-codegen with the options in the example below.    The URL references where the yaml for this documentation is stored on    github. Make sure to replace CURRENT_VERSION with the correct version.    (The current version is `0.1.7`.)    To generate Java classes rather, change the `-l python` to `-l java`.    Change the value to the `-o` option to the desired output location.    ```    swagger-codegen generate -l python -o ./client -i https://raw.githubusercontent.com/ml4ai/automates-v2/master/docs/source/gromet_FN_v{CURRENT_VERSION}.yaml    ``` 3. Once it executes, the client code will be generated at your specified    location.    For python, the classes will be located in    `$OUTPUT_PATH/swagger_client/models/`.    For java, they will be located in    `$OUTPUT_PATH/src/main/java/io/swagger/client/model/`  If generating GroMEt schema data model classes in SKEMA (AutoMATES), then after generating the above, follow the instructions here: ``` <automates>/automates/model_assembly/gromet/model/README.md ```   # noqa: E501

    OpenAPI spec version: 0.1.9
    Contact: claytonm@arizona.edu
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""

import pprint
import re  # noqa: F401

import six
from skema.gromet.fn.gromet_object import GrometObject  # noqa: F401,E501

class GrometFNModule(GrometObject):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """
    """
    Attributes:
      swagger_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    swagger_types = {
        'schema': 'str',
        'schema_version': 'str',
        'name': 'str',
        'fn': 'GrometFN',
        'fn_array': 'list[GrometFN]',
        'metadata_collection': 'list[list[Metadata]]',
        'gromet_type': 'str'
    }
    if hasattr(GrometObject, "swagger_types"):
        swagger_types.update(GrometObject.swagger_types)

    attribute_map = {
        'schema': 'schema',
        'schema_version': 'schema_version',
        'name': 'name',
        'fn': 'fn',
        'fn_array': 'fn_array',
        'metadata_collection': 'metadata_collection',
        'gromet_type': 'gromet_type'
    }
    if hasattr(GrometObject, "attribute_map"):
        attribute_map.update(GrometObject.attribute_map)

    def __init__(self, schema='FN', schema_version=None, name=None, fn=None, fn_array=None, metadata_collection=None, gromet_type='GrometFNModule', *args, **kwargs):  # noqa: E501
        """GrometFNModule - a model defined in Swagger"""  # noqa: E501
        self._schema = None
        self._schema_version = None
        self._name = None
        self._fn = None
        self._fn_array = None
        self._metadata_collection = None
        self._gromet_type = None
        self.discriminator = None
        if schema is not None:
            self.schema = schema
        if schema_version is not None:
            self.schema_version = schema_version
        if name is not None:
            self.name = name
        if fn is not None:
            self.fn = fn
        if fn_array is not None:
            self.fn_array = fn_array
        if metadata_collection is not None:
            self.metadata_collection = metadata_collection
        if gromet_type is not None:
            self.gromet_type = gromet_type
        GrometObject.__init__(self, *args, **kwargs)

    @property
    def schema(self):
        """Gets the schema of this GrometFNModule.  # noqa: E501


        :return: The schema of this GrometFNModule.  # noqa: E501
        :rtype: str
        """
        return self._schema

    @schema.setter
    def schema(self, schema):
        """Sets the schema of this GrometFNModule.


        :param schema: The schema of this GrometFNModule.  # noqa: E501
        :type: str
        """

        self._schema = schema

    @property
    def schema_version(self):
        """Gets the schema_version of this GrometFNModule.  # noqa: E501


        :return: The schema_version of this GrometFNModule.  # noqa: E501
        :rtype: str
        """
        return self._schema_version

    @schema_version.setter
    def schema_version(self, schema_version):
        """Sets the schema_version of this GrometFNModule.


        :param schema_version: The schema_version of this GrometFNModule.  # noqa: E501
        :type: str
        """

        self._schema_version = schema_version

    @property
    def name(self):
        """Gets the name of this GrometFNModule.  # noqa: E501

        The name of the Function Network Module.   # noqa: E501

        :return: The name of this GrometFNModule.  # noqa: E501
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """Sets the name of this GrometFNModule.

        The name of the Function Network Module.   # noqa: E501

        :param name: The name of this GrometFNModule.  # noqa: E501
        :type: str
        """

        self._name = name

    @property
    def fn(self):
        """Gets the fn of this GrometFNModule.  # noqa: E501


        :return: The fn of this GrometFNModule.  # noqa: E501
        :rtype: GrometFN
        """
        return self._fn

    @fn.setter
    def fn(self, fn):
        """Sets the fn of this GrometFNModule.


        :param fn: The fn of this GrometFNModule.  # noqa: E501
        :type: GrometFN
        """

        self._fn = fn

    @property
    def fn_array(self):
        """Gets the fn_array of this GrometFNModule.  # noqa: E501

        Array of GrometFNs   # noqa: E501

        :return: The fn_array of this GrometFNModule.  # noqa: E501
        :rtype: list[GrometFN]
        """
        return self._fn_array

    @fn_array.setter
    def fn_array(self, fn_array):
        """Sets the fn_array of this GrometFNModule.

        Array of GrometFNs   # noqa: E501

        :param fn_array: The fn_array of this GrometFNModule.  # noqa: E501
        :type: list[GrometFN]
        """

        self._fn_array = fn_array

    @property
    def metadata_collection(self):
        """Gets the metadata_collection of this GrometFNModule.  # noqa: E501

        Table (array) of lists (arrays) of metadata, where each list in the Table-array represents the collection of metadata associated with a GrometFNModule object.   # noqa: E501

        :return: The metadata_collection of this GrometFNModule.  # noqa: E501
        :rtype: list[list[Metadata]]
        """
        return self._metadata_collection

    @metadata_collection.setter
    def metadata_collection(self, metadata_collection):
        """Sets the metadata_collection of this GrometFNModule.

        Table (array) of lists (arrays) of metadata, where each list in the Table-array represents the collection of metadata associated with a GrometFNModule object.   # noqa: E501

        :param metadata_collection: The metadata_collection of this GrometFNModule.  # noqa: E501
        :type: list[list[Metadata]]
        """

        self._metadata_collection = metadata_collection

    @property
    def gromet_type(self):
        """Gets the gromet_type of this GrometFNModule.  # noqa: E501


        :return: The gromet_type of this GrometFNModule.  # noqa: E501
        :rtype: str
        """
        return self._gromet_type

    @gromet_type.setter
    def gromet_type(self, gromet_type):
        """Sets the gromet_type of this GrometFNModule.


        :param gromet_type: The gromet_type of this GrometFNModule.  # noqa: E501
        :type: str
        """

        self._gromet_type = gromet_type

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.swagger_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(map(
                    lambda x: x.to_dict() if hasattr(x, "to_dict") else x,
                    value
                ))
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(map(
                    lambda item: (item[0], item[1].to_dict())
                    if hasattr(item[1], "to_dict") else item,
                    value.items()
                ))
            else:
                result[attr] = value
        if issubclass(GrometFNModule, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self):
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal"""
        if not isinstance(other, GrometFNModule):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
