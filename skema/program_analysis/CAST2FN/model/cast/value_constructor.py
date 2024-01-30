# coding: utf-8

"""
    SKEMA Common Abstract Syntax Tree (CAST)

    This document outlines the structure of the CAST that will be used as a generic representation of the semantics of a program written in any language. This will be used when creating functions networks from programs using the SKEMA Program Analysis pipeline.   __Generating Class Structure__    To automatically generate Python or Java models corresponding to this document, you can use [swagger-codegen](https://swagger.io/tools/swagger-codegen/). We can use this to generate client code based off of this spec that will also generate the class structure.    1. Install via the method described for your operating system [here](https://github.com/swagger-api/swagger-codegen#Prerequisites). Make sure to install a version after 3.0 that will support openapi 3.  2. Run swagger-codegen with the options in the example below. The URL references where the yaml for this documentation is stored on github. Make sure to replace CURRENT_VERSION with the correct version. To generate Java classes rather, change the `-l python` to `-l java`. Change the value to the `-o` option to the desired output location.       ```      swagger-codegen generate -l python -o ./client -i https://raw.githubusercontent.com/ml4ai/automates-v2/master/docs/source/cast_v{CURRENT_VERSION}.yaml      ```  3. Once it executes, the client code will be generated at your specified location. For python, the classes will be located in `$OUTPUT_PATH/swagger_client/models/`. For java, they will be located in `$OUTPUT_PATH/src/main/java/io/swagger/client/model/`      # noqa: E501

    OpenAPI spec version: 1.2.6
    
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""

import pprint
import re  # noqa: F401

import six
from skema.program_analysis.CAST2FN.model.cast.ast_node import AstNode  # noqa: F401,E501

class ValueConstructor(AstNode):
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
        'dim': 'object',
        'operator': 'object',
        'size': 'object',
        'initial_value': 'object'
    }
    if hasattr(AstNode, "swagger_types"):
        swagger_types.update(AstNode.swagger_types)

    attribute_map = {
        'dim': 'dim',
        'operator': 'operator',
        'size': 'size',
        'initial_value': 'initial_value'
    }
    if hasattr(AstNode, "attribute_map"):
        attribute_map.update(AstNode.attribute_map)

    def __init__(self, dim=None, operator=None, size=None, initial_value=None, *args, **kwargs):  # noqa: E501
        """ValueConstructor - a model defined in Swagger"""  # noqa: E501
        self._dim = None
        self._operator = None
        self._size = None
        self._initial_value = None
        self.discriminator = None
        if dim is not None:
            self.dim = dim
        if operator is not None:
            self.operator = operator
        if size is not None:
            self.size = size
        if initial_value is not None:
            self.initial_value = initial_value
        AstNode.__init__(self, *args, **kwargs)

    @property
    def dim(self):
        """Gets the dim of this ValueConstructor.  # noqa: E501


        :return: The dim of this ValueConstructor.  # noqa: E501
        :rtype: object
        """
        return self._dim

    @dim.setter
    def dim(self, dim):
        """Sets the dim of this ValueConstructor.


        :param dim: The dim of this ValueConstructor.  # noqa: E501
        :type: object
        """

        self._dim = dim

    @property
    def operator(self):
        """Gets the operator of this ValueConstructor.  # noqa: E501


        :return: The operator of this ValueConstructor.  # noqa: E501
        :rtype: object
        """
        return self._operator

    @operator.setter
    def operator(self, operator):
        """Sets the operator of this ValueConstructor.


        :param operator: The operator of this ValueConstructor.  # noqa: E501
        :type: object
        """

        self._operator = operator

    @property
    def size(self):
        """Gets the size of this ValueConstructor.  # noqa: E501


        :return: The size of this ValueConstructor.  # noqa: E501
        :rtype: object
        """
        return self._size

    @size.setter
    def size(self, size):
        """Sets the size of this ValueConstructor.


        :param size: The size of this ValueConstructor.  # noqa: E501
        :type: object
        """

        self._size = size

    @property
    def initial_value(self):
        """Gets the initial_value of this ValueConstructor.  # noqa: E501


        :return: The initial_value of this ValueConstructor.  # noqa: E501
        :rtype: object
        """
        return self._initial_value

    @initial_value.setter
    def initial_value(self, initial_value):
        """Sets the initial_value of this ValueConstructor.


        :param initial_value: The initial_value of this ValueConstructor.  # noqa: E501
        :type: object
        """

        self._initial_value = initial_value

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
        if issubclass(ValueConstructor, dict):
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
        if not isinstance(other, ValueConstructor):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
