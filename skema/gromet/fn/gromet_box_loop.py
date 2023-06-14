# coding: utf-8

"""
    Grounded Model Exchange (GroMEt) schema for Function Networks

    This document defines the GroMEt Function Network data model. Note that Metadata is defined in separate spec.  __Using Swagger to Generate Class Structure__  To automatically generate Python or Java models corresponding to this document, you can use [swagger-codegen](https://swagger.io/tools/swagger-codegen/). This can be used to generate the client code based off of this spec, and in the process this will generate the data model class structure.  1. Install via the method described for your operating system    [here](https://github.com/swagger-api/swagger-codegen#Prerequisites).    Make sure to install a version after 3.0 that will support openapi 3. 2. Run swagger-codegen with the options in the example below.    The URL references where the yaml for this documentation is stored on    github. Make sure to replace CURRENT_VERSION with the correct version.    (The current version is `0.1.7`.)    To generate Java classes rather, change the `-l python` to `-l java`.    Change the value to the `-o` option to the desired output location.    ```    swagger-codegen generate -l python -o ./client -i https://raw.githubusercontent.com/ml4ai/automates-v2/master/docs/source/gromet_FN_v{CURRENT_VERSION}.yaml    ``` 3. Once it executes, the client code will be generated at your specified    location.    For python, the classes will be located in    `$OUTPUT_PATH/swagger_client/models/`.    For java, they will be located in    `$OUTPUT_PATH/src/main/java/io/swagger/client/model/`  If generating GroMEt schema data model classes in SKEMA (AutoMATES), then after generating the above, follow the instructions here: ``` <automates>/automates/model_assembly/gromet/model/README.md ```   # noqa: E501

    OpenAPI spec version: 0.1.7
    Contact: claytonm@arizona.edu
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""

import pprint
import re  # noqa: F401

import six
from skema.gromet.fn.gromet_box import GrometBox  # noqa: F401,E501

class GrometBoxLoop(GrometBox):
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
        'pre': 'int',
        'condition': 'int',
        'body': 'int',
        'post': 'int',
        'gromet_type': 'str'
    }
    if hasattr(GrometBox, "swagger_types"):
        swagger_types.update(GrometBox.swagger_types)

    attribute_map = {
        'pre': 'pre',
        'condition': 'condition',
        'body': 'body',
        'post': 'post',
        'gromet_type': 'gromet_type'
    }
    if hasattr(GrometBox, "attribute_map"):
        attribute_map.update(GrometBox.attribute_map)

    def __init__(self, pre=None, condition=None, body=None, post=None, gromet_type='GrometBoxLoop', *args, **kwargs):  # noqa: E501
        """GrometBoxLoop - a model defined in Swagger"""  # noqa: E501
        self._pre = None
        self._condition = None
        self._body = None
        self._post = None
        self._gromet_type = None
        self.discriminator = None
        if pre is not None:
            self.pre = pre
        if condition is not None:
            self.condition = condition
        if body is not None:
            self.body = body
        if post is not None:
            self.post = post
        if gromet_type is not None:
            self.gromet_type = gromet_type
        GrometBox.__init__(self, *args, **kwargs)

    @property
    def pre(self):
        """Gets the pre of this GrometBoxLoop.  # noqa: E501

        OPTIONAL. The index to the entry in the `fn_array` table of the GrometFNModule representing the GrometFN implementing the pre (aka 'init') (Function) of the Loop. This enables support of for-loops, where many languages have provided an extension to loops that have an initialization that is only in the scope of the loop, but not part of the loop body.   # noqa: E501

        :return: The pre of this GrometBoxLoop.  # noqa: E501
        :rtype: int
        """
        return self._pre

    @pre.setter
    def pre(self, pre):
        """Sets the pre of this GrometBoxLoop.

        OPTIONAL. The index to the entry in the `fn_array` table of the GrometFNModule representing the GrometFN implementing the pre (aka 'init') (Function) of the Loop. This enables support of for-loops, where many languages have provided an extension to loops that have an initialization that is only in the scope of the loop, but not part of the loop body.   # noqa: E501

        :param pre: The pre of this GrometBoxLoop.  # noqa: E501
        :type: int
        """

        self._pre = pre

    @property
    def condition(self):
        """Gets the condition of this GrometBoxLoop.  # noqa: E501

        The index to the entry in the `fn_array` table of the GrometFNModule representing the GrometFN implementing the condition (Predicate) of the Loop.   # noqa: E501

        :return: The condition of this GrometBoxLoop.  # noqa: E501
        :rtype: int
        """
        return self._condition

    @condition.setter
    def condition(self, condition):
        """Sets the condition of this GrometBoxLoop.

        The index to the entry in the `fn_array` table of the GrometFNModule representing the GrometFN implementing the condition (Predicate) of the Loop.   # noqa: E501

        :param condition: The condition of this GrometBoxLoop.  # noqa: E501
        :type: int
        """

        self._condition = condition

    @property
    def body(self):
        """Gets the body of this GrometBoxLoop.  # noqa: E501

        The index to the entry in the `fn_array` table of the GrometFNModule representing the GrometFN implementing the body (Function) of the Loop.   # noqa: E501

        :return: The body of this GrometBoxLoop.  # noqa: E501
        :rtype: int
        """
        return self._body

    @body.setter
    def body(self, body):
        """Sets the body of this GrometBoxLoop.

        The index to the entry in the `fn_array` table of the GrometFNModule representing the GrometFN implementing the body (Function) of the Loop.   # noqa: E501

        :param body: The body of this GrometBoxLoop.  # noqa: E501
        :type: int
        """

        self._body = body

    @property
    def post(self):
        """Gets the post of this GrometBoxLoop.  # noqa: E501

        OPTIONAL. The index to the entry in the `fn_array` table of the GrometFNModule representing the GrometFN implementing the post-loop (Function) just prior to exiting the Loop. This is useful for representing control structures such as in Fortran where loop variable update is advanced one more step than what Python-style iterators support, and where that updated value is available outside the loop.   # noqa: E501

        :return: The post of this GrometBoxLoop.  # noqa: E501
        :rtype: int
        """
        return self._post

    @post.setter
    def post(self, post):
        """Sets the post of this GrometBoxLoop.

        OPTIONAL. The index to the entry in the `fn_array` table of the GrometFNModule representing the GrometFN implementing the post-loop (Function) just prior to exiting the Loop. This is useful for representing control structures such as in Fortran where loop variable update is advanced one more step than what Python-style iterators support, and where that updated value is available outside the loop.   # noqa: E501

        :param post: The post of this GrometBoxLoop.  # noqa: E501
        :type: int
        """

        self._post = post

    @property
    def gromet_type(self):
        """Gets the gromet_type of this GrometBoxLoop.  # noqa: E501


        :return: The gromet_type of this GrometBoxLoop.  # noqa: E501
        :rtype: str
        """
        return self._gromet_type

    @gromet_type.setter
    def gromet_type(self, gromet_type):
        """Sets the gromet_type of this GrometBoxLoop.


        :param gromet_type: The gromet_type of this GrometBoxLoop.  # noqa: E501
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
        if issubclass(GrometBoxLoop, dict):
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
        if not isinstance(other, GrometBoxLoop):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
