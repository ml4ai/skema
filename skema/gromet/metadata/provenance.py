# coding: utf-8

"""
    GroMEt Metadata spec

    Grounded Model Exchange (GroMEt) Metadata schema specification  __Using Swagger to Generate Class Structure__  To automatically generate Python or Java models corresponding to this document, you can use [swagger-codegen](https://swagger.io/tools/swagger-codegen/). We can use this to generate client code based off of this spec that will also generate the class structure.  1. Install via the method described for your operating system    [here](https://github.com/swagger-api/swagger-codegen#Prerequisites).    Make sure to install a version after 3.0 that will support openapi 3. 2. Run swagger-codegen with the options in the example below.    The URL references where the yaml for this documentation is stored on    github. Make sure to replace CURRENT_VERSION with the correct version.    (The current version is `0.1.4`.)    To generate Java classes rather, change the `-l python` to `-l java`.    Change the value to the `-o` option to the desired output location.    ```    swagger-codegen generate -l python -o ./client -i https://raw.githubusercontent.com/ml4ai/automates-v2/master/docs/source/gromet_metadata_v{CURRENT_VERSION}.yaml    ``` 3. Once it executes, the client code will be generated at your specified    location.    For python, the classes will be located in    `$OUTPUT_PATH/swagger_client/models/`.    For java, they will be located in    `$OUTPUT_PATH/src/main/java/io/swagger/client/model/`  If generating GroMEt Metadata schema data model classes in SKEMA (AutoMATES), then after generating the above, follow the instructions here: ``` <automates>/automates/model_assembly/gromet/metadata/README.md ```   # noqa: E501

    OpenAPI spec version: 0.1.8
    Contact: claytonm@arizona.edu
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""

import pprint
import re  # noqa: F401

import six

class Provenance(object):
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
        'gromet_type': 'str',
        'method': 'str',
        'timestamp': 'datetime'
    }

    attribute_map = {
        'gromet_type': 'gromet_type',
        'method': 'method',
        'timestamp': 'timestamp'
    }

    def __init__(self, gromet_type='Provenance', method=None, timestamp=None):  # noqa: E501
        """Provenance - a model defined in Swagger"""  # noqa: E501
        self._gromet_type = None
        self._method = None
        self._timestamp = None
        self.discriminator = None
        if gromet_type is not None:
            self.gromet_type = gromet_type
        if method is not None:
            self.method = method
        if timestamp is not None:
            self.timestamp = timestamp

    @property
    def gromet_type(self):
        """Gets the gromet_type of this Provenance.  # noqa: E501


        :return: The gromet_type of this Provenance.  # noqa: E501
        :rtype: str
        """
        return self._gromet_type

    @gromet_type.setter
    def gromet_type(self, gromet_type):
        """Sets the gromet_type of this Provenance.


        :param gromet_type: The gromet_type of this Provenance.  # noqa: E501
        :type: str
        """

        self._gromet_type = gromet_type

    @property
    def method(self):
        """Gets the method of this Provenance.  # noqa: E501

        The inference method (with version) used to derive data.   # noqa: E501

        :return: The method of this Provenance.  # noqa: E501
        :rtype: str
        """
        return self._method

    @method.setter
    def method(self, method):
        """Sets the method of this Provenance.

        The inference method (with version) used to derive data.   # noqa: E501

        :param method: The method of this Provenance.  # noqa: E501
        :type: str
        """

        self._method = method

    @property
    def timestamp(self):
        """Gets the timestamp of this Provenance.  # noqa: E501

        Date and time that metadata was generated.   # noqa: E501

        :return: The timestamp of this Provenance.  # noqa: E501
        :rtype: datetime
        """
        return self._timestamp

    @timestamp.setter
    def timestamp(self, timestamp):
        """Sets the timestamp of this Provenance.

        Date and time that metadata was generated.   # noqa: E501

        :param timestamp: The timestamp of this Provenance.  # noqa: E501
        :type: datetime
        """

        self._timestamp = timestamp

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
        if issubclass(Provenance, dict):
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
        if not isinstance(other, Provenance):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
