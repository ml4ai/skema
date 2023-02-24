# coding: utf-8

"""
    GroMEt Metadata spec

    Grounded Model Exchange (GroMEt) Metadata schema specification  __Using Swagger to Generate Class Structure__  To automatically generate Python or Java models corresponding to this document, you can use [swagger-codegen](https://swagger.io/tools/swagger-codegen/). We can use this to generate client code based off of this spec that will also generate the class structure.  1. Install via the method described for your operating system    [here](https://github.com/swagger-api/swagger-codegen#Prerequisites).    Make sure to install a version after 3.0 that will support openapi 3. 2. Run swagger-codegen with the options in the example below.    The URL references where the yaml for this documentation is stored on    github. Make sure to replace CURRENT_VERSION with the correct version.    (The current version is `0.1.4`.)    To generate Java classes rather, change the `-l python` to `-l java`.    Change the value to the `-o` option to the desired output location.    ```    swagger-codegen generate -l python -o ./client -i https://raw.githubusercontent.com/ml4ai/skema-v2/master/docs/source/gromet_metadata_v{CURRENT_VERSION}.yaml    ``` 3. Once it executes, the client code will be generated at your specified    location.    For python, the classes will be located in    `$OUTPUT_PATH/swagger_client/models/`.    For java, they will be located in    `$OUTPUT_PATH/src/main/java/io/swagger/client/model/`  If generating GroMEt Metadata schema data model classes in SKEMA (AutoMATES), then after generating the above, follow the instructions here: ``` <skema>/skema/model_assembly/gromet/metadata/README.md ```   # noqa: E501

    OpenAPI spec version: 0.1.5
    Contact: claytonm@arizona.edu
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""

import pprint
import re  # noqa: F401

import six


class EquationExtraction(object):
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
        "source_type": "str",
        "document_reference_uid": "str",
        "equation_number": "int",
    }

    attribute_map = {
        "source_type": "source_type",
        "document_reference_uid": "document_reference_uid",
        "equation_number": "equation_number",
    }

    def __init__(
        self,
        source_type="equation_document_source",
        document_reference_uid=None,
        equation_number=None,
    ):  # noqa: E501
        """EquationExtraction - a model defined in Swagger"""  # noqa: E501
        self._source_type = None
        self._document_reference_uid = None
        self._equation_number = None
        self.discriminator = None
        if source_type is not None:
            self.source_type = source_type
        if document_reference_uid is not None:
            self.document_reference_uid = document_reference_uid
        if equation_number is not None:
            self.equation_number = equation_number

    @property
    def source_type(self):
        """Gets the source_type of this EquationExtraction.  # noqa: E501


        :return: The source_type of this EquationExtraction.  # noqa: E501
        :rtype: str
        """
        return self._source_type

    @source_type.setter
    def source_type(self, source_type):
        """Sets the source_type of this EquationExtraction.


        :param source_type: The source_type of this EquationExtraction.  # noqa: E501
        :type: str
        """

        self._source_type = source_type

    @property
    def document_reference_uid(self):
        """Gets the document_reference_uid of this EquationExtraction.  # noqa: E501

        uid of the source DocumentReference  # noqa: E501

        :return: The document_reference_uid of this EquationExtraction.  # noqa: E501
        :rtype: str
        """
        return self._document_reference_uid

    @document_reference_uid.setter
    def document_reference_uid(self, document_reference_uid):
        """Sets the document_reference_uid of this EquationExtraction.

        uid of the source DocumentReference  # noqa: E501

        :param document_reference_uid: The document_reference_uid of this EquationExtraction.  # noqa: E501
        :type: str
        """

        self._document_reference_uid = document_reference_uid

    @property
    def equation_number(self):
        """Gets the equation_number of this EquationExtraction.  # noqa: E501

        Number of equation (relative to document source)  # noqa: E501

        :return: The equation_number of this EquationExtraction.  # noqa: E501
        :rtype: int
        """
        return self._equation_number

    @equation_number.setter
    def equation_number(self, equation_number):
        """Sets the equation_number of this EquationExtraction.

        Number of equation (relative to document source)  # noqa: E501

        :param equation_number: The equation_number of this EquationExtraction.  # noqa: E501
        :type: int
        """

        self._equation_number = equation_number

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.swagger_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(
                    map(
                        lambda x: x.to_dict() if hasattr(x, "to_dict") else x,
                        value,
                    )
                )
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(
                    map(
                        lambda item: (item[0], item[1].to_dict())
                        if hasattr(item[1], "to_dict")
                        else item,
                        value.items(),
                    )
                )
            else:
                result[attr] = value
        if issubclass(EquationExtraction, dict):
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
        if not isinstance(other, EquationExtraction):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
