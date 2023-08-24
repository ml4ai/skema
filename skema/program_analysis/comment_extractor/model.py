from typing import List, Dict, Union, Optional
from pydantic import BaseModel, Field


class SingleLineComment(BaseModel):
    content: str = Field(
        ...,
        description="The content of the single line comment.",
        example="#Single line Fortran comment",
    )
    line_number: int = Field(
        ..., description="The line number where the comment appears.", example=10
    )


class MultiLineComment(BaseModel):
    content: List[str] = Field(
        ...,
        description="The content of the multi line comment.",
        example=["/*", "Multi-line", "C", "comment", "*/"],
    )
    start_line_number: int = Field(
        ...,
        description="The line number where the multi line comment starts.",
        example=15,
    )
    end_line_number: int = Field(
        ...,
        description="The line number where the multi line comment ends.",
        example=20,
    )


class Docstring(BaseModel):
    content: List[str] = Field(
        ...,
        description="The content of the docstring.",
        example=[
            '"""',
            "This is a Python docstring.",
            "It provides information about a function.",
            '"""',
        ],
    )
    function_name: str = Field(
        ...,
        description="The name of the function that the docstring belongs to.",
        example="foo",
    )
    start_line_number: int = Field(
        ..., description="The line number where the docstring starts.", example=25
    )
    end_line_number: int = Field(
        ..., description="The line number where the docstring ends.", example=30
    )


class SingleFileCommentRequest(BaseModel):
    source: str = Field(
        ...,
        description="The source code of the file.",
        example="def foo():\n    # Single line Python comment\n    pass",
    )
    language: str = Field(
        ...,
        description="The programming language of the source code.",
        example="python",
    )


class SingleFileCommentResponse(BaseModel):
    single: List[SingleLineComment] = Field(
        ...,
        description="List of single line comments in the file.",
        example=[
            {"content": "# Comment 1", "line_number": 10},
            {"content": "# Comment 2", "line_number": 15},
        ],
    )
    multi: List[MultiLineComment] = Field(
        ...,
        description="List of multi line comments in the file.",
        example=[
            {
                "content": ["/*", "Multi-line", "C comment", "*/"],
                "start_line_number": 5,
                "end_line_number": 8,
            }
        ],
    )
    docstring: List[Docstring] = Field(
        ...,
        description="List of docstrings in the file.",
        example=[
            {
                "content": ['"""', "This is a sample", "docstring.", '"""'],
                "function_name": "my_function",
                "start_line_number": 20,
                "end_line_number": 25,
            }
        ],
    )


class MultiFileCommentRequest(BaseModel):
    files: Dict[str, SingleFileCommentRequest] = Field(
        ...,
        description="Dictionary of file names and SingleFileCommentRequest objects.",
        example={
            "file1.py": {
                "source": "def func():\n    # Comment\n    pass",
                "language": "Python",
            },
            "file2.c": {"source": "/*\nMulti-line\ncomment\n*/", "language": "C"},
        },
    )


class MultiFileCommentResponse(BaseModel):
    files: Dict[str, SingleFileCommentResponse] = Field(
        ...,
        description="Dictionary of file names and SingleFileCommentResponse objects.",
        example={
            "file1.py": {
                "single": [{"content": "# Comment 1", "line_number": 5}],
                "multi": [],
                "docstring": [],
            },
            "file2.c": {
                "single": [],
                "multi": [
                    {
                        "content": ["/*", "Multi-line", "C comment", "*/"],
                        "start_line_number": 10,
                        "end_line_number": 13,
                    }
                ],
                "docstring": [],
            },
        },
    )


class SupportedLanguage(BaseModel):
    name: str = Field(
        ...,
        description="The name of the supported programming language.",
        example="python",
    )
    extensions: List[str] = Field(
        ...,
        description="List of file extensions supported for corresponding language",
    )
    single: bool = Field(
        ...,
        description="Indicates whether single line comments are supported for this language.",
    )
    multi: bool = Field(
        ...,
        description="Indicates whether multi line comments are supported for this language.",
    )
    docstring: bool = Field(
        ..., description="Indicates whether docstrings are supported for this language."
    )


class SupportedLanguageResponse(BaseModel):
    languages: List[SupportedLanguage] = Field(
        ...,
        description="List of SupportedLanguage objects representing the supported languages.",
        example=[
            {"name": "python", "single": True, "multi": False, "docstring": True},
            {"name": "c", "single": True, "multi": True, "docstring": True},
        ],
    )


CodeComments = Union[SingleFileCommentRequest, MultiFileCommentRequest]
