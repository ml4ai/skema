from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field


class SingleLineComment(BaseModel):
    contents: str
    line_number: int


class SingleFileCodeComments(BaseModel):
    docstrings: Dict[str, List[str]] = Field(
        description="A dictionary mapping a function name (str) to its associated docstring (List[str])",
        examples=[{"fun1": ["Inputs: x,y", "Outputs: z"]}],
    )
    comments: List[SingleLineComment] = Field(
        description="A list of comments, where each comment has a 'line_number' (int) and 'contents' (str) field",
        examples=[{"contents": "Hello World!", "line_number": 0}],
    )


class MultiFileCodeComments(BaseModel):
    files: Dict[str, SingleFileCodeComments] = Field(
        description="Dictionary mapping file name (str) to extracted comments (SingleFileCodeComments)",
        examples=[{
            "file1.py": {
                "comments": [{"contents": "Hello World!", "line_number": 0}],
                "docstrings": {"fun1": ["Inputs: x,y", "Outputs: z"]},
            }
        }],
    )


CodeComments = Union[SingleFileCodeComments, MultiFileCodeComments]
