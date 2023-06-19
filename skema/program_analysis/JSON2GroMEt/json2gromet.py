import json
import sys
import inspect
from typing import List, Dict, Any
from pathlib import Path

import skema.gromet.fn
import skema.gromet.metadata
from skema.gromet.fn import GrometFNModuleCollection


def json_to_gromet(path: str) -> GrometFNModuleCollection:
    """Given the path to a Gromet JSON file as input, return the Python data-model representation.
    The return type is the top-level GrometFNModuleCollection object.
    """

    # gromet_fn_map: Mapping from gromet_type field to Python type object. Supports all Gromet FN classes.
    # gromet_metadata_map: Mapping from metadata_type field to Python type object. Supports most Gromet Metadata classes (all with metadata_type field).
    # gromet_field_map: Mapping from a tuple of class fields to a Python type object. Supports remaining metadata types. Requires unique field signature between types.
    # NOTE: These dicts map to a Python type object, not an instance of a type.
    # An instance can be created by calling the constructor on the type object: gromet_fn_map[gromet_type]()
    gromet_fn_map = {}
    gromet_metadata_map = {}
    gromet_field_map = {}

    for fn_name, fn_object in inspect.getmembers(
        sys.modules["skema.gromet.fn"], inspect.isclass
    ):
        gromet_fn_map[fn_name] = fn_object

    for metadata_name, metadata_object in inspect.getmembers(
        sys.modules["skema.gromet.metadata"], inspect.isclass
    ):
        instance = metadata_object()
        if "metadata_type" in instance.attribute_map:
            gromet_metadata_map[instance.metadata_type] = metadata_object

    def get_obj_type(obj: Dict) -> Any:
        """Given a dictionary representing a Gromet object (i.e. BoxFunction), return an instance of that object.
        Returns None if the dictionary does not represent a Gromet object.
        """

        # First check if we already have a mapping to a data-class memeber. All Gromet FN and most Gromet Metadata classes will fall into this category.
        # There are a few Gromet Metadata fields such as Provenance that do not have a "metadata_type" field
        if "gromet_type" in obj:
            return gromet_fn_map[obj["gromet_type"]]()
        elif "metadata_type" in obj:
            return gromet_metadata_map[obj["metadata_type"]]()

        # If there is not a mapping to an object, we will check the fields to see if they match an existing class in the data-model.
        # For example: (id, box, metadata) would map to GrometPort
        obj_fields = tuple(obj.keys())
        if obj_fields in gromet_field_map:
            return gromet_field_map[obj_fields]()

        for gromet_name, gromet_object in inspect.getmembers(
            sys.modules[__name__], inspect.isclass
        ):
            found = True
            for field, value in obj.items():
                if not hasattr(gromet_object, field):
                    found = False
                    break
            if found:
                gromet_field_map[obj_fields] = gromet_object
                return gromet_object()

        return None

    def process_list(obj: List) -> List:
        """Handles importing a JSON list into the Gromet data-model"""

        # NOTE: The reason this is a seperate function from process_object() is to handle the case of nested Lists.
        # This is something that can occur in a few places, for example the metadata_collection field.
        out = []
        for element in obj:
            if isinstance(element, List):
                out.append(process_list(element))
            elif isinstance(element, Dict):
                # A dictionary may or may not represent a Gromet object.
                # Some LiteralValue representations are dictionaries too.
                obj_type = get_obj_type(element)
                if obj_type:
                    out.append(process_object(element, obj_type))
                else:
                    out.append(element)
            else:
                out.append(element)

        return out

    def process_object(obj: Dict, gromet_obj: Any):
        """Recursivly itterate over a Gromet JSON obj, importing the values into the fields of gromet_obj"""
        for field, value in obj.items():
            if isinstance(value, List):
                setattr(gromet_obj, field, process_list(value))
            elif isinstance(value, Dict):
                # Like the case in process_list, not all dicts will represent a Gromet object, so we check for that first.
                obj_type = get_obj_type(value)
                if obj_type:
                    setattr(gromet_obj, field, process_object(value, obj_type))
                else:
                    setattr(gromet_obj, field, value)
            else:
                setattr(gromet_obj, field, value)
        return gromet_obj

    json_object = json.loads(Path(path).read_text())
    return process_object(json_object, get_obj_type(json_object))
