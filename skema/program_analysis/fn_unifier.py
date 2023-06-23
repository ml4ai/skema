# Function Network (FN) Unifier
# Given a GroMEt FN JSON and a Comments JSON file, we 'unify' them by
# 1. Extracting the GroMEt JSON and turning it back into an object
# 2. Extracting the comments JSON file
# 3. Appending all comments from the comments JSON into the respective MetadataCollections for each FN

from skema.program_analysis.JSON2GroMEt.json2gromet import json_to_gromet
from skema.gromet.metadata.source_code_comment import SourceCodeComment
from skema.gromet.metadata.source_code_reference import SourceCodeReference
from skema.gromet.metadata.comment_type import CommentType

import argparse
import json

from skema.utils.fold import dictionary_to_gromet_json, del_nulls

def normalize_module_path(path: str):
    # The module paths in the GroMEt FN are dotted
    # We need slashes for the dictionary
    return path.replace(".","/")

def normalize_extraction_names(extraction: dict):
    # Removes extraneous characters and filename extensions
    # from the extraction dictionary
    # Currently removes, ".py" extension 
    # and "./" from the keys
    return { k.replace(".py","").replace("./", "") : v for k,v in extraction.items() }

def find_source_code_reference(metadatum):
    # Find a SourceCodeReference metadata in the metadatum entry 
    # we're looking at
    for elem in metadatum:
        if isinstance(elem, SourceCodeReference):
            return elem
    
    return None

def find_comment(comments, line_num):
    # Given the comments for a file and a line number, we find
    # the comment that goes with that line number, if it exists
    for entry in comments["comments"]:
        if entry["line_number"] == line_num:
            return entry["contents"]

    return None

def insert_metadata(gromet_metadata, new_metadata):
    # Appends a new metadata to the end of the gromet_metadata
    gromet_metadata.append([new_metadata])
    return len(gromet_metadata)


def align_fn(gromet_metadata, gromet_comments, gromet_fn):
    for box in gromet_fn.b:
        if box.metadata != None and type(box.metadata) == int:
            metadatum = gromet_metadata[box.metadata - 1]
            source_ref = find_source_code_reference(metadatum)
            if source_ref != None:
                # Look at line_begin
                line_start = source_ref.line_begin
                comment = find_comment(gromet_comments, line_start)
                if comment != None:
                    source_comment = SourceCodeComment(
                        comment = comment,
                        comment_type = CommentType.OTHER,
                        context_function_name=None,
                        code_file_reference_uid=None,
                        line_begin=source_ref.line_begin,
                        line_end=source_ref.line_end,
                        col_begin=source_ref.col_begin,
                        col_end=source_ref.col_end
                    )

                    metadatum.append(source_comment)
                    # print(f"Comment ({comment}) found at line {line_start}")

                # Find a comment metadata associated with that

    # Check if the current FN has a name, and if it's associated
    # With a docstring, align the docstring with it if that's the case
    if gromet_fn.b != None and gromet_fn.b[0].name != None:
        func_name = gromet_fn.b[0].name
        if func_name in gromet_comments["docstrings"].keys():
            metadata_idx = gromet_fn.b[0].metadata
            if metadata_idx != None:
                print(func_name)
                print(gromet_fn.b[0].metadata)
                docstring = gromet_comments["docstrings"][func_name]

                source_comment = SourceCodeComment(
                    comment = docstring,
                    comment_type = CommentType.DOCSTRING,
                    context_function_name=func_name,
                    code_file_reference_uid=None,
                    line_begin=source_ref.line_begin,
                    line_end=source_ref.line_end,
                    col_begin=source_ref.col_begin,
                    col_end=source_ref.col_end
                )

                gromet_metadata[metadata_idx-1].append
    

def find_fn(gromet_modules, fn_name):
    # Given the gromet_modules list of FNs, we find fn_name in it
    modified_fn_name = fn_name.split("/")[-1]

    for FN in gromet_modules:
        if modified_fn_name == FN.name:
            return FN

    return None


def append_comments(gromet_obj, extraction):
    # Comments extraction file holds comments for all files in the system
    extraction = normalize_extraction_names(extraction)
    # print(gromet_obj.module_index)

    for module in gromet_obj.module_index:
        # Go through each file in the GroMEt FN
        normalized_path = normalize_module_path(module)
        if normalized_path in extraction.keys():
            # Find the current FN in the collection
            module_FN = find_fn(gromet_obj.modules, normalized_path)
            if module_FN != None:
                file_comments = extraction[normalized_path]
                
                FN_metadata = module_FN.metadata_collection
                align_fn(FN_metadata, file_comments, module_FN.fn)

                if len(module_FN.fn_array) > 0:
                    for FN in module_FN.fn_array:
                        align_fn(FN_metadata, file_comments, FN)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gromet", type=str, help="Path to a GroMEt JSON file"
    )
    parser.add_argument(
        "--comments", type=str, help="Path to a Comments JSON file"
    )
    args = parser.parse_args()

    # Get the GroMEt JSON and turn it back into an object
    gromet_object = json_to_gromet(args.gromet)
    
    # Get the comments data from the JSON file
    comments_file = open(args.comments, "r")
    comments_json = json.load(comments_file)
    comments_file.close()

    append_comments(gromet_object, comments_json)

    # Write out the gromet with the comments
    with open(args.gromet, "w") as f:
        gromet_collection_dict = gromet_object.to_dict()
        f.write(
            dictionary_to_gromet_json(del_nulls(gromet_collection_dict))
        )

