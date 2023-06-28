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
import re

from skema.utils.fold import dictionary_to_gromet_json, del_nulls

def normalize_module_path(path: str):
    # The module paths in the GroMEt FN are dotted
    # We need slashes for the comments dictionary
    return path.replace(".","/")

def normalize_extraction_names(extraction: dict):
    # Removes extraneous characters and filename extensions
    # from the extraction dictionary
    # Currently removes, ".py" extension 
    # and "./" from the keys
    return { k.replace(".py","").replace("./", "") : v for k,v in extraction.items() }

def strip_id(func_name):
    # Given a function name that ends with "_id###" where ### is a number
    # We remove that sequence of characters from the function name
    # The id is appended by the GroMEt generation, and so we can safely remove it
    # because we need the pure name of the function and not the identifier part
    
    # Only strip the id if the func_name contains the pattern "_id###..." which
    # is appended by the Gromet generation
    if re.search("_id\d+", func_name):
        to_ret = list(func_name)
        to_ret.reverse()
        i = 0
        while i < len(to_ret) and to_ret[i] != '_':
            to_ret[i] = ''
            i += 1
        to_ret[i] = ''
        to_ret.reverse()
        return ''.join(to_ret)
    else:
        return func_name

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
    # NOTE: not used now but will be in the future
    gromet_metadata.append([new_metadata])
    return len(gromet_metadata)

def align_gromet_elements(gromet_metadata, gromet_comments, gromet_elements):
    # Gromet elements are generic enough that we can use
    # the same function to iterate through gromet elements
    # and append comment metadata as necessary
    # TODO: associate code_file_reference_uid
    if gromet_elements != None:
        for elem in gromet_elements:
            if elem.metadata != None: 
                metadatum = gromet_metadata[elem.metadata - 1]
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

                        # Find a comment metadata associated with that

def align_fn(gromet_metadata, gromet_comments, gromet_fn):
    # Align the GroMEt b table
    # We might be able to use the generic aligner but for now we align 
    # independently 
    if gromet_fn.b != None:
        for box in gromet_fn.b:
            if box.metadata != None:
                metadatum = gromet_metadata[box.metadata - 1]
                source_ref = find_source_code_reference(metadatum)
                if source_ref != None:
                    # NOTE: Look at line_begin in the source ref info
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


    # All these GroMEt elements all have metadata stored in the same way
    # So we can align any comments for all these using a generic aligner
    align_gromet_elements(gromet_metadata, gromet_comments, gromet_fn.bf)
    align_gromet_elements(gromet_metadata, gromet_comments, gromet_fn.opi)
    align_gromet_elements(gromet_metadata, gromet_comments, gromet_fn.opo)
    align_gromet_elements(gromet_metadata, gromet_comments, gromet_fn.pif)
    align_gromet_elements(gromet_metadata, gromet_comments, gromet_fn.pof)
    align_gromet_elements(gromet_metadata, gromet_comments, gromet_fn.pic)
    align_gromet_elements(gromet_metadata, gromet_comments, gromet_fn.poc)

    # Check if the current FN has a name, and if it's associated
    # With a docstring, align the docstring with it if that's the case
    if gromet_fn.b != None and gromet_fn.b[0].name != None:
        func_name = gromet_fn.b[0].name
        normalized_func_name = strip_id(func_name)
        if normalized_func_name in gromet_comments["docstrings"].keys():
            metadata_idx = gromet_fn.b[0].metadata
            if metadata_idx != None:
                docstring = "".join(gromet_comments["docstrings"][normalized_func_name])

                source_comment = SourceCodeComment(
                    comment = docstring,
                    comment_type = CommentType.DOCSTRING,
                    context_function_name=normalized_func_name,
                    code_file_reference_uid=None,
                    line_begin=source_ref.line_begin,
                    line_end=source_ref.line_end,
                    col_begin=source_ref.col_begin,
                    col_end=source_ref.col_end
                )

                gromet_metadata[metadata_idx-1].append(source_comment)
    

def find_fn(gromet_modules, fn_name):
    # Given the gromet_modules list of FNs, we find fn_name in it
    modified_fn_name = fn_name.split("/")[-1]

    for FN in gromet_modules:
        if modified_fn_name == FN.name:
            return FN

    return None


def align_full_system(gromet_obj, extraction, extraction_file_name):
    # Comments extraction file holds comments for all files in the system

    # The extracted comments json file can appear in two ways:
    #   - extractions for a single file:
    #     A single file consists of one top level dictionary containing 
    #     the comments and docstrings for that file
    #   - extractions for a multi file
    #     A multi file consists of a top level dictionary that maps each file
    #     in the system to a dictionary containing the comments and docstrings for that file
    # We can check what kind of extracted comments file we have by checking the structure of the dictionary
    if "comments" in extraction.keys() and "docstrings" in extraction.keys(): 
        # Single file system
        # NOTE: We assume for the moment that if we're aligning a single file that
        # The corresponding GroMEt has exactly one module
        
        if len(gromet_obj.modules[0]) != 1:
            raise NotImplementedError("Single file alignment from a multi module GroMEt system not supported yet")

        module_FN = gromet_obj.modules[0]
        if module_FN != None:
            FN_metadata = module_FN.metadata_collection
            align_fn(FN_metadata, extraction, module_FN.fn)

            if len(module_FN.fn_array) > 0:
                for FN in module_FN.fn_array:
                    align_fn(FN_metadata, extraction, FN)
    else:
        # Multi-file system
        extraction = normalize_extraction_names(extraction)
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
        
def process_alignment(gromet_json, comments_json):
    # Given a GroMEt json and a comments json 
    # We run the alignment on the GroMEt to unify the comments with
    # The gromet JSON
    gromet_object = json_to_gromet(gromet_json)
    align_full_system(gromet_object, comments_json)

    return gromet_object

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

    align_full_system(gromet_object, comments_json)

    # Write out the gromet with the comments
    with open(args.gromet, "w") as f:
        gromet_collection_dict = gromet_object.to_dict()
        f.write(
            dictionary_to_gromet_json(del_nulls(gromet_collection_dict))
        )

