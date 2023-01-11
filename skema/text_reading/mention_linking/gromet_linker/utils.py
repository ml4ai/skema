from .gromet_helper import GrometHelper
from .provenance_helper import ProvenanceHelper
from .text_reading_linker import TextReadingLinker
from skema.gromet.fn import GrometFNModule
from skema.gromet.metadata import (
    SourceCodeComment,
    TextGrounding,
    TextExtraction,
    TextDescription,
    TextLiteralValue,
    TextualDocumentCollection,
    TextualDocumentReference,
    TextUnits,
)
from typing import Optional, Tuple

import itertools as it


class Utils:
    @staticmethod
    def get_code_file_ref(
        comments_file_name: str, gromet_fn_module: GrometFNModule
    ) -> Optional[str]:
        """Fetches the UUID of the code_file_reference that matches the one from the comments file"""

        metadata_collection = GrometHelper.get_element_metadata(
            gromet_fn_module, gromet_fn_module
        )
        code_collection = None
        uid = None

        for metadata in metadata_collection:
            if metadata.metadata_type == "source_code_collection":
                code_collection = metadata
                break

        if code_collection:
            prefix = ".".join(comments_file_name.split(".")[:-1])

            for file in code_collection.files:
                if file.name.startswith(prefix):
                    uid = file.uid
                    break

        return uid

    @staticmethod
    def get_doc_file_ref(
        provenance_helper: ProvenanceHelper,
        uid_stamper,
        scored_mention,
        linker: TextReadingLinker,
        gromet_fn_module: GrometFNModule,
    ) -> Optional[str]:
        """Fetches the UUID of the text doc reference that matches the one from the mention's file"""

        mention, _ = scored_mention
        metadata_collection = GrometHelper.get_element_metadata(
            gromet_fn_module, gromet_fn_module
        )
        text_collection = None
        uid = None

        for metadata in metadata_collection:
            if metadata.metadata_type == "textual_document_collection":
                text_collection = metadata
                break

        # If the text doc collection doesn't exist, then add it
        if not text_collection:
            text_collection = TextualDocumentCollection(
                provenance=provenance_helper.build_embedding(),
                documents=list(),
            )

            metadata_collection.append(text_collection)

        if text_collection:
            mention_doc = linker.documents[mention["document"]]
            doc_id = mention_doc["id"]
            existing_docs_refs = text_collection.documents

            doc_ref = None
            for dr in existing_docs_refs:
                if doc_id == dr.cosmos_id:
                    doc_ref = dr
                    break

            # Create a new TextDocumentReference if it doesn't exist yet
            if not doc_ref:
                # TODO Figure out all the correct values here
                doc_ref = TextualDocumentReference(
                    uid=uid_stamper.stamp(doc_id),
                    global_reference_id="TBD",
                    cosmos_id=doc_id,
                    cosmos_version_number=0.1,
                    skema_id=0.1,
                )

                existing_docs_refs.append(doc_ref)

            uid = doc_ref.uid

        return uid

    @staticmethod
    def build_comment_metadata(
        provenance_helper: ProvenanceHelper,
        comment: str,
        code_file_ref: str,
        element,
        gromet: GrometFNModule,
    ):
        # TODO: Differentiate between line comments and docstrings
        line, text = None, comment
        md = SourceCodeComment(
            provenance=provenance_helper.build_comment(),
            code_file_reference_uid=code_file_ref,
            comment=text,
            line_begin=line,
            line_end=line,  # TODO Should it be +1?
        )
        Utils.attach_metadata(md, element, gromet)

    @staticmethod
    def build_line_comment_metadata(
        provenance_helper: ProvenanceHelper,
        line_comment: Tuple[int, str],
        code_file_ref: str,
        element,
        gromet: GrometFNModule,
    ):
        # TODO: Differentiate between line comments and docstrings
        line, text = line_comment
        md = SourceCodeComment(
            provenance=provenance_helper.build_comment(),
            code_file_reference_uid=code_file_ref,
            comment=text,
            line_begin=line,
            line_end=line,  # TODO Should it be +1?
        )

        Utils.attach_metadata(md, element, gromet)

    @staticmethod
    def build_textreading_mention_metadata(
        provenance_helper: ProvenanceHelper,
        scored_mention,
        doc_file_ref: str,
        element,
        gromet: GrometFNModule,
    ):

        mention, score = scored_mention

        page_num, block = None, None
        for attachment in mention["attachments"]:
            if "pageNum" in attachment:
                page_num = attachment["pageNum"][0]
            if "blockIdx" in attachment:
                block = attachment["blockIdx"][0]

        text_extraction = TextExtraction(
            document_reference_uid=doc_file_ref,
            page=page_num,
            block=block,
            char_begin=mention["characterStartOffset"],
            char_end=mention["characterEndOffset"],
        )

        # if 'value' in mention['arguments'] and 'variable' in mention['arguments']:
        if mention["labels"][0] == "ParameterSetting":
            # ParameterSetting
            md = TextLiteralValue(
                provenance=provenance_helper.build_embedding(),
                text_extraction=text_extraction,
                value=mention["arguments"]["value"][0]["text"],
                variable_identifier=mention["arguments"]["variable"][0][
                    "text"
                ],
            )
        # elif 'variable' in mention['arguments']:
        elif mention["labels"][0] == "ParamAndUnit":
            # UnitRelation, ParamAndUnit
            # Candidate definition argument names

            md = TextDescription(
                provenance=provenance_helper.build_embedding(),
                text_extraction=text_extraction,
                variable_identifier=mention["arguments"]["variable"][0][
                    "text"
                ],
                variable_definition=mention["arguments"]["description"][0][
                    "text"
                ],
            )
        elif mention["labels"][0] == "UnitRelation":
            # UnitRelation, ParamAndUnit
            # Candidate definition argument names

            md = TextUnits(
                provenance=provenance_helper.build_embedding(),
                text_extraction=text_extraction,
                variable_identifier=mention["arguments"]["variable"][0][
                    "text"
                ],
                unit_type=mention["arguments"]["unit"][0]["text"],
            )
        else:
            md = None

        if md:
            md.score = score
            # Metametadata, the metadata of the metadata
            # Generate the groundings
            groundings = list()
            for arg_name, arg in mention["arguments"].items():
                for attachment in arg[0]["attachments"]:
                    if type(attachment) == list:
                        for g in attachment[0]:
                            grounding = TextGrounding(
                                argument_name=arg_name,
                                id=g["id"],
                                description=g["name"],
                                score=g["score"],
                            )
                            groundings.append(grounding)

            md.grounding = groundings

            # Attach the tr linking metadata to the gromet element
            Utils.attach_metadata(md, element, gromet)

    @staticmethod
    def attach_metadata(new_metadata, element, gromet: GrometFNModule):

        existing_metadata = GrometHelper.get_element_metadata(element, gromet)

        # First, comments in the same line
        # Get the line numbers, if available
        if existing_metadata:
            metadata_array = existing_metadata
        else:
            # If it doesn't exist, add it
            # First create an empty list
            metadata_array = list()
            # Then add it to the gromet fn
            md_collection = gromet.metadata_collection
            md_collection.append(metadata_array)
            # Finally, cross reference it to the gromet element
            md_index = len(md_collection)
            element.metadata = md_index

        # Append the new metadata element to the appropriate array
        metadata_array.append(new_metadata)
