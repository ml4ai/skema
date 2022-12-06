import datetime
from typing import Optional, Tuple
from automates.gromet.metadata import SourceCodeComment, Provenance, TextGrounding, TextExtraction, TextDescription, TextLiteralValue, TextualDocumentCollection, TextualDocumentReference, TextUnits, TextExtractionMetadata

from gromet_linker.mention_linking import TextReadingLinker

import itertools as it

def build_provenance(method:str) -> Provenance:
	return Provenance(
		method,
		str(datetime.datetime.now())
	)

def get_element_metadata(elem, fn):
	# Get the metadata, if exists
	if elem.metadata:
		metadata = fn.metadata_collection[elem.metadata - 1] # Minus 1 because it is 1-based indexing
	else:
		metadata = None

	return metadata

def get_element_line_numbers(elem, fn):
	metadata = get_element_metadata(elem, fn)

	# First, comments in the same line
	# Get the line numbers, if available
	if metadata:
		for m in metadata:
			if m.metadata_type == "source_code_reference":
				return m.line_begin, m.line_end
		
	return None

def get_code_file_ref(comments_file_name: str, gromet) -> Optional[str]:
	""" Fetches the UUID of the code_file_reference that matches the one from the comments file """

	mdc = get_element_metadata(gromet, gromet)
	code_collection = None
	uid = None

	for md in mdc:
		if md.metadata_type == "source_code_collection":
			code_collection = md
			break

	if code_collection:
		prefix = ".".join(comments_file_name.split(".")[:-1])
		
		for file in code_collection.files:
			if file.name.startswith(prefix):
				uid = file.uid
				break

	return uid

def get_doc_file_ref(scored_mention, linker:TextReadingLinker, gromet) -> Optional[str]:
	""" Fetches the UUID of the text doc reference that matches the one from the mention's file """

	mention, _ = scored_mention
	mdc = get_element_metadata(gromet, gromet)
	text_collection = None
	uid = None

	for md in mdc:
		if md.metadata_type == "textual_document_collection":
			text_collection = md
			break

	# If the text doc collection doesn't exist, then add it
	if not text_collection:
		text_collection = TextualDocumentCollection(
			provenance = build_provenance("embedding_similarity_1.0"),
			documents= list()
		)

		mdc.append(text_collection)

	if text_collection:
		mention_doc = linker.documents[mention['document']]
		doc_id = mention_doc['id']
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
				uid = str(hash(doc_id)),
				global_reference_id= "TBD",
				cosmos_id= doc_id,
				cosmos_version_number= 0.1,
				skema_id= 0.1
			)

			existing_docs_refs.append(doc_ref)

		uid = doc_ref.uid


	return uid

def build_comment_metadata(comment:Tuple[int, str] | str, code_file_ref:str , element, gromet):

	# TODO: Differientiate between line comments and docstrings

	if type(comment) == tuple:		
		line, text = comment
	else:
		line, text = None, comment

	md = SourceCodeComment(
		provenance= build_provenance("heuristic_1.0"),
		code_file_reference_uid= code_file_ref,
		comment=text,
		line_begin= line,
		line_end=line, # TODO Should it be +1?
	)

	attach_metadata(md, element, gromet)

def build_tr_mention_metadata(scored_mention, doc_file_ref: str, element, gromet):

	mention, score = scored_mention

	page_num, block = None, None
	for attachment in mention['attachments']:
		if 'pageNum' in attachment:
			page_num = attachment['pageNum'][0]
		if 'blockIdx' in attachment:
			block = attachment['blockIdx'][0]

	text_extraction = TextExtraction(
		document_reference_uid= doc_file_ref,
		page= page_num,
		block= block,
		char_begin= mention['characterStartOffset'],
		char_end= mention['characterEndOffset']
	)

	

	# if 'value' in mention['arguments'] and 'variable' in mention['arguments']:
	if mention['labels'][0] == "ParameterSetting":
		# ParameterSetting
		md = TextLiteralValue(
			provenance = build_provenance("embedding_similarity_1.0"),
			text_extraction= text_extraction,
			value= mention['arguments']['value'][0]['text'],
			variable_identifier= mention['arguments']['variable'][0]['text']
		)
	# elif 'variable' in mention['arguments']:
	elif mention['labels'][0] == "ParamAndUnit":
		# UnitRelation, ParamAndUnit
		# Candidate definition argument names

		md = TextDescription(
			provenance = build_provenance("embedding_similarity_1.0"),
			text_extraction= text_extraction,
			variable_identifier= mention['arguments']['variable'][0]['text'],
			variable_definition= mention['arguments']['description'][0]['text']
		)
	elif mention['labels'][0] == "UnitRelation":
		# UnitRelation, ParamAndUnit
		# Candidate definition argument names

		md = TextUnits(
			provenance = build_provenance("embedding_similarity_1.0"),
			text_extraction= text_extraction,
			variable_identifier= mention['arguments']['variable'][0]['text'],
			unit_type= mention['arguments']["unit"][0]['text']
		)
	else:
		md = None

	if md:
		md.score = score
		# Metametadata, the metadata of the metadata
		# Generate the groundings
		groundings = list()
		for arg_name, arg in mention['arguments'].items():
			for attachment in arg[0]['attachments']:
				if type(attachment) == list:
					for g in attachment[0]:
						grounding = TextGrounding(
							argument_name= arg_name,
							id = g['id'],
							description= g['name'],
							score= g['score']
						)
						groundings.append(grounding)

		md.grounding = groundings

		# Attach the tr linking metadata to the gromet element
		attach_metadata(md, element, gromet)


def attach_metadata(new_metadata, element, gromet):

	existing_metadata = get_element_metadata(element, gromet)

	# First, comments in the same line
	# Get the line numbers, if available
	if existing_metadata:
		assert(type(existing_metadata) == list, "Existing metadata type is not a list")
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