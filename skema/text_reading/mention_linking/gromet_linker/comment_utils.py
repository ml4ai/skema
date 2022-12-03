import datetime
from automates.gromet.metadata import SourceCodeComment, Provenance, TextExtraction, TextDefinition, TextParameter

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


def build_comment_metadata(comment, element, gromet):

	# TODO: Differientiate between line comments and docstrings

	if type(comment) == tuple:		
		line, text = comment
	else:
		line, text = None, comment

	provenance = Provenance(
		"heuristic_1.0",
		str(datetime.datetime.now())
	)

	md = SourceCodeComment(
		provenance=provenance,
		comment=text,
		line_begin= line,
		line_end=line, # TODO Should it be +1?
	)

	attach_metadata(md, element, gromet)

def build_tr_mention_metadata(mention, element, gromet):

	provenance = Provenance(
		"embedding_similarity_1.0",
		str(datetime.datetime.now())
	)

	# TODO
	text_extraction = TextExtraction(

	)

	extraction, score = mention
	extraction = extraction[0] # TODO Fix me, shouldn't be a list

	if 'value' in extraction['arguments'] and 'variable' in extraction['arguments']:
		md = TextParameter(
			provenance = provenance,
			text_extraction= text_extraction,
			value= extraction['arguments']['value'][0]['text'],
			variable_identifier= extraction['arguments']['variable'][0]['text']
		)
	elif 'variable' in extraction['arguments']:
		# unit, description
		# Candidate definition argument names
		candidates = {"unit", "description"}
		definition_name = None
		for c in candidates:
			if c in extraction['arguments']:
				definition_name = c
				break

		md = TextDefinition(
			provenance = provenance,
			text_extraction= text_extraction,
			variable_identifier= extraction['arguments']['variable'][0]['text'],
			variable_definition= extraction['arguments'][definition_name][0]['text']
		)
		
	else:
		md = None

	if md:
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