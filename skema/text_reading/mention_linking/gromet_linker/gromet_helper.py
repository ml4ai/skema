# This is for basic gromet functions that requite two or more Gromet objects.

class GrometHelper():

	@staticmethod
	def get_element_metadata(elem, gromet_fn_module):
		# Get the metadata, if exists
		if elem.metadata:
			return gromet_fn_module.metadata_collection[elem.metadata - 1] # Minus 1 because it is 1-based indexing
		return None

	@staticmethod
	def get_element_line_numbers(elem, gromet_fn_module):
		metadata = GrometHelper.get_element_metadata(elem, gromet_fn_module)

		# First, comments in the same line
		# Get the line numbers, if available
		if metadata:
			for m in metadata:
				if m.metadata_type == "source_code_reference":
					return m.line_begin, m.line_end
		return None
