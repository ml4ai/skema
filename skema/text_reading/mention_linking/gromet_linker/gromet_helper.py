# This is for basic gromet functions that require two or more Gromet objects.

class GrometHelper():

	@staticmethod
	def get_element_metadata(element, gromet_fn_module):
		if element.metadata:
			return gromet_fn_module.metadata_collection[element.metadata - 1] # minus 1 because it is 1-based indexing
		return None

	@staticmethod
	def get_element_line_numbers(element, gromet_fn_module):
		metadata = GrometHelper.get_element_metadata(element, gromet_fn_module)

		if metadata:
			for m in metadata:
				if m.metadata_type == "source_code_reference":
					return range(m.line_begin, m.line_end + 1) # [beg, end)
		return range(0, 0) # empty range
