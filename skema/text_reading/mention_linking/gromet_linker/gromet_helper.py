# This is for basic gromet functions that require two or more Gromet objects.

class GrometHelper():

	@staticmethod
	def get_element_metadata(gromet_object, gromet_fn_module):
		if gromet_object.metadata:
			return gromet_fn_module.metadata_collection[gromet_object.metadata - 1] # minus 1 because it is 1-based indexing
		return None

	@staticmethod
	def get_element_line_numbers(gromet_object, gromet_fn_module) -> range:
		metadata = GrometHelper.get_element_metadata(gromet_object, gromet_fn_module)

		if metadata:
			for metadatum in metadata:
				if metadatum.metadata_type == "source_code_reference":
					return range(metadatum.line_begin, metadatum.line_end + 1) # [beg, end)
		return range(0, 0) # empty range
