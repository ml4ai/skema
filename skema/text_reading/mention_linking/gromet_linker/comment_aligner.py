""" Aligns source code comments to Gromet function networks """

from automates.program_analysis.JSON2GroMEt.json2gromet import json_to_gromet
from collections import defaultdict
from typing import Optional

from .comment_debugger import CommentDebugger
from .debug_info import DebugInfo
from .gromet_helper import GrometHelper
from .source_comments import SourceComments
from .text_reading_linker import TextReadingLinker
from .utils import get_code_file_ref, build_comment_metadata, build_tr_mention_metadata, get_doc_file_ref
from .variable_name_matcher import VariableNameMatcher

class CommentAligner():

	def __init__(self, time_stamper, uid_stamper, gromet_path: str, comments_path: str, extractions_path: str, embeddings_path: str):
		self.time_stamper = time_stamper
		self.uid_stamper = uid_stamper
		self.gromet_path = gromet_path
		self.src_comments = SourceComments.from_file(comments_path)
		# TODO: Add the codefile references.
		self.linker = TextReadingLinker(extractions_path, embeddings_path)
		self.variable_name_matcher = VariableNameMatcher("python")

	# Return new gromet_fn_module that has been aligned and linked.
	def align(self, debug: bool = False):
		debugger = CommentDebugger.create(debug)
		# This is read anew here, because it will be modified before return.
		gromet_fn_module = json_to_gromet(self.gromet_path)

		container_gromet_box_function = None

		def get_info(gromet_object, key):
			condition = container_gromet_box_function and container_gromet_box_function.function_type == "FUNCTION"
			docstrings = self.src_comments.doc_strings.get(container_gromet_box_function.name, []) if condition else []

			condition = gromet_object != container_gromet_box_function
			aligned_docstrings = self.variable_name_matcher.match(gromet_object.name, docstrings) if condition else docstrings

			line_range = GrometHelper.get_element_line_numbers(gromet_object, gromet_fn_module)

			# Get both kinds?
			aligned_comments = self.enhance_attribute_with_comments(gromet_object, key, container_gromet_box_function, gromet_fn_module, line_range)
			info = self.step2(gromet_object, gromet_fn_module, line_range, aligned_docstrings, aligned_comments)
			debugger.add_info(info)

		# Identify the output ports with variable names
		## attributes -> type:"FN" -> value:"b" with name
		##                            value:"opi" with name
		##                            value:"pof" with [unnecessary] name
		##                            value:"poc" with name??

		for attribute in gromet_fn_module.attributes:
			type, value = attribute.type, attribute.value
			if type == "FN":
				gromet_fn = value # It is really just a dict, not a gromet_fn.
				
				if gromet_fn.b:
					# This is sticky.  Attributes are in order and the container function appears before anything contained.
					assert len(gromet_fn.b) == 1
					container_gromet_box_function = gromet_fn.b[0]
					for gromet_box_function in gromet_fn.b:
						if gromet_box_function.name:
							get_info(gromet_box_function, "b")
				
				if gromet_fn.opi:
					for gromet_port in gromet_fn.opi:
						if gromet_port.name:
							get_info(gromet_port, "opi")
				if gromet_fn.pof:
					for gromet_port in gromet_fn.pof:
						if gromet_port.name:
							get_info(gromet_port, "pof")
				if gromet_fn.bf:
					for gromet_box_function in gromet_fn.bf:
						if True: # A name is not required here.
							get_info(gromet_box_function, "bf")
				# TODO: Is this redundant?
				# for poc in v.poc:
				# 	if poc.name:
				# 		align_comments(b, fn, line_comments, doc_strings)
		
		debugger.debug()
		return gromet_fn_module

	# Get the comments in contiguous lines above the function.
	def get_function_comments(self, gromet_box_function, gromet_fn_module, line_comments):
		""" Gets the block of comments adjacent to the function's definition """
		comments = list()
		line_numbers = GrometHelper.get_element_line_numbers(gromet_box_function, gromet_fn_module)
		for line_num in range(line_numbers.start - 1, -1, -1): # decreasing line_num counter
			if line_num in line_comments:
				comments.append(line_comments[line_num])
			else:
				break
		comments.reverse()
		return comments

	# This gromet_object should be either a gromet_box_function or gromet_port.
	def enhance_attribute_with_comments(self, gromet_object, key, container_gromet_box_function, gromet_fn_module, line_range: range):
		# Identify the variables with the same name to each output port.
		# In case of a tie, resolve the correct variable using the containing line spans.

		aligned_comments = list()

		# So b and bf are special for the boxes.
		if key in {"b", "bf"} and gromet_object.function_type not in {"PRIMITIVE", "LITERAL"}:
			for line_num in line_range:
				if line_num in self.src_comments.line_comments:
					aligned_comments.append((line_num, self.src_comments.line_comments[line_num]))

		# Use comments not in the same line, but likely of the container function.
		name = gromet_object.name
		if name:
			function_comments = self.get_function_comments(container_gromet_box_function, gromet_fn_module, self.src_comments.line_comments)
			aligned_comments.extend(self.variable_name_matcher.match(name, function_comments))

		return aligned_comments

	def step2(self, gromet_object, gromet_fn_module, line_range, aligned_docstrings, aligned_comments):
		name = gromet_object.name

		# Build new metadata object and append it to the metadata list of each port.
		comments = [name] if name else [] + aligned_docstrings + [c if type(c) == str else c[1] for c in aligned_comments]
		aligned_mentions = self.linker.align_to_comments(comments)

		## Build metadata object for each comments aligned
		# Get the comment reference
		code_file_ref = get_code_file_ref(self.src_comments.file_name, gromet_fn_module)
		# aligned_comments
		for c in aligned_comments:
			build_comment_metadata(self.time_stamper, c, code_file_ref, gromet_object, gromet_fn_module)
		# aligned docstring
		for d in aligned_docstrings:
			build_comment_metadata(self.time_stamper, d, code_file_ref, gromet_object, gromet_fn_module)
		# linked text reading mentions
		for m in 	aligned_mentions:
			doc_file_ref = get_doc_file_ref(self.time_stamper, self.uid_stamper, m, self.linker, gromet_fn_module)
			build_tr_mention_metadata(self.time_stamper, m, doc_file_ref, gromet_object, gromet_fn_module)

		if aligned_docstrings or aligned_comments:
			return DebugInfo(line_range, name, aligned_docstrings, aligned_comments, aligned_mentions)
		return None
