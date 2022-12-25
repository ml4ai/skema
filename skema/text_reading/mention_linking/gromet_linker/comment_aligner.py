""" Aligns source code comments to Gromet function networks """

from automates.program_analysis.JSON2GroMEt.json2gromet import json_to_gromet
from collections import defaultdict
from typing import Optional

from .debug_info import DebugInfo
from .gromet_helper import GrometHelper
from .source_comments import SourceComments
from .text_reading_linker import TextReadingLinker
from .utils import get_code_file_ref, build_comment_metadata, build_tr_mention_metadata, get_doc_file_ref
from .variable_name_matcher import VariableNameMatcher

class CommentAligner():

	class Debugger():
		def add_info(self, info):
			pass

		def debug():
			pass
	class FakeDebugger(Debugger):
		pass

	class RealDebugger(Debugger):
		def __init__(self):
			self.debug_info = list()

		def add_info(self, info):
			if info:
				self.debug_info.append(info)

		def debug(self):
			# Aggregate the debug info per name
			grouped_info = defaultdict(list)
			for info in self.debug_info:
				key = info.name if info.name else ''
				grouped_info[key].append(info)

			# Print them in lexicographical order
			for key in sorted(grouped_info):
				for info in grouped_info[key]:
					info.print()


	def __init__(self, time_stamper, uid_stamper, gromet_path: str, comments_path: str, extractions_path: str, embeddings_path: str):
		self.time_stamper = time_stamper
		self.uid_stamper = uid_stamper
		# Read the function network
		self.gromet_path = gromet_path
		# Parse the comments
		self.src_comments = SourceComments.from_file(comments_path)
		# TODO: Add the codefile references
		# Build the TR linking engine
		self.linker = TextReadingLinker(extractions_path, embeddings_path)
		self.variable_name_matcher = VariableNameMatcher("python")

	# Return new fn that has been aligned and linked
	def align(self, debug: bool = False):
		debugger = self.RealDebugger() if debug else self.FakeDebugger()
		gromet_fn_module = json_to_gromet(self.gromet_path)

		container_gromet_box_function = None
		gromet_fn = None

		def get_info(gromet_object, key):
			info = self.enhance_attribute_with_comments(gromet_object, key, container_gromet_box_function, gromet_fn_module)
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
					for gromet_box_function in gromet_fn.b:
						container_gromet_box_function = gromet_box_function
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

	def get_function_comments(self, box, fn, line_comments):
		""" Gets the block of comments adjacent to the function's definition """

		comments = list()

		function_lines = GrometHelper.get_element_line_numbers(box, fn)
		if function_lines:
			start = function_lines[0]
			for line_num in range(start-1, 0, -1): # Decreasing line num counter
				if line_num in line_comments:
					comments.append(line_comments[line_num])
				else:
					break

		comments.reverse()
		return comments

	def enhance_attribute_with_comments(self, gromet_object, key, container_box, gromet_fn_module) -> Optional[DebugInfo]:
		# Identify the variables with the same name to each output port
		# In case of a tie, resolve the correct variable using the containing line spans

		aligned_comments = list()
		aligned_docstring = list()
		name = gromet_object.name

		# Get docstring, if any
		if container_box and container_box.function_type == "FUNCTION":
			docstring = self.src_comments.doc_strings.get(container_box.name)
		else:
			docstring = None

		# First, comments in the same line
		# Get the line numbers, if available
		line_range = GrometHelper.get_element_line_numbers(gromet_object, gromet_fn_module)
		if line_range and key in {"b", "bf"}:
			if gromet_object.function_type not in {"PRIMITIVE", "LITERAL"}:
				start, end = line_range
				for line_num in range(start, end+1):
					if line_num in self.src_comments.line_comments:
						aligned_comments.append((line_num, self.src_comments.line_comments[line_num]))

		# Second, get subsection of docstring, if present. If current attr is the containing box itself, then get the complete docstring
		if docstring:
			if gromet_object == container_box:
				aligned_docstring.extend(docstring)
			else:
				aligned_docstring.extend(self.variable_name_matcher.match(name, docstring))

		# Third, comments not in the same line, but likely of the container function
		if name:
			function_comments = self.get_function_comments(container_box, gromet_fn_module, self.src_comments.line_comments)
			aligned_comments.extend(self.variable_name_matcher.match(name, function_comments))

		# Build new metadata object and append it to the metadata list of each port
		comments = [name] if name else [] + aligned_docstring + [c if type(c) == str else c[1] for c in aligned_comments]
		aligned_mentions = self.linker.align_to_comments(comments)

		## Build metadata object for each comments aligned
		# Get the comment reference
		code_file_ref = get_code_file_ref(self.src_comments.file_name, gromet_fn_module)
		# aligned_comments
		for c in aligned_comments:
			build_comment_metadata(self.time_stamper, c, code_file_ref, gromet_object, gromet_fn_module)
		# aligned docstring
		for d in aligned_docstring:
			build_comment_metadata(self.time_stamper, d, code_file_ref, gromet_object, gromet_fn_module)
		# linked text reading mentions
		for m in aligned_mentions:
			doc_file_ref = get_doc_file_ref(self.time_stamper, self.uid_stamper, m, self.linker, gromet_fn_module)
			build_tr_mention_metadata(self.time_stamper, m, doc_file_ref, gromet_object, gromet_fn_module)

		if len(aligned_docstring + aligned_comments) > 0:
			return DebugInfo(
				line_range, name,
				aligned_docstring, aligned_comments, aligned_mentions
			)
		else:
			return None
