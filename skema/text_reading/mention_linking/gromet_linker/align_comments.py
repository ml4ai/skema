""" Aligns source code comments to Gromet function networks """

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, List, Mapping, NamedTuple, Optional, Tuple, Union

from automates.program_analysis.JSON2GroMEt.json2gromet import json_to_gromet

from .mention_linking import TextReadingLinker
from .utils import get_code_file_ref, get_element_line_numbers, build_comment_metadata, build_tr_mention_metadata, get_doc_file_ref

import re

	
@dataclass
class SourceComments:
	path:Path
	line_comments:dict[int, str]
	doc_strings:dict[str, str]

	@property
	def file_name(self) -> str:
		return self.path.name

	@property
	def file_path(self) -> str:
		return str(self.path.absolute)

	@staticmethod
	def from_file(comments_path:str) -> "SourceComments":
		""" Reads the automatically extracted comments from the json file """

		def _helper(data):
			line_comments = {l[0]:l[1] for l in data['comments']}
			doc_strings = data['docstrings']

			return SourceComments(
				path = Path(comments_path),
				line_comments= line_comments,
				doc_strings= doc_strings
			)

		with open(comments_path) as f:
			data = json.load(f)

		# Make either a singtle instance or a dictionary of instances in the case of multiple files
		if "comments" in data:
			return _helper(data)
		else:
			return {k:_helper(v) for k, v in data.items()}
			


def get_function_comments(box, fn,  line_comments):
	""" Gets the block of comments adjacent to the function's definition """

	comments = list()

	function_lines = get_element_line_numbers(box, fn)
	if function_lines:
		start = function_lines[0]
		for line_num in range(start-1, 0, -1): # Decreasing line num counter
			if line_num in line_comments:
				comments.append(line_comments[line_num])
			else:
				break

	comments.reverse()
	return comments


def match_variable_name(name, comments):
	ret = list()
	try:
		pattern = re.compile(r"[:,\.\s]" + name + r"[:,\.\s]")
	except Exception as  e:
		# print(e)
		pass
	else:
		for l in comments:
			# TODO this can be done more smartly if we assume a specific programming language (i.e. Python)
			if len(pattern.findall(l)) > 0:
				ret.append(l)

	return ret
				

class DebugInfo(NamedTuple):
	line_range: Optional[Tuple[int]]
	name: Optional[str]
	docstring: List[str]
	comments: List[Union[str, Tuple[int, str]]]
	mentions: Mapping[str, Any]


def print_debug_info(info: DebugInfo) -> None:
	print("===================")
	print(f"{info.line_range} {info.name if info.name else ''}:")
	print()
	print("\n".join(info.docstring))
	print("\n".join(c[1] if type(c) == tuple else c for c in info.comments))
	if len(info.mentions) > 0:
		print("Aligned mentions:\n" + '\n'.join(f"{s}: {m['text']}" for m, s in info.mentions))
	print()



def enhance_attribute_with_comments(attr, attr_type, box, fn, src_comments: SourceComments, linker) -> Optional[DebugInfo]:
	# Identify the variables with the same name to each output port
	  # In case of a tie, resolve the correct variable using the containing line spans

	aligned_comments = list()
	aligned_docstring = list()
	name = attr.name

	# Get docstring, if any
	if box and box.function_type == "FUNCTION":
		docstring = src_comments.doc_strings.get(box.name)
	else:
		docstring = None

	# docstring = doc_strings.get(name)

	# First, comments in the same line
	# Get the line numbers, if available
	line_range = get_element_line_numbers(attr, fn)
	if line_range and attr_type in {"b", "bf"}:
		if attr.function_type not in {"PRIMITIVE", "LITERAL"}:
			start, end = line_range
			for line_num in range(start, end+1):
				if line_num in src_comments.line_comments:
					aligned_comments.append((line_num, src_comments.line_comments[line_num]))

	# Second, get subsection of docstring, if present. If current attr is the containing box itself, then get the complete docstring
	if docstring:
		if attr == box:
			aligned_docstring.extend(docstring)
		else:
			aligned_docstring.extend(match_variable_name(name, docstring))

	# Third, comments not in the same line, but likely of the container function
	if name:
		function_comments = get_function_comments(box, fn, src_comments.line_comments)
		aligned_comments.extend(match_variable_name(name, function_comments))

	# Build new metadata object and append it to the metadata list of each port
	comments = [name] if name else [] + aligned_docstring + [c if type(c) == str else c[1] for c in aligned_comments]
	aligned_mentions = linker.align_to_comments(comments)

	## Build metadata object for each comments aligned
	# Get the comment reference
	code_file_ref = get_code_file_ref(src_comments.file_name, fn)
	# aligned_comments
	for c in aligned_comments:
		build_comment_metadata(c, code_file_ref, attr, fn)
	# aligned docstring
	for d in aligned_docstring:
		build_comment_metadata(d, code_file_ref, attr, fn)
	# linked text reading mentions
	for m in aligned_mentions:
		doc_file_ref = get_doc_file_ref(m, linker, fn)
		build_tr_mention_metadata(m, doc_file_ref, attr, fn)

	if len(aligned_docstring + aligned_comments) > 0:
		return DebugInfo(
			line_range, name,
			aligned_docstring, aligned_comments, aligned_mentions
		)

	else:
		return None

	

	


def align_comments(gromet_path:str, comments_path:str, extractions_path:str, embeddings_path:str, debug: bool = False):
	# Read the function network
	fn = json_to_gromet(gromet_path)
	# Parse the comments
	src_comments = SourceComments.from_file(comments_path)

	# TODO: Add the codefile references
	# Build the TR linking engine
	linker = TextReadingLinker(extractions_path, embeddings_path)

	# Identify the output ports with variable names
	## attributes -> type:"FN" -> value:"b" with name
	##                            value:"opi" with name
	##                            value:"pof" with name
	##                            value:"poc" with name??

	# Iterate over the attributes of the function network
	debug_info = list()
	for attr in fn.attributes:
		t, v = attr.type, attr.value
		if t == "FN":
			container_box = None
			for b in v.b:
				container_box = b
				if b.name:
					info = enhance_attribute_with_comments(b, "b", container_box, fn, src_comments, linker)
					if info:
						debug_info.append(info)
			if v.opi:
				for opi in v.opi:
					if opi.name:
						info = enhance_attribute_with_comments(opi, "opi", container_box, fn, src_comments, linker)
						if info:
							debug_info.append(info)
			if v.pof:
				for pof in v.pof:
					if pof.name:
						info = enhance_attribute_with_comments(pof, "pof", container_box, fn, src_comments, linker)
						if info:
							debug_info.append(info)
			if v.bf:
				for bf in v.bf:
					info = enhance_attribute_with_comments(bf, "bf", container_box, fn, src_comments, linker)
					if info:
						debug_info.append(info)
			# TODO: Is this redundant?
			# for poc in v.poc:
			# 	if poc.name:
			# 		align_comments(b, fn, line_comments, doc_strings)
			
	
	if debug:
		# Aggregate the debug info per name
		grouped_info = defaultdict(list)
		for info in debug_info:
			key = info.name if info.name else ''
			grouped_info[key].append(info)


		# Print them in lexicographical order
		for key in sorted(grouped_info):
			for info in grouped_info[key]:
				print_debug_info(info)
	

	return fn