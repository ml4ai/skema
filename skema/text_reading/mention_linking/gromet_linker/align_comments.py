""" Aligns source code comments to Gromet function networks """

from dataclasses import dataclass
import json
from pathlib import Path
from typing import NamedTuple, Optional

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

		with open(comments_path) as f:
			all_comments = json.load(f)

		line_comments = {l[0]:l[1] for l in all_comments['comments']}
		doc_strings = all_comments['docstrings']

		return SourceComments(
			path = Path(comments_path),
			line_comments= line_comments,
			doc_strings= doc_strings
		)


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
				


def enhance_attribute_with_comments(attr, attr_type, box, fn, src_comments: SourceComments, linker):
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
		print("===================")
		print(f"{line_range} {name if name else ''}:")
		print()
		print("\n".join(aligned_docstring))
		print("\n".join(c[1] if type(c) == tuple else c for c in aligned_comments))
		if len(aligned_mentions) > 0:
			print("Aligned mentions:" + '\n'.join(f"{s}: {m['text']}" for m, s in aligned_mentions))
		print()

	


def align_comments(gromet_path:str, comments_path:str, extractions_path:str, embeddings_path:str):
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
	for attr in fn.attributes:
		t, v = attr.type, attr.value
		if t == "FN":
			container_box = None
			for b in v.b:
				container_box = b
				if b.name:
					enhance_attribute_with_comments(b, "b", container_box, fn, src_comments, linker)
			if v.opi:
				for opi in v.opi:
					if opi.name:
						enhance_attribute_with_comments(opi, "opi", container_box, fn, src_comments, linker)
			if v.pof:
				for pof in v.pof:
					if pof.name:
						enhance_attribute_with_comments(pof, "pof", container_box, fn, src_comments, linker)
			if v.bf:
				for bf in v.bf:
					enhance_attribute_with_comments(bf, "bf", container_box, fn, src_comments, linker)
			# TODO: Is this redundant?
			# for poc in v.poc:
			# 	if poc.name:
			# 		align_comments(b, fn, line_comments, doc_strings)

	
	

	return fn