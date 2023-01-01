""" Aligns source code comments to Gromet function networks """

from .comment_debugger import CommentDebugger
from .comment_info import CommentInfo
from .gromet_helper import GrometHelper
from .source_comments import SourceComments
from .text_reading_linker import TextReadingLinker
from .time_stamper import TimeStamper
from .uid_stamper import UidStamper
from .utils import Utils
from .variable_name_matcher import VariableNameMatcher

from automates.gromet.fn import GrometBoxFunction, GrometFN, GrometFNModule, GrometPort, TypedValue

class CommentAlignerHelper():

	def __init__(self, debugger: CommentDebugger, time_stamper: TimeStamper, uid_stamper: UidStamper, gromet_fn_module: GrometFNModule, variable_name_matcher: VariableNameMatcher, source_comments: SourceComments, linker: TextReadingLinker):
		self.debugger = debugger
		self.time_stamper = time_stamper
		self.uid_stamper = uid_stamper
		self.gromet_fn_module = gromet_fn_module
		self.variable_name_matcher = variable_name_matcher 
		self.source_comments = source_comments
		self.linker = linker

class CommentAligner():

	def __init__(self, comment_aligner_helper: CommentAlignerHelper):
		self.comment_aligner_helper = comment_aligner_helper
		self.debugger = comment_aligner_helper.debugger
		self.time_stamper = comment_aligner_helper.time_stamper
		self.uid_stamper = comment_aligner_helper.uid_stamper
		self.gromet_fn_module = comment_aligner_helper.gromet_fn_module
		self.variable_name_matcher = comment_aligner_helper.variable_name_matcher
		self.source_comments = comment_aligner_helper.source_comments
		self.linker = comment_aligner_helper.linker

	def get_aligned_comments(self, name: str, inner_line_range: range, outer_line_range: range) -> list[tuple[int, str]]:
		if name:
			inner_comments = self.get_inner_comments(inner_line_range, outer_line_range)
			aligned_comments = self.variable_name_matcher.match_line_comment(name, inner_comments)
			return aligned_comments
		else:
			return []

	# Get the comments in contiguous lines above the function.
	def get_inner_comments(self, inner_line_range: range, outer_line_range: range) -> list[tuple[int, str]]:
		""" Gets the block of comments adjacent to the function's definition """
		line_comments = self.source_comments.line_comments
		comments = list()
		# TODO: This range is likely too wide!  It shouldn't go all the way to the top of the file.
		for line_num in range(inner_line_range.start - 1, -1, -1): # decreasing line_num counter
			if line_num in line_comments: # and line_num in outer_line_numbers
				comments.append((line_num, line_comments[line_num]))
			else:
				break
		comments.reverse()
		return comments

	# TODO pass in comments, all of them, maybe with line numbers?  Name would be an exception, so apart?
	def align_mentions(self, name: str, gromet_object, aligned_docstrings: list[str], box_aligned_comments: list[tuple[int, str]], aligned_comments: list[tuple[int, str]]):
		# Build new metadata object and append it to the metadata list of each port.
		comments = [name] if name else [] + aligned_docstrings + [comment for _, comment in box_aligned_comments] + [comment for _, comment in aligned_comments]
		aligned_mentions = self.linker.align_to_comments(comments)
		code_file_ref = Utils.get_code_file_ref(self.source_comments.file_name(), self.gromet_fn_module)
		# aligned_comments
		for comment in box_aligned_comments:
			Utils.build_comment_metadata(self.time_stamper, comment, code_file_ref, gromet_object, self.gromet_fn_module)
		for _, comment in aligned_comments:
			Utils.build_comment_metadata(self.time_stamper, comment, code_file_ref, gromet_object, self.gromet_fn_module)
		# aligned docstring
		for docstring in aligned_docstrings:
			Utils.build_comment_metadata(self.time_stamper, docstring, code_file_ref, gromet_object, self.gromet_fn_module)
		# linked text reading mentions
		for mention in aligned_mentions:
			doc_file_ref = Utils.get_doc_file_ref(self.time_stamper, self.uid_stamper, mention, self.linker, self.gromet_fn_module)
			Utils.build_textreading_mention_metadata(self.time_stamper, mention, doc_file_ref, gromet_object, self.gromet_fn_module)
		return aligned_mentions

class GrometBoxFunctionCommentAligner(CommentAligner):

	def __init__(self, gromet_box_function: GrometBoxFunction, comment_aligner_helper: CommentAlignerHelper):
		super().__init__(comment_aligner_helper)
		self.gromet_box_function = gromet_box_function
		self.name = self.gromet_box_function.name
		self.line_range = GrometHelper.get_element_line_numbers(self.gromet_box_function, self.gromet_fn_module)

	def get_box_aligned_comments(self, line_range: range) -> list[tuple[int, str]]:
		comments = list()
		if self.gromet_box_function.function_type not in {"PRIMITIVE", "LITERAL"}:
			for line_num in line_range:
				if line_num in self.source_comments.line_comments:
					comments.append((line_num, self.source_comments.line_comments[line_num]))
		return comments

class OuterGrometBoxFunctionCommentAligner(GrometBoxFunctionCommentAligner):

	def __init__(self, gromet_box_function: GrometBoxFunction, comment_aligner_helper: CommentAlignerHelper):
		super().__init__(gromet_box_function, comment_aligner_helper)
		self.docstrings = self.calc_docstrings()

	def calc_docstrings(self) -> list[str]:
		# docstrings are within the function, just after the signature and before the body.
		condition = self.gromet_box_function.function_type == "FUNCTION"
		docstrings = self.source_comments.doc_strings.get(self.gromet_box_function.name, []) if condition else []
		return docstrings

	def align(self) -> None:
		if self.name:
			aligned_docstrings = self.docstrings
			box_aligned_comments = self.get_box_aligned_comments(self.line_range)
			aligned_comments = self.get_aligned_comments(self.name, self.line_range, self.line_range)
			aligned_mentions = self.align_mentions(self.name, self.gromet_box_function, aligned_docstrings, box_aligned_comments, aligned_comments)
			info = CommentInfo(self.line_range, self.name, aligned_docstrings, box_aligned_comments + aligned_comments, aligned_mentions)
			self.debugger.add_info(info)

class InnerGrometBoxFunctionCommentAligner(GrometBoxFunctionCommentAligner):

	def __init__(self, gromet_box_function: GrometBoxFunction, comment_aligner_helper: CommentAlignerHelper, docstrings: list[str], outer_line_range: range):
		super().__init__(gromet_box_function, comment_aligner_helper)
		self.docstrings = docstrings
		self.outer_line_range = outer_line_range

	def align(self) -> None:
		aligned_docstrings = self.variable_name_matcher.match_comment(self.name, self.docstrings)
		box_aligned_comments = self.get_box_aligned_comments(self.line_range)
		aligned_comments = self.get_aligned_comments(self.name, self.outer_line_range, self.outer_line_range)
		aligned_mentions = self.align_mentions(self.name, self.gromet_box_function, aligned_docstrings, box_aligned_comments, aligned_comments)
		comment_info = CommentInfo(self.line_range, self.name, aligned_docstrings, box_aligned_comments + aligned_comments, aligned_mentions)
		self.debugger.add_info(comment_info)

class GrometPortCommentAligner(CommentAligner):

	def __init__(self, gromet_port: GrometPort, comment_aligner_helper: CommentAlignerHelper, docstrings: list[str], outer_line_range: range):
		super().__init__(comment_aligner_helper)
		self.gromet_port = gromet_port
		self.docstrings = docstrings
		self.name = self.gromet_port.name
		self.outer_line_range = outer_line_range
		self.line_range = GrometHelper.get_element_line_numbers(self.gromet_port, self.gromet_fn_module)

	def align(self) -> None:
		aligned_docstrings = self.variable_name_matcher.match_comment(self.name, self.docstrings)
		aligned_comments = self.get_aligned_comments(self.name, self.outer_line_range, self.outer_line_range)
		aligned_mentions = self.align_mentions(self.name, self.gromet_port, aligned_docstrings, [], aligned_comments)
		comment_info = CommentInfo(self.line_range, self.name, aligned_docstrings, aligned_comments, aligned_mentions)
		self.debugger.add_info(comment_info)

class GrometFNCommentAligner(CommentAligner):

	def __init__(self, gromet_fn: GrometFN, comment_aligner_helper: CommentAlignerHelper):
		super().__init__(comment_aligner_helper)
		self.gromet_fn = gromet_fn
		# b: The FN Outer Box (although not enforced, there is always only 1).
		# The gromet_fn value comes from the superclass that gets it from the helper.
		assert self.gromet_fn.b and len(self.gromet_fn.b) == 1
		self.outer_gromet_box_function = self.gromet_fn.b[0]

	def align(self) -> None:
		# Identify the output ports with variable names
		## attributes -> type:"FN" -> value:"b" with name
		##                            value:"opi" with name
		##                            value:"pof" with [unnecessary] name
		##                            value:"poc" with name??
		outer_gromet_box_function_comment_aligner = OuterGrometBoxFunctionCommentAligner(self.outer_gromet_box_function, self.comment_aligner_helper)
		outer_gromet_box_function_comment_aligner.align()
		docstrings = outer_gromet_box_function_comment_aligner.docstrings
		outer_line_range = outer_gromet_box_function_comment_aligner.line_range

		# opi: The Outer Port Inputs of the FN Outer Box (b)
		if self.gromet_fn.opi:
			for gromet_port in self.gromet_fn.opi:
				GrometPortCommentAligner(gromet_port, self.comment_aligner_helper, docstrings, outer_line_range).align()
		# pof: The Port Outputs of the GrometBoxFunctions (bf).
		if self.gromet_fn.pof:
			for gromet_port in self.gromet_fn.pof:
				GrometPortCommentAligner(gromet_port, self.comment_aligner_helper, docstrings, outer_line_range).align()
		# bf: The GrometBoxFunctions within this GrometFN.
		if self.gromet_fn.bf:
			for gromet_box_function in self.gromet_fn.bf:
				InnerGrometBoxFunctionCommentAligner(gromet_box_function, self.comment_aligner_helper, docstrings, outer_line_range).align()
		# poc: The Port Outputs of the GrometBoxConditionals (bc)
		# TODO: Is this redundant?
		# for poc in v.poc:
		# 	GrometPortCommentAligner(gromet_port, self.comment_aligner_helper, docstrings, outer_line_range).align()

class GrometAttributeCommentAligner(CommentAligner):

	def __init__(self, gromet_attribute: TypedValue, comment_aligner_helper: CommentAlignerHelper):
		super().__init__(comment_aligner_helper)
		self.gromet_attribute = gromet_attribute

	def align(self) -> None:
		if self.gromet_attribute.type == "FN":
			GrometFNCommentAligner(self.gromet_attribute.value, self.comment_aligner_helper).align()

class GrometFNModuleCommentAligner(CommentAligner):

	def __init__(self, gromet_fn_module: GrometFNModule, comment_aligner_helper: CommentAlignerHelper, embeddings_path: str):
		# TODO: The incoming gromet_fn_module should not yet have been aligned.  Assert that!
		super().__init__(comment_aligner_helper)
		self.gromet_fn_module = gromet_fn_module
		# TODO: Add the codefile references.

	def align(self) -> None:
		for gromet_attribute in self.gromet_fn_module.attributes:
			GrometAttributeCommentAligner(gromet_attribute, self.comment_aligner_helper).align()
		self.debugger.debug()
