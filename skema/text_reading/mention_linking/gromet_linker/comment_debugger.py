from .comment_info import CommentInfo

from collections import defaultdict

class CommentDebugger():
	def add_info(self, comment_info: CommentInfo) -> None:
		pass

	def debug() -> None:
		pass

	@staticmethod
	def create(debug: bool):
		if (debug):
			return RealCommentDebugger()
		else:
			return FakeCommentDebugger()

class FakeCommentDebugger(CommentDebugger):
	pass

class RealCommentDebugger(CommentDebugger):
	def __init__(self):
		self.debug_info = list()

	def add_info(self, comment_info: CommentInfo) -> None:
		if comment_info.line_docstrings or comment_info.comments:
			self.debug_info.append(comment_info)

	def debug(self) -> None:
		# Aggregate the debug info per name.
		grouped_info = defaultdict(list)
		for info in self.debug_info:
			key = info.name if info.name else ''
			grouped_info[key].append(info)

		# Print them in lexicographical order.
		for key in sorted(grouped_info):
			for info in grouped_info[key]:
				info.print()
