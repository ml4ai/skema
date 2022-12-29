from .debug_info import DebugInfo

from collections import defaultdict

class CommentDebugger():
	def add_info(self, info: DebugInfo) -> None:
		pass

	def debug() -> None:
		pass

	@staticmethod
	def create(debug: bool):
		if (debug):
			return RealDebugger()
		else:
			return FakeDebugger()

class FakeDebugger(CommentDebugger):
	pass

class RealDebugger(CommentDebugger):
	def __init__(self):
		self.debug_info = list()

	def add_info(self, info: DebugInfo) -> None:
		if info:
			self.debug_info.append(info)

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
