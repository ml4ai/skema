

class UidStamper():
	def stamp(self, doc_id: str) -> str:
		pass

class DocIdStamper(UidStamper):
	def stamp(self, doc_id: str) -> str:
		return doc_id


class HashStamper(UidStamper):
	def stamp(self, doc_id: str) -> str:
		return str(hash(doc_id))
