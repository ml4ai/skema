from skema.gromet.metadata import Provenance

class ProvenanceHelper():

	def __init__(self, time_stamper):
		self.time_stamper = time_stamper

	def build(self, method: str) -> Provenance:
		return Provenance(
			method,
			self.time_stamper.stamp()
		)

	def build_comment(self, method: str = "heuristic_1.0") -> Provenance:
		return self.build(method)

	def build_embedding(self, method: str = "embedding_similarity_1.0") -> Provenance:
		return self.build(method)
