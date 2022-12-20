from automates.gromet.metadata import Provenance

class ProvenanceHelper():

	def __init__(self, time_stamper):
		self.time_stamper = time_stamper

	def build(self, method: str) -> Provenance:
		return Provenance(
			method,
			self.time_stamper.stamp()
		)
