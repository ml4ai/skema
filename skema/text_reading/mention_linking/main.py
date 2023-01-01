import os

""" Temporary test script for development This will change to unit tests and example usage script """

from automates.utils.fold import dictionary_to_gromet_json, del_nulls
from skema.text_reading.mention_linking.gromet_linker.comment_aligner import CommentAlignerHelper, GrometFNModuleCommentAligner
from skema.text_reading.mention_linking.gromet_linker.comment_debugger import CommentDebugger
from skema.text_reading.mention_linking.gromet_linker.gromet_helper import GrometHelper
from skema.text_reading.mention_linking.gromet_linker.source_comments import SourceComments
from skema.text_reading.mention_linking.gromet_linker.text_reading_linker import TextReadingLinker
from skema.text_reading.mention_linking.gromet_linker.time_stamper import DebugTimeStamper
from skema.text_reading.mention_linking.gromet_linker.uid_stamper import DocIdStamper
from skema.text_reading.mention_linking.gromet_linker.variable_name_matcher import VariableNameMatcher

if __name__ == "__main__":
	gromet_path = "../../../../mention_linking_files/gromet/CHIME_SIR--Gromet-FN-auto.json"
	comments_path = "../../../../mention_linking_files/comments/CHIME_SIR.json"
	extractions_path = "../../../../mention_linking_files/extractions/CHIME_SIR.json"

	# gromet_path = "data/gromet/CHIME_SIR--Gromet-FN-auto.json"
	# comments_path = "data/comments/CHIME_SIR.json"
	# extractions_path = 'data/extractions/CHIME_SIR.json'

	# gromet_path = "data/gromet/chime_penn--Gromet-FN-auto.json"
	# comments_path = "data/comments/CHIME_full_penn.json"
	# extractions_path = 'data/extractions/CHIME_SIR.json'

	# gromet_path = "data/gromet/bucky_simplified_v1--Gromet-FN-auto.json"
	# comments_path = "data/comments/BUCKY.json"
	# extractions_path = 'data/extractions/BUCKY.json'

	embeddings_path = "../../../../word_embeddings/epi+code_comments/embeddings.kv"
	# embeddings_path = "/data/covid_comments_models/xdd_covid_19_1x_word2vec/alternate/embeddings.kv"

	for path in [gromet_path, comments_path, extractions_path, embeddings_path, embeddings_path + ".vectors.npy"]:
		if not os.path.exists(path):
			raise Exception(f"A file doesn't seem to exist: {path}")

	debug = True
	gromet_fn_module = GrometHelper.json_to_gromet(gromet_path)
	debugger = CommentDebugger.create(debug)
	time_stamper = DebugTimeStamper()
	uid_stamper = DocIdStamper()
	variable_name_matcher = VariableNameMatcher("python")
	source_comments = SourceComments.from_file(comments_path)
	linker = TextReadingLinker(extractions_path, embeddings_path)

	comment_aligner_helper = CommentAlignerHelper(debugger, time_stamper, uid_stamper, gromet_fn_module, variable_name_matcher, source_comments, linker)
	comment_aligner = GrometFNModuleCommentAligner(gromet_fn_module, comment_aligner_helper, embeddings_path)
	comment_aligner.align()

	# Save gromet file with the new metadata aligned
	with open("test.json", 'w') as file:
		file.write(dictionary_to_gromet_json(del_nulls(gromet_fn_module.to_dict())))
