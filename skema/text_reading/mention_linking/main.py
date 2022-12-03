""" Temporary test script for development This will change to unit tests and example usage script """

from gromet_linker.align_comments import align_comments

from automates.utils.fold import dictionary_to_gromet_json, del_nulls

if __name__ == "__main__":
	gromet_path = "data/gromet/CHIME_SIR--Gromet-FN-auto.json"
	comments_path = "data/comments/CHIME_SIR.json"
	extractions_path = 'data/extractions/CHIME_SIR.json'
	embeddings_path = "/data/covid_comments_models/xdd_covid_19_1x_word2vec/alternate/embeddings.kv"
	enriched_gromet = align_comments(gromet_path, comments_path, extractions_path, embeddings_path)

	# Save gromet file with the new metadata aligned
	with open("test.json", 'w') as f:
		f.write(dictionary_to_gromet_json(del_nulls(enriched_gromet.to_dict())))