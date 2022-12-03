import itertools
import json
from typing import Any, Union
from collections import defaultdict
from gensim.models import KeyedVectors
import numpy as np
import itertools as it

class TextReadingLinker:
	""" Encapsulates the logic of the linker to text reading mentions """

	def __init__(self, mentions_path:str, embeddings_path:str):

		
		# Load the embeddings
		self._model = KeyedVectors.load(embeddings_path)

		# Read the mentions
		raw_mentions, documents = self._read_text_mentions(mentions_path)
		self._docs = documents
		self._mentions = defaultdict(list)

		linkable_mentions = [m for m in raw_mentions if any('variable' in a for a in m['arguments'])]
		self._linkable_variables = defaultdict(list)
		self._linkable_descriptions = dict()
		
		for m in linkable_mentions:
			var = m['arguments']['variable'][0]['text']
			if len(self._preprocess(var)) > 0:
				self._linkable_variables[var].append(m)

			# This is a level of indirection in which we also look at the variable descriptions for linking
			if 'description' in m['arguments']:
				desc = m['arguments']['description'][0]['text']
				if len(self._preprocess(desc)) > 0:
					self._linkable_descriptions[desc] = var # We resolve to the variable name, which will use later to do the "graph" linking


		# Preprocess the vectors for each mention text
		keys, vectors = list(), list()
		for k in it.chain(self._linkable_variables, self._linkable_descriptions):
			keys.append(k)
			vectors.append(self._average_vector(self._preprocess(k)))

		vectors = np.stack(vectors, axis=0)

		self._keys = keys
		self._vectors = vectors



	def _preprocess(self, text:str | list[str]) -> list[str]:
		""" Prepares the text for before fetching embeddings """
		if type(text) == str:
			text = [text]

		return [word 
			for word 
			in itertools.chain.from_iterable(sent.split() for sent in text)
			if word in self._model
		]


	def _read_text_mentions(self, path:str) -> dict[str, Any]:
		with open(path) as f:
			data = json.load(f)

		# TODO Filter out irrelevant extractions
		relevant_labels = {
			# For ports
			"Parameter",
			"ParamAndUnit",
			"GreekLetter",
			"Model",
			"model",
			"ParameterSetting",
			"ModelComponent",
			# For box functions
			"Function",
			"ParameterSetting",
			"UnitRelation",
		}

		relevant_mentions = [m for m in data['mentions'] if m['type'] != 'TextBoundMention' and len(set(m['labels']) & relevant_labels) > 0]

		# Add context to the mentions
		docs = data['documents']
		for m in relevant_mentions:
			doc = docs[m['document']]
			sent = doc['sentences'][m['sentence']]
			# TODO perhaps extend this to a window of text
			context = ' '.join(sent['raw'])
			m['context'] = context

		return relevant_mentions, docs

	def _average_vector(self, words:list[str]):
		""" Precomputes and l2 normalizes the average vector of the requested word embeddings """
		
		vectors = self._model[words]
		avg = vectors.mean(axis=0)
		norm = np.linalg.norm(avg)
		normalized_avg = avg / norm
		return normalized_avg

	def align_to_comments(self, comments, k = 10):
		tokens = self._preprocess(comments)
		if len(tokens) > 0:
			emb = self._average_vector(tokens)
			similarities = self._vectors @ emb
			if k > self._vectors.shape[0]:
				k = self._vectors.shape[0]
			topk = np.argsort(-1*similarities)[:k]
			# We have to account for variables and descriptions
			chosen_mentions = list()
			for i in topk:
				key = self._keys[i]
				score = similarities[i]
				if key in self._linkable_variables:
					var = key
				else:
					var = self._linkable_descriptions[key]
				for m in self._linkable_variables[var]:
					chosen_mentions.append((m, score))
			return chosen_mentions

			# chosen_mentions = list(it.chain.from_iterable(self._linkable_variables[self._keys[i]] for i in topk))
			# scores = similarities[topk]
			# return [(m, k) for m, k in zip(chosen_mentions, scores)]
		else:
			return []


	@property
	def documents(self):
		return self._docs