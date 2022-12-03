import itertools
import json
from typing import Any, Union
from collections import defaultdict
from gensim.models import KeyedVectors
import numpy as np

class TextReadingLinker:
	""" Encapsulates the logic of the linker to text reading mentions """

	def __init__(self, mentions_path:str, embeddings_path:str):

		
		# Load the embeddings
		self._model = KeyedVectors.load(embeddings_path)

		# Read the mentions
		raw_mentions = self._read_text_mentions(mentions_path)
		self._mentions = defaultdict(list)
		
		for m in raw_mentions:
			if len(self._preprocess(m['text'])) > 0:
				self._mentions[m['text']].append(m)

		# Preprocess the vectors for each mention text
		keys, vectors = list(), list()
		for k in self._mentions:
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

		return relevant_mentions

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
			chosen_mentions = [self._mentions[self._keys[i]] for i in topk]
			scores = similarities[topk]
			return [(m, k) for m, k in zip(chosen_mentions, scores)]
		else:
			return []