import itertools
import json
from pathlib import Path
from typing import Any, Dict, Optional
from collections import defaultdict
from gensim.models import KeyedVectors
from typing import List, Union
import numpy as np
import itertools as it
from transformers import *
import torch


class TextReadingLinker:
    """Encapsulates the logic of the linker to text reading mentions"""

    def __init__(self, mentions_path: str, embeddings_path: Optional[str]):

        # Load the embeddings if embeddings path, otherwise use scibert
        # if embeddings_path:
        #     self._model = KeyedVectors.load(embeddings_path)
        # else:
        if torch.cuda.is_available():
            device = 'cuda:1' # TODO parameterize this
        else:
            device = 'cpu'
        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        self._model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased').to(device).eval()
        

        # Read the mentions
        raw_mentions, tb_mentions, documents = self._read_text_mentions(
            mentions_path
        )
        self._docs = documents
        self._tb_mentions = tb_mentions

        linkable_mentions = [
            m
            for m in raw_mentions
            if any("variable" in a for a in m["arguments"])
        ]
        self._linkable_mentions = linkable_mentions
        self._linkable_variables = defaultdict(list)
        self._linkable_descriptions = dict()

        for m in linkable_mentions:
            arg = m["arguments"]["variable"][0]
            var = arg["text"]
            context = (tuple(m['context'].split()), m['tokenInterval']['start'], m['tokenInterval']['end']-1)
            
            if type(self._model) == KeyedVectors and len(self._preprocess(var)) > 0:
                self._linkable_variables[(var, context)].append(m)
            else:
                self._linkable_variables[(var, context)].append(m)

            # This is a level of indirection in which we also look at the variable descriptions for linking
            if "description" in m["arguments"]:
                desc = m["arguments"]["description"][0]["text"]

                if type(self._model) == KeyedVectors and len(self._preprocess(desc)) > 0:
                    self._linkable_descriptions[
                        (desc, context)
                    ] = var  # We resolve to the variable name, which will use later to do the "graph" linking
                else:
                    self._linkable_descriptions[
                        (desc, context)
                    ] = var  # We resolve to the variable name, which will use later to do the "graph" linking


        # Preprocess the vectors for each mention text
        keys, vectors = list(), list()
        if type(self._model) == KeyedVectors:
            agg_function = lambda t, s, e: self._average_vector(self._preprocess(' '.join(t)))
        else:
            agg_function = self._contextualized_vector

        for (k, ctx) in it.chain(
            self._linkable_variables, self._linkable_descriptions
        ):
            keys.append((k, ctx))
            # TODO make the agg_function ctx aware
            tokens, start, end = ctx
            vectors.append(agg_function(tokens, start, end))

        vectors = np.stack(vectors, axis=0)

        self._keys = keys
        self._vectors = vectors

        # TODO Remove after this is fixed in TR
        self._fix_groundings()

    def _preprocess(self, text: Union[str, List[str]]) -> List[str]:
        """Prepares the text for before fetching embeddings"""
        if type(text) == str:
            text = [text]

        return [
            word
            for word in itertools.chain.from_iterable(
                sent.split() for sent in text
            )
            if word in self._model
        ]

    def _read_text_mentions(self, input_path: str) -> Dict[str, Any]:

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

        input_path = Path(input_path)

        if not input_path.is_dir():
            files =  [input_path]
        else:
            files = input_path.glob("*.json")

        relevant_mentions, text_bound_mentions, docs = list(), list(), dict()

        for file in files:

            doc_name = file.name

            with file.open() as f:
                data = json.load(f)

            local_relevant_mentions = [
                m
                for m in data["mentions"]
                if m["type"] != "TextBoundMention"
                and len(set(m["labels"]) & relevant_labels) > 0
            ]

            # Fix the mention document
            for m in local_relevant_mentions:
                m['document'] = (doc_name, m.get('document', "N/A"))

            local_text_bound_mentions = list(
                it.chain.from_iterable(
                    it.chain.from_iterable(
                        m["arguments"].values() for m in local_relevant_mentions
                    )
                )
            )

            # Add context to the mentions
            local_docs = data["documents"]
            if len(local_docs) > 0:
                for m in local_relevant_mentions:
                    doc = local_docs[m["document"][1]]
                    sent = doc["sentences"][m["sentence"]]
                    # TODO perhaps extend this to a window of text
                    context = " ".join(sent["raw"])
                    m["context"] = context

            relevant_mentions.extend(local_relevant_mentions)
            text_bound_mentions.extend(local_text_bound_mentions)
            docs.update({(doc_name, key):value for key, value in local_docs.items()})

        print(len(relevant_mentions))

        return relevant_mentions, text_bound_mentions, docs

    def _contextualized_vector(self, input_text: Union[List[str], str], start: int = 0, end: int = - 1):
        """Computes the contextualized vector using a bert model for cosine similarity"""

        # Tokenize the input
        if type(input_text) != str:
            if len(input_text) == 1:
                end = start 
            input = self._tokenizer(input_text, is_split_into_words=True, return_tensors='pt').to(self._device)
            # Map word indices to sub word tokens
            start = input.word_to_tokens(start).start
            end = (input.word_to_tokens(end).end)
        else:
            input = self._tokenizer(input_text, return_tensors='pt').to(self._device)
            start, end = 1, -1

        # Forward pass
        output = self._model(**input).last_hidden_state[0, :, :]
        # Select the first and last token of the mention
        first, last = output[start, :].detach().cpu().numpy(), output[end-1, :].detach().cpu().numpy()
        emb = np.concatenate([first, last])

        return emb

    def _average_vector(self, words: List[str]):
        """Precomputes and l2 normalizes the average vector of the requested word embeddings"""

        vectors = self._model[words]
        avg = vectors.mean(axis=0)
        norm = np.linalg.norm(avg)
        normalized_avg = avg / norm
        return normalized_avg

    def align_to_comments(self, comments, threshold=0.5, k=10):

        if type(self._model) == KeyedVectors:
            tokens = self._preprocess(comments)
            if len(tokens) > 0:
                emb = self._average_vector(tokens)
            else:
                return []

        else:
            if len(comments) > 0:
                emb = self._contextualized_vector(comments)
            else:
                return []
            

        similarities = self._vectors @ emb
        similarities = similarities[similarities >= threshold]
        if k > self._vectors.shape[0]:
            k = self._vectors.shape[0]
        topk = np.argsort(-1 * similarities)[:k]
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

    @property
    def documents(self):
        return self._docs

    def _fix_groundings(self):
        """This is a temporary fix to restore the missing groundings on the arguments of the events and relations. This will be obsolete once this issue is fixed in the TR pipeline"""

        # Find the text bound mentions
        tb_mentions = {m["id"]: m for m in self._tb_mentions}

        # Iterate over the arguments and restore the attachments
        for m in self._linkable_mentions:
            if m["type"] != "TextBoundMention":
                for arg in m["arguments"].values():
                    arg = arg[0]
                    id = arg["id"]
                    if id in tb_mentions:
                        tb = tb_mentions[id]
                        arg["attachments"] = tb["attachments"]
